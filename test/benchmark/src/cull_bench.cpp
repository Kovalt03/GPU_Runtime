// The device deciding how much to draw, on something worth deciding about.
//
// stream_bench measured an indirect launch with a culling pass in front of it,
// and that pass walked its candidates with one thread and a loop — 810 cycles of
// 900. It was written that way because there was nothing to cull: a flag buffer
// stood in for a scene, and widening the walk would have been widening a
// placeholder. Instancing supplied the thing it was standing in for.
//
// Two questions:
//
//   1. what culling removes, against drawing everything
//   2. what having to learn the number costs — a host that reads the count back
//      has to synchronise, and a synchronisation in the middle of a frame stops
//      whatever else was running
//
//   ./build/benchmarks/cull_bench                     result/cull.{md,csv}
//   ./build/benchmarks/cull_bench --instances 256     how many candidates
//   ./build/benchmarks/cull_bench --machine <file>    latencies come from it
//   ./build/benchmarks/cull_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "bvh.hpp"
#include "gpu_spec.hpp"
#include "pipeline/cull.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/vertex.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// Vertices an instance. Enough that pass 1 is worth more than the cull, which is
// the condition under which culling is worth anything at all.
constexpr uint32_t VERTICES = 96;

// A row of boxes, spread so that `spread` controls how much of it the camera
// sees. One is roughly the whole row on screen; larger pushes the ends out.
std::vector<Box> candidates(uint32_t count, float spread)
{
    std::vector<Box> boxes;
    for (uint32_t i = 0; i < count; ++i) {
        const float t =
            (static_cast<float>(i) / static_cast<float>(count) - 0.5f) * 2.0f * spread;
        const float y = (static_cast<float>(i % 5) - 2.0f) * 0.3f;
        boxes.push_back(
            Box{Float3{t - 0.08f, y - 0.08f, 0.0f}, Float3{t + 0.08f, y + 0.08f, 0.05f}});
    }
    return boxes;
}

std::vector<Float3> geometry()
{
    std::vector<Float3> world;
    for (uint32_t i = 0; i < VERTICES; ++i) {
        world.push_back(Float3{static_cast<float>(i % 8) * 0.02f - 0.08f,
                               static_cast<float>(i % 5) * 0.03f - 0.06f, 0.0f});
    }
    return world;
}

struct Reading {
    uint64_t work = 0;
    uint64_t cycles = 0;
    uint32_t drawn = 0;
};

// One run. `decide_on_device` picks whether the grid comes from the cull or from
// the host having read the count back.
//
// The other stream carries independent work, which is what makes a
// synchronisation cost anything: with nothing else to run, waiting is free and
// the comparison would say so.
Reading frame(const std::vector<Box>& boxes, const Frustum& frustum, bool cull_at_all,
              bool decide_on_device, uint32_t other_stream_blocks)
{
    const uint32_t count = static_cast<uint32_t>(boxes.size());
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const std::vector<Float3> world = geometry();

    std::vector<float> box_floats;
    std::vector<float> matrices;
    for (const Box& box : boxes) {
        for (float v : {box.lo.x, box.lo.y, box.lo.z, box.hi.x, box.hi.y, box.hi.z}) {
            box_floats.push_back(v);
        }
        Float4x4 m = Float4x4::identity();
        m.at(0, 3) = (box.lo.x + box.hi.x) * 0.5f;
        m.at(1, 3) = (box.lo.y + box.hi.y) * 0.5f;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                matrices.push_back(m.at(r, c));
            }
        }
    }

    MyGPURuntime rt(1u << 27);
    rt.myrt_set_spec(MACHINE);

    void* device_boxes = rt.myrt_malloc(box_floats.size() * sizeof(float));
    void* device_matrices = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* survivors = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    void* device_world = rt.myrt_malloc(world.size() * sizeof(Float3));
    void* screen = rt.myrt_malloc(instanced_screen_bytes(VERTICES, count));
    void* scratch = rt.myrt_malloc(other_stream_blocks * WARP_SIZE * sizeof(float));

    rt.myrt_memcpy(device_boxes, box_floats.data(), box_floats.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(device_matrices, matrices.data(), matrices.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(device_world, world.data(), world.size() * sizeof(Float3),
                   Direction::HostToDevice);
    const float start[3] = {(VERTICES + WARP_SIZE - 1) / WARP_SIZE, 0.0f, 1.0f};
    rt.myrt_memcpy(grid, start, sizeof(start), Direction::HostToDevice);

    // Independent work on a second queue, so that a wait has something to stop.
    const size_t scratch_at = rt.myrt_device_offset(scratch);
    const auto filler = [scratch_at](void**) {
        IRBuilder k;
        const Reg<Scalar> at = k.add(k.constant(static_cast<float>(scratch_at)),
                                     k.mul(k.thread_x(), k.constant(4.0f)));
        Reg<Scalar> v = k.constant(1.0f);
        for (uint32_t i = 0; i < 64; ++i) {
            k.fma(v, v, k.constant(1.0f));
        }
        k.store(at, v);
        return k.build();
    };
    const StreamId elsewhere = rt.myrt_stream_create();
    LaunchConfig other{dim3{other_stream_blocks, 1, 1}, dim3{WARP_SIZE, 1, 1}};
    rt.myrt_launch_async(filler, other, nullptr, elsewhere);

    VertexStageArgs pass1;
    pass1.view_projection = target.camera.view_projection(target.aspect());
    pass1.world_offset = rt.myrt_device_offset(device_world);
    pass1.screen_offset = rt.myrt_device_offset(screen);
    pass1.vertex_count = VERTICES;
    pass1.width = WIDTH;
    pass1.height = HEIGHT;
    pass1.instance_count = count;

    uint32_t drawn = count;
    if (!cull_at_all) {
        // Everything, whether or not the camera can see it.
        pass1.instance_offset = rt.myrt_device_offset(device_matrices);
        pass1.uniform_offset = pass1.instance_offset;
        run_vertex_stage(rt, pass1);
    } else {
        CullStageArgs cull;
        cull.frustum = frustum;
        cull.boxes_offset = rt.myrt_device_offset(device_boxes);
        cull.matrices_offset = rt.myrt_device_offset(device_matrices);
        cull.survivors_offset = rt.myrt_device_offset(survivors);
        cull.grid_offset = rt.myrt_device_offset(grid);
        cull.instance_count = count;
        run_cull_stage(rt, cull);

        pass1.instance_offset = rt.myrt_device_offset(survivors);
        pass1.uniform_offset = pass1.instance_offset;

        if (decide_on_device) {
            // Enqueued behind the cull. Nothing outside the device learns the
            // number, and the other queue keeps running throughout.
            run_vertex_stage_indirect(rt, pass1, cull.grid_offset);
        } else {
            // The host has to read the count, which means draining everything
            // first — including the queue that had nothing to do with it.
            rt.myrt_wait();
            float wrote[3] = {0.0f, 0.0f, 0.0f};
            rt.myrt_memcpy(wrote, grid, sizeof(wrote), Direction::DeviceToHost);
            drawn = static_cast<uint32_t>(wrote[1]);
            pass1.instance_count = drawn;
            run_vertex_stage(rt, pass1);
        }
    }
    rt.myrt_wait();

    Reading r;
    r.work = rt.stats().weighted_lane_ops;
    r.cycles = rt.stats().cycles;
    if (cull_at_all && decide_on_device) {
        float wrote[3] = {0.0f, 0.0f, 0.0f};
        rt.myrt_memcpy(wrote, grid, sizeof(wrote), Direction::DeviceToHost);
        drawn = static_cast<uint32_t>(wrote[1]);
    }
    r.drawn = drawn;
    return r;
}

std::string with_commas(uint64_t v)
{
    std::string digits = std::to_string(v);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(digits[i]);
    }
    return out;
}

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
}

struct Row {
    float spread = 0.0f;
    uint32_t drawn = 0;
    uint32_t total = 0;
    Reading everything;
    Reading host_decides;
    Reading device_decides;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "cull";
    MACHINE = machine_from(args);
    const uint32_t count = args.flag("instances", 64);
    const uint32_t other = args.flag("other-blocks", 8);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const Frustum frustum = frustum_of(target.camera.view_projection(target.aspect()));

    std::printf("\n[BENCH] the device deciding how much to draw\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf(
        "  %u candidates, %u vertices each, %u blocks of unrelated work on a "
        "second queue\n\n",
        count, VERTICES, other);

    std::vector<Row> rows;
    std::printf("  what culling removes\n");
    std::printf("  %8s | %11s | %11s %11s | %s\n", "visible", "drawn", "no cull",
                "culled", "change");
    for (const float spread : {1.0f, 2.0f, 4.0f, 8.0f}) {
        const std::vector<Box> boxes = candidates(count, spread);
        Row row;
        row.spread = spread;
        row.total = count;
        row.everything = frame(boxes, frustum, false, false, other);
        row.host_decides = frame(boxes, frustum, true, false, other);
        row.device_decides = frame(boxes, frustum, true, true, other);
        row.drawn = row.device_decides.drawn;
        rows.push_back(row);

        std::printf("  %7.0f%% | %5u of %3u | %11s %11s | %+8.1f%%\n",
                    100.0 * row.drawn / count, row.drawn, count,
                    with_commas(row.everything.cycles).c_str(),
                    with_commas(row.device_decides.cycles).c_str(),
                    change(row.everything.cycles, row.device_decides.cycles));
    }

    // The second question is not about the scene at all. Reading the count back
    // means draining every queue, so what the synchronisation costs is whatever
    // the other one still had to do — swept here rather than fixed, since a
    // single figure would look like a property of culling.
    const std::vector<Box> half = candidates(count, 4.0f);
    std::printf("\n  what deciding on the device removes\n");
    std::printf("  %10s | %13s %13s | %s\n", "other work", "host reads it",
                "device decides", "change");
    std::vector<Row> syncs;
    for (const uint32_t blocks : {1u, 4u, 16u, 64u, 256u}) {
        Row row;
        row.total = blocks;
        row.host_decides = frame(half, frustum, true, false, blocks);
        row.device_decides = frame(half, frustum, true, true, blocks);
        row.drawn = row.device_decides.drawn;
        syncs.push_back(row);

        std::printf("  %6u blk | %13s %13s | %+8.1f%%\n", blocks,
                    with_commas(row.host_decides.cycles).c_str(),
                    with_commas(row.device_decides.cycles).c_str(),
                    change(row.host_decides.cycles, row.device_decides.cycles));
    }

    std::printf(
        "\n  Culling is worth what it removes from the draw. Deciding on "
        "the device is\n  worth the synchronisation it avoids, which is "
        "whatever the other queue\n  still had to do — nothing at all when "
        "it had nothing.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "cull_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "table,candidates,drawn,other_blocks,route,weighted_lane_ops,cycles\n";
    for (const Row& r : rows) {
        const auto line = [&](const char* name, const Reading& x) {
            csv << "culling," << count << ',' << r.drawn << ',' << other << ',' << name
                << ',' << x.work << ',' << x.cycles << '\n';
        };
        line("no-cull", r.everything);
        line("host-grid", r.host_decides);
        line("device-grid", r.device_decides);
    }
    for (const Row& r : syncs) {
        const auto line = [&](const char* name, const Reading& x) {
            csv << "sync," << count << ',' << r.drawn << ',' << r.total << ',' << name
                << ',' << x.work << ',' << x.cycles << '\n';
        };
        line("host-grid", r.host_decides);
        line("device-grid", r.device_decides);
    }

    std::ofstream md(prefix + ".md");
    char buf[400];
    md << "<!-- generated by benchmarks/cull_bench; do not edit by hand -->\n\n"
       << "## The device deciding how much to draw\n\n"
       << count << " candidates, " << VERTICES << " vertices each, " << other
       << " blocks of unrelated work on a second queue.\nCycles, so that a "
          "synchronisation costs what it stops.\n\n"
       << "| Visible | Drawn | No cull | Culled | Change |\n"
       << "|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %.0f%% | %u of %u | %s | %s | **%+.1f%%** |\n",
                      100.0 * r.drawn / r.total, r.drawn, r.total,
                      with_commas(r.everything.cycles).c_str(),
                      with_commas(r.device_decides.cycles).c_str(),
                      change(r.everything.cycles, r.device_decides.cycles));
        md << buf;
    }
    md << "\n### What deciding on the device removes\n\n"
       << "Reading the count back means draining every queue, so the "
          "synchronisation\ncosts whatever the other one still had to do.\n\n"
       << "| Other work | Host reads the count | Device decides | Change |\n"
       << "|---:|---:|---:|---:|\n";
    for (const Row& r : syncs) {
        std::snprintf(buf, sizeof(buf), "| %u blocks | %s | %s | **%+.1f%%** |\n",
                      r.total, with_commas(r.host_decides.cycles).c_str(),
                      with_commas(r.device_decides.cycles).c_str(),
                      change(r.host_decides.cycles, r.device_decides.cycles));
        md << buf;
    }
    md << "\nNothing at all when the other queue had nothing, which is the honest "
          "shape of\nit: an indirect launch buys the overlap a synchronisation "
          "would have thrown away.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
