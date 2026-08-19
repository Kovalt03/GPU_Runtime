// One tree over the geometry, another over where the copies of it went.
//
// A scene of the same object many times can be given to a ray tracer two ways.
// Flatten it — transform every vertex of every copy and build one tree over the
// lot — or keep one tree over one copy and a second tree over the placements,
// moving the ray into an instance's space at each leaf of the upper one.
//
// The second is what DXR and Vulkan RT call a TLAS over BLASes, and what this
// measures is that it is a memory answer rather than a speed one. Here, at
// least: an instance visited costs sixteen scalar loads to fetch its matrix,
// because the wide global matrix load the naming scheme reserves does not exist.
//
//   ./build/benchmarks/tlas_bench                     result/tlas.{md,csv}
//   ./build/benchmarks/tlas_bench --triangles 64      geometry per copy
//   ./build/benchmarks/tlas_bench --machine <file>    latencies come from it
//   ./build/benchmarks/tlas_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "bvh.hpp"
#include "gpu_spec.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

std::vector<Float3> geometry(uint32_t triangles, uint32_t seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> place(-0.15f, 0.15f);
    std::uniform_real_distribution<float> size(0.02f, 0.06f);

    std::vector<Float3> world;
    world.reserve(triangles * 3);
    for (uint32_t i = 0; i < triangles; ++i) {
        const Float3 centre{place(rng), place(rng), place(rng)};
        const float s = size(rng);
        world.push_back(centre + Float3{-s, -s, 0.0f});
        world.push_back(centre + Float3{s, -s, 0.0f});
        world.push_back(centre + Float3{0.0f, s, s});
    }
    return world;
}

std::vector<Instance> placements(uint32_t count)
{
    std::vector<Instance> instances;
    for (uint32_t i = 0; i < count; ++i) {
        Float4x4 model = Float4x4::identity();
        model.at(0, 3) = -0.9f + static_cast<float>(i % 16) * 0.12f;
        model.at(1, 3) = static_cast<float>((i / 16) % 8) * 0.12f - 0.4f;
        model.at(2, 3) = static_cast<float>(i / 128) * 0.3f;
        instances.push_back(Instance{model});
    }
    return instances;
}

struct Reading {
    uint64_t work = 0;
    uint64_t issued = 0;
    size_t bytes = 0;
    uint32_t triangles = 0;
};

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

double ratio(uint64_t from, uint64_t to)
{
    return static_cast<double>(from) / static_cast<double>(to);
}

struct Row {
    uint32_t instances = 0;
    Reading flat;
    Reading two_level;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "tlas";
    MACHINE = machine_from(args);
    const uint32_t triangles = args.flag("triangles", 32);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const std::vector<Float3> world = geometry(triangles);

    std::printf("\n[BENCH] one tree over the copies, or one over all of them\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf("  %ux%u, %u triangles a copy\n\n", WIDTH, HEIGHT, triangles);

    std::vector<Row> rows;
    std::printf("  %6s | %6s %13s %9s | %6s %13s %9s | %8s %7s\n", "copies", "tris",
                "flat work", "flat KB", "tris", "two-level", "KB", "memory", "work");
    for (const uint32_t count : {4u, 16u, 64u, 256u}) {
        const std::vector<Instance> instances = placements(count);

        std::vector<Float3> flattened;
        flattened.reserve(world.size() * count);
        for (const Instance& instance : instances) {
            for (const Float3& v : world) {
                const Float4 w = transform(instance.model, v, 1.0f);
                flattened.push_back(Float3{w.x, w.y, w.z});
            }
        }

        Row row;
        row.instances = count;

        {
            MyGPURuntime rt(1u << 28);
            rt.myrt_set_spec(MACHINE);
            DeviceGeometry g = upload_accelerated(rt, flattened);
            DeviceFrame frame = allocate_frame(rt, target);
            draw_raytrace(rt, g, frame, target);
            const Bvh tree = build_bvh(flattened);
            row.flat.work = rt.stats().weighted_lane_ops;
            row.flat.issued = rt.stats().warp_steps;
            row.flat.triangles = g.triangle_count;
            row.flat.bytes =
                flattened.size() * sizeof(Float3) + tree.nodes.size() * sizeof(float);
            release(rt, frame);
            release(rt, g);
        }
        {
            MyGPURuntime rt(1u << 28);
            rt.myrt_set_spec(MACHINE);
            DeviceGeometry g = upload_instanced_accelerated(rt, world, instances);
            DeviceFrame frame = allocate_frame(rt, target);
            draw_raytrace(rt, g, frame, target);

            const Bvh blas = build_bvh(world);
            std::vector<Float4x4> models;
            for (const Instance& instance : instances) {
                models.push_back(instance.model);
            }
            const Tlas tlas = build_tlas(blas, models, BvhSplit::SAH, 1);
            row.two_level.work = rt.stats().weighted_lane_ops;
            row.two_level.issued = rt.stats().warp_steps;
            row.two_level.triangles = g.triangle_count;
            row.two_level.bytes = world.size() * sizeof(Float3) +
                                  blas.nodes.size() * sizeof(float) +
                                  tlas.tree.nodes.size() * sizeof(float) +
                                  tlas.instances.size() * sizeof(float);
            release(rt, frame);
            release(rt, g);
        }
        rows.push_back(row);

        std::printf("  %6u | %6u %13s %9.1f | %6u %13s %9.1f | %7.1fx %6.2fx\n", count,
                    row.flat.triangles, with_commas(row.flat.work).c_str(),
                    row.flat.bytes / 1024.0, row.two_level.triangles,
                    with_commas(row.two_level.work).c_str(), row.two_level.bytes / 1024.0,
                    static_cast<double>(row.flat.bytes) /
                        static_cast<double>(row.two_level.bytes),
                    ratio(row.flat.work, row.two_level.work));
    }

    std::printf(
        "\n  Memory is what the second level buys, and the ratio grows with "
        "the\n  copies: one tree over the geometry however many there are. "
        "Work goes\n  the other way, an instance visited costing sixteen "
        "scalar loads for its\n  matrix — the wide global matrix load the "
        "scheme reserves is not built.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "tlas_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "copies,triangles_a_copy,route,device_triangles,weighted_lane_ops,issued,"
           "device_bytes\n";
    for (const Row& r : rows) {
        const auto line = [&](const char* name, const Reading& x) {
            csv << r.instances << ',' << triangles << ',' << name << ',' << x.triangles
                << ',' << x.work << ',' << x.issued << ',' << x.bytes << '\n';
        };
        line("flattened", r.flat);
        line("two-level", r.two_level);
    }

    std::ofstream md(prefix + ".md");
    char buf[400];
    md << "<!-- generated by benchmarks/tlas_bench; do not edit by hand -->\n\n"
       << "## One tree over the copies, or one over all of them\n\n"
       << WIDTH << "x" << HEIGHT << ", " << triangles
       << " triangles a copy. Flattened means every vertex of every copy "
          "transformed on\nthe host and given one tree; two-level keeps one tree "
          "over one copy and walks a\nsecond tree over the placements.\n\n"
       << "| Copies | Flat triangles | Flat work | Flat KB | Two-level triangles | "
          "Work | KB | Memory | Work |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(
            buf, sizeof(buf),
            "| %u | %u | %s | %.1f | %u | %s | %.1f | **%.1fx** | %.2fx |\n", r.instances,
            r.flat.triangles, with_commas(r.flat.work).c_str(), r.flat.bytes / 1024.0,
            r.two_level.triangles, with_commas(r.two_level.work).c_str(),
            r.two_level.bytes / 1024.0,
            static_cast<double>(r.flat.bytes) / static_cast<double>(r.two_level.bytes),
            ratio(r.flat.work, r.two_level.work));
        md << buf;
    }
    md << "\nMemory is what the second level buys and the ratio grows with the "
          "copies: one\ntree over the geometry however many there are. Work goes "
          "the other way, and by\na widening margin — an instance visited costs "
          "sixteen scalar loads to fetch its\nmatrix, the wide global matrix load "
          "the naming scheme reserves not being built.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
