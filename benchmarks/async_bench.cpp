// What issuing a copy and not waiting for it buys.
//
// cp.async moves global memory into shared memory without a register in
// between and without the warp waiting, and a kernel meets it later at
// S_CP_ASYNC_WAIT. Two questions, and they do not have the same answer:
//
//   The mechanism.  A fill on its own, sixteen floats or sixty-four, with every
//                   float in a line of its own so every one of them misses. This
//                   is where the instruction is worth what it claims.
//   The renderer.   The same mechanism inside the shared-memory raster route,
//                   which stages a tile and then walks it. Here it is worth
//                   almost nothing, and the third table says why.
//
//   ./build/benchmarks/async_bench                     result/async.{md,csv}
//   ./build/benchmarks/async_bench --machine <file>    caches and latencies from it
//   ./build/benchmarks/async_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "ir_builder.hpp"
#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"
#include "scenes.hpp"

namespace {

GPUSpec MACHINE;

struct Reading {
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    uint64_t stalls = 0;
};

// A fill and nothing else: `floats` floats from global into shared memory, one
// warp, each float in a cache line of its own so that none of them is found.
Program fill_kernel(uint32_t floats, bool asynchronous)
{
    IRBuilder k;
    const Reg<Scalar> source = k.constant(0.0f);
    const Reg<Scalar> destination = k.constant(0.0f);
    for (uint32_t i = 0; i < floats; ++i) {
        const float from = static_cast<float>(i * CACHE_LINE_BYTES);
        k.set(destination, static_cast<float>(i * sizeof(float)));
        if (asynchronous) {
            k.cp_async(destination, source, from);
        } else {
            k.store_shared(destination, k.load(source, from));
        }
    }
    if (asynchronous) {
        k.cp_async_wait(0);
    }
    k.barrier();
    return k.build();
}

MyGPURuntime machine(size_t bytes)
{
    MyGPURuntime rt(bytes);
    rt.myrt_set_spec(MACHINE);
    rt.myrt_set_memory_model(MemoryModel::Cached);

    // Without latency there is nothing to hide, and the instruction is then only
    // the shared store it saves.
    rt.myrt_set_latency_model(LatencyModel::Modelled);
    return rt;
}

Reading measure_fill(uint32_t floats, bool asynchronous)
{
    MyGPURuntime rt = machine(1u << 22);
    rt.myrt_malloc(64 * 1024);
    const Program prog = fill_kernel(floats, asynchronous);
    rt.myrt_launch([prog](void**) { return prog; }, dim3{1, 1, 1}, dim3{WARP_SIZE, 1, 1},
                   nullptr);
    return Reading{rt.stats().weighted_lane_ops, rt.stats().cycles,
                   rt.stats().stall_steps};
}

Reading measure_route(const std::vector<Float3>& world, bool asynchronous)
{
    MyGPURuntime rt = machine(1u << 28);
    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, bench_target());
    draw_shared(rt, geometry, frame, bench_target(), false, Shading{}, asynchronous);
    const Reading r{rt.stats().weighted_lane_ops, rt.stats().cycles,
                    rt.stats().stall_steps};
    release(rt, frame);
    release(rt, geometry);
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

double change(uint64_t before, uint64_t after)
{
    return 100.0 * (static_cast<double>(after) - static_cast<double>(before)) /
           static_cast<double>(before);
}

struct FillRow {
    uint32_t floats = 0;
    Reading sync;
    Reading async;
};

struct RouteRow {
    const char* scene = "";
    uint32_t triangles = 0;
    Reading sync;
    Reading async;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "async";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] a copy the warp does not wait for\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());

    // --- the mechanism -------------------------------------------------------

    std::printf("  A fill on its own, one warp, a line a float\n\n");
    std::printf("  %8s %12s %12s %10s %12s %12s %10s\n", "floats", "sync issued",
                "async issued", "change", "sync cycles", "async cycles", "change");

    std::vector<FillRow> fills;
    for (const uint32_t floats : {8u, 16u, 32u, 64u}) {
        const FillRow row{floats, measure_fill(floats, false),
                          measure_fill(floats, true)};
        fills.push_back(row);
        std::printf("  %8u %12s %12s %9.1f%% %12s %12s %9.1f%%\n", floats,
                    with_commas(row.sync.weighted).c_str(),
                    with_commas(row.async.weighted).c_str(),
                    change(row.sync.weighted, row.async.weighted),
                    with_commas(row.sync.cycles).c_str(),
                    with_commas(row.async.cycles).c_str(),
                    change(row.sync.cycles, row.async.cycles));
    }

    // --- the renderer --------------------------------------------------------

    std::printf(
        "\n  The same mechanism in the shared-memory raster route, which stages a "
        "tile and then walks it\n\n");
    std::printf("  %-26s %10s %14s %14s %10s %12s %12s %10s\n", "scene", "triangles",
                "sync issued", "async issued", "change", "sync cycles", "async cycles",
                "change");

    std::vector<RouteRow> routes;
    for (const auto& [name, world] :
         std::vector<std::pair<const char*, std::vector<Float3>>>{
             {"small, spread", spread(4)},
             {"full-frame, stacked", stacked(16, 1.0f)},
             {"full-frame, stacked", stacked(64, 1.0f)},
             {"full-frame, stacked", stacked(128, 1.0f)}}) {
        const auto triangles = static_cast<uint32_t>(world.size() / 3);
        const RouteRow row{name, triangles, measure_route(world, false),
                           measure_route(world, true)};
        routes.push_back(row);
        std::printf("  %-26s %10u %14s %14s %9.1f%% %12s %12s %9.1f%%\n", name, triangles,
                    with_commas(row.sync.weighted).c_str(),
                    with_commas(row.async.weighted).c_str(),
                    change(row.sync.weighted, row.async.weighted),
                    with_commas(row.sync.cycles).c_str(),
                    with_commas(row.async.cycles).c_str(),
                    change(row.sync.cycles, row.async.cycles));
    }

    // --- why the two tables disagree -----------------------------------------

    // A staged float is fetched once and read by every thread of the block that
    // walks the triangle it belongs to. The ratio is a property of the kernel
    // rather than of the scene, and it is what decides whether hiding the fetch
    // can matter at all.
    std::printf(
        "\n  A staged float is fetched once and read %u times — once by each of "
        "the block's threads.\n  Hiding the fetch can save at most that fraction "
        "of the walk.\n",
        TILE_BLOCK_THREADS);

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "async_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "table,scene,size,sync_issued,async_issued,sync_cycles,async_cycles,"
           "sync_stalls,async_stalls\n";
    for (const FillRow& r : fills) {
        csv << "fill,fill," << r.floats << ',' << r.sync.weighted << ','
            << r.async.weighted << ',' << r.sync.cycles << ',' << r.async.cycles << ','
            << r.sync.stalls << ',' << r.async.stalls << '\n';
    }
    for (const RouteRow& r : routes) {
        csv << "route,\"" << r.scene << "\"," << r.triangles << ',' << r.sync.weighted
            << ',' << r.async.weighted << ',' << r.sync.cycles << ',' << r.async.cycles
            << ',' << r.sync.stalls << ',' << r.async.stalls << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[256];
    md << "<!-- generated by benchmarks/async_bench; do not edit by hand -->\n\n"
       << "## A fill on its own\n\n"
       << "One warp, a cache line a float, so every float misses.\n\n"
       << "| Floats | sync issued | async issued | change | sync cycles | async cycles "
          "| change |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const FillRow& r : fills) {
        std::snprintf(
            buf, sizeof(buf), "| %u | %s | %s | %.1f%% | %s | %s | **%.1f%%** |\n",
            r.floats, with_commas(r.sync.weighted).c_str(),
            with_commas(r.async.weighted).c_str(),
            change(r.sync.weighted, r.async.weighted), with_commas(r.sync.cycles).c_str(),
            with_commas(r.async.cycles).c_str(), change(r.sync.cycles, r.async.cycles));
        md << buf;
    }

    md << "\n## The same mechanism in a renderer\n\n"
       << "`draw_shared`, which stages a tile into shared memory and then walks it. "
          "The\nasynchronous form takes the tile a chunk at a time and issues the next "
          "chunk's\ncopies before walking this one.\n\n"
       << "| Scene | triangles | sync issued | async issued | change | sync cycles | "
          "async cycles | change |\n"
       << "|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const RouteRow& r : routes) {
        std::snprintf(
            buf, sizeof(buf), "| %s | %u | %s | %s | %.1f%% | %s | %s | **%.1f%%** |\n",
            r.scene, r.triangles, with_commas(r.sync.weighted).c_str(),
            with_commas(r.async.weighted).c_str(),
            change(r.sync.weighted, r.async.weighted), with_commas(r.sync.cycles).c_str(),
            with_commas(r.async.cycles).c_str(), change(r.sync.cycles, r.async.cycles));
        md << buf;
    }
    md << "\nA staged float is fetched once and read by each of the block's "
       << TILE_BLOCK_THREADS << " threads.\nHiding the fetch can save at most that "
       << "fraction of the walk, whatever the scene.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
