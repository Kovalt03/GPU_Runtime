// What several SMs buy, and what stops a kernel from using them.
//
// Every other benchmark here measures a kernel against another kernel on one
// machine. This one measures one kernel against several machines: the same
// blocks, spread differently.
//
// Three questions, and the third is the one that could not be asked before
// residency existed:
//
//   Parallelism.  More SMs finish sooner. The issued work must not move, or the
//                 accounting is wrong rather than the machine fast.
//   Occupancy.    More blocks on an SM cover each other's waiting. That is the
//                 same mechanism [m3] found among warps, one level up.
//   Refusal.      A kernel that declares the whole scratchpad cannot hold two
//                 blocks whatever the machine allows, so occupancy is not a knob
//                 every kernel can turn.
//
//   ./build/benchmarks/occupancy_bench                     result/occupancy.{md,csv}
//   ./build/benchmarks/occupancy_bench --machine <file>    the ceiling comes from it
//   ./build/benchmarks/occupancy_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"
#include "scenes.hpp"

namespace {

// The machine this run was asked for, which the sweeps vary from.
GPUSpec MACHINE;

// Cycles only mean anything with latency modelled, and a cache is what makes a
// second block's read cheaper than a first's — both sweeps want both.
struct Reading {
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    uint64_t stalls = 0;
    uint64_t l1_hits = 0;
    uint64_t l2_hits = 0;
};

using Route = std::vector<Float3> (*)(MyGPURuntime&, const DeviceGeometry&,
                                      const DeviceFrame&, const DrawTarget&, bool,
                                      const Shading&);

Reading measure(Route route, const std::vector<Float3>& world, uint32_t sm_count,
                uint32_t blocks_per_sm)
{
    MyGPURuntime rt(1u << 27);
    GPUSpec spec = MACHINE;
    spec.sms.sm_count = sm_count;
    spec.sms.blocks_per_sm = blocks_per_sm;
    rt.myrt_set_spec(spec);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, bench_target());
    route(rt, geometry, frame, bench_target(), false, Shading{});

    const Reading r{rt.stats().weighted_lane_ops, rt.stats().cycles,
                    rt.stats().stall_steps, rt.stats().l1_hits, rt.stats().l2_hits};
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

struct Row {
    uint32_t sm_count = 0;
    uint32_t blocks_per_sm = 0;
    const char* route = "";
    Reading reading;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "occupancy";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    // Sixteen small triangles over the frame: the walk's blocks all read them,
    // which is what makes an SM's L1 worth sharing.
    const std::vector<Float3> world = spread(4);
    const uint32_t blocks = (BENCH_WIDTH / WARP_SIZE) * BENCH_HEIGHT;

    std::printf("\n[BENCH] what several SMs buy — %ux%u, %u blocks of one warp\n\n",
                BENCH_WIDTH, BENCH_HEIGHT, blocks);
    std::printf("%s\n", MACHINE.describe().c_str());

    // --- parallelism against occupancy ---------------------------------------

    std::vector<Row> grid;
    for (const uint32_t sms : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
        for (const uint32_t per_sm : {1u, 8u}) {
            grid.push_back(
                Row{sms, per_sm, "walk", measure(draw_walk, world, sms, per_sm)});
        }
    }

    const uint64_t base = grid.front().reading.cycles;
    std::printf("  %5s %10s %14s %14s %9s %12s\n", "SMs", "blocks/SM", "issued", "cycles",
                "vs 1x1", "stalls");
    for (const Row& r : grid) {
        std::printf("  %5u %10u %14s %14s %8.2fx %12s\n", r.sm_count, r.blocks_per_sm,
                    with_commas(r.reading.weighted).c_str(),
                    with_commas(r.reading.cycles).c_str(),
                    static_cast<double>(base) / static_cast<double>(r.reading.cycles),
                    with_commas(r.reading.stalls).c_str());
    }

    // --- what each route can hold --------------------------------------------

    std::printf(
        "\n  Occupancy on one SM, by route. shared declares the whole "
        "scratchpad and so holds one block whatever is asked\n\n");
    std::printf("  %10s %14s %14s %14s\n", "blocks/SM", "walk", "tiled", "shared");

    std::vector<Row> routes;
    for (const uint32_t per_sm : {1u, 2u, 4u, 8u, 16u, 32u}) {
        const Reading walk = measure(draw_walk, world, 1, per_sm);
        const Reading tiled = measure(draw_tiled, world, 1, per_sm);
        const Reading shared = measure(draw_shared, world, 1, per_sm);
        routes.push_back(Row{1, per_sm, "walk", walk});
        routes.push_back(Row{1, per_sm, "tiled", tiled});
        routes.push_back(Row{1, per_sm, "shared", shared});
        std::printf("  %10u %14s %14s %14s\n", per_sm, with_commas(walk.cycles).c_str(),
                    with_commas(tiled.cycles).c_str(),
                    with_commas(shared.cycles).c_str());
    }

    // --- what co-residency does to a cache -----------------------------------

    // Scaled down until the scene outgrows it, because at the hardware size
    // nothing is ever evicted and blocks sharing an SM have nothing to fight
    // over — or to hand each other.
    std::printf("\n  Blocks sharing an SM share its L1, at L1 %u lines\n\n", 16u);
    std::printf("  %8s %10s %12s %12s %14s\n", "route", "blocks/SM", "L1 hits", "L2 hits",
                "issued");

    std::vector<Row> sharing;
    for (const auto& [name, route] : std::vector<std::pair<const char*, Route>>{
             {"walk", draw_walk}, {"tiled", draw_tiled}}) {
        for (const uint32_t per_sm : {1u, 8u}) {
            GPUSpec scaled = MACHINE;
            scaled.l1_lines = 16;
            scaled.l2_lines = 512;
            const GPUSpec keep = MACHINE;
            MACHINE = scaled;
            const Reading r = measure(route, spread(8), 1, per_sm);
            MACHINE = keep;

            sharing.push_back(Row{1, per_sm, name, r});
            std::printf("  %8s %10u %12s %12s %14s\n", name, per_sm,
                        with_commas(r.l1_hits).c_str(), with_commas(r.l2_hits).c_str(),
                        with_commas(r.weighted).c_str());
        }
    }

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "occupancy_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "table,route,sm_count,blocks_per_sm,weighted,cycles,stalls,l1_hits,l2_hits\n";
    const auto rows_to_csv = [&csv](const char* table, const std::vector<Row>& rows) {
        for (const Row& r : rows) {
            csv << table << ',' << r.route << ',' << r.sm_count << ',' << r.blocks_per_sm
                << ',' << r.reading.weighted << ',' << r.reading.cycles << ','
                << r.reading.stalls << ',' << r.reading.l1_hits << ','
                << r.reading.l2_hits << '\n';
        }
    };
    rows_to_csv("grid", grid);
    rows_to_csv("routes", routes);
    rows_to_csv("sharing", sharing);

    std::ofstream md(prefix + ".md");
    char buf[256];
    md << "<!-- generated by benchmarks/occupancy_bench; do not edit by hand -->\n\n"
       << "## What several SMs buy\n\n"
       << "`draw_walk` over " << blocks << " blocks of one warp, with a cache and "
       << "latency modelled.\n\n"
       << "| SMs | blocks an SM | issued work | cycles | vs 1x1 | stalls |\n"
       << "|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : grid) {
        std::snprintf(buf, sizeof(buf), "| %u | %u | %s | %s | %.2fx | %s |\n",
                      r.sm_count, r.blocks_per_sm,
                      with_commas(r.reading.weighted).c_str(),
                      with_commas(r.reading.cycles).c_str(),
                      static_cast<double>(base) / static_cast<double>(r.reading.cycles),
                      with_commas(r.reading.stalls).c_str());
        md << buf;
    }

    md << "\n## Occupancy on one SM, by route\n\n"
       << "Cycles, so lower is better. `shared` declares the whole scratchpad, so "
          "its\nresidency is one block whatever the machine allows.\n\n"
       << "| blocks an SM | walk | tiled | shared |\n|---:|---:|---:|---:|\n";
    for (size_t i = 0; i < routes.size(); i += 3) {
        std::snprintf(buf, sizeof(buf), "| %u | %s | %s | %s |\n",
                      routes[i].blocks_per_sm,
                      with_commas(routes[i].reading.cycles).c_str(),
                      with_commas(routes[i + 1].reading.cycles).c_str(),
                      with_commas(routes[i + 2].reading.cycles).c_str());
        md << buf;
    }

    md << "\n## Blocks sharing an SM share its L1\n\n"
       << "L1 scaled to 16 lines, which the scene outgrows — at the hardware size "
          "nothing\nis evicted and there is nothing to hand over.\n\n"
       << "| Route | blocks an SM | L1 hits | L2 hits | issued work |\n"
       << "|---|---:|---:|---:|---:|\n";
    for (const Row& r : sharing) {
        std::snprintf(buf, sizeof(buf), "| %s | %u | %s | %s | %s |\n", r.route,
                      r.blocks_per_sm, with_commas(r.reading.l1_hits).c_str(),
                      with_commas(r.reading.l2_hits).c_str(),
                      with_commas(r.reading.weighted).c_str());
        md << buf;
    }

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
