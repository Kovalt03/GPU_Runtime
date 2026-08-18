// What reordering the threads is worth, and what has to be true for it to be.
//
// A warp issues one instruction, so lanes that disagree take turns. REORDER
// regroups a block's threads by a key the kernel supplies, so that the ones
// about to do the same thing share a warp. Nothing a thread holds changes —
// registers and pc travel with it — which is why the answer is identical and
// only the divergence moves.
//
// Two things decide whether it pays, and the tables separate them:
//
//   How scattered.  A key that already lines up with the lanes has nothing to
//                   regroup, and the reorder is pure cost. One scattered across
//                   the block is what it exists for.
//   How much after. Reordering pays for the divergent stretch that follows it.
//                   A long one repays the sort; a short one does not.
//
//   ./build/benchmarks/ser_bench                     result/ser.{md,csv}
//   ./build/benchmarks/ser_bench --machine <file>    latencies come from it
//   ./build/benchmarks/ser_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "isa.hpp"
#include "scheduler.hpp"
#include "thread.hpp"

namespace {

GPUSpec MACHINE;

// Eight warps, which is a tile-sized block and enough that a scattered key has
// somewhere to be gathered from.
constexpr uint32_t WARPS = 8;
constexpr uint32_t THREADS = WARPS * WARP_SIZE;

constexpr uint8_t R_KEY = 1;
constexpr uint8_t R_ONE = 2;
constexpr uint8_t R_WORK = 3;

// if (key != 0) { `length` instructions }, optionally preceded by a reorder.
//
// One arm rather than two, because what SER removes is a warp visiting both: a
// warp whose lanes all skip the body issues the branch and nothing else.
Program divergent_kernel(uint32_t length, bool reordered)
{
    Program p;
    p.push_back(make_v_mov_f32(R_ONE, 1.0f));
    if (reordered) {
        p.push_back(make_reorder(R_KEY));
    }

    const size_t branch = p.size();
    p.push_back(make_bra_div(R_KEY, 0));
    const size_t skip = p.size();
    p.push_back(make_bra(0));

    const size_t body = p.size();
    for (uint32_t i = 0; i < length; ++i) {
        p.push_back(make_v_add_f32(R_WORK, R_WORK, R_ONE));
    }

    const size_t join = p.size();
    p[branch] = make_bra_div(R_KEY, static_cast<int32_t>(body - branch));
    p[skip] = make_bra(static_cast<int32_t>(join - skip));
    p.push_back(make_ret());
    return p;
}

// How the threads that take the branch are spread over the block's warps.
enum class Spread {
    // Every warp holds the same mixture, which is the worst case for divergence
    // and the best for reordering.
    Scattered,

    // The threads that take it are already together. A reorder has nothing to
    // gain and still costs what it costs.
    Coherent,
};

struct Reading {
    uint64_t warp_steps = 0;
    uint64_t active_lane_ops = 0;
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    double divergence = 0.0;
};

Reading measure(uint32_t taking, Spread spread, uint32_t length, bool reordered)
{
    ThreadBlock block = make_block(WARPS, 0);
    for (uint32_t t = 0; t < THREADS; ++t) {
        // Scattered: every 256/taking-th thread, so each warp gets its share.
        // Coherent: the first `taking` threads, which is whole warps at a time.
        const bool takes =
            spread == Spread::Scattered
                ? (taking > 0 && (t * taking) / THREADS != ((t + 1) * taking) / THREADS)
                : t < taking;
        block.warps[t / WARP_SIZE].threads[t % WARP_SIZE].regs[R_KEY] =
            takes ? 1.0f : 0.0f;
    }

    WarpScheduler scheduler;
    scheduler.set_spec(MACHINE);
    scheduler.set_latency_model(LatencyModel::Modelled);

    std::vector<uint8_t> global(64, 0);
    scheduler.run(divergent_kernel(length, reordered), block,
                  DeviceSpan{global.data(), global.size()});

    return Reading{scheduler.stats().warp_steps, scheduler.stats().active_lane_ops,
                   scheduler.stats().weighted_lane_ops, scheduler.stats().cycles,
                   scheduler.stats().divergence_rate()};
}

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
}

struct Row {
    uint32_t taking = 0;
    const char* spread = "";
    uint32_t length = 0;
    Reading plain;
    Reading reordered;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "ser";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] regrouping the threads before they disagree\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());

    std::vector<Row> rows;

    // --- how scattered the key is --------------------------------------------

    std::printf(
        "  A block of %u warps, a %u-instruction branch, by how the takers "
        "are spread\n\n",
        WARPS, 32u);
    std::printf("  %10s %12s %12s %12s %10s %12s %12s %10s\n", "spread", "taking",
                "steps", "reordered", "change", "cycles", "reordered", "change");

    for (const auto& [name, spread] : std::vector<std::pair<const char*, Spread>>{
             {"scattered", Spread::Scattered}, {"coherent", Spread::Coherent}}) {
        for (const uint32_t taking : {32u, 128u, 224u}) {
            const Row row{taking, name, 32, measure(taking, spread, 32, false),
                          measure(taking, spread, 32, true)};
            rows.push_back(row);
            std::printf("  %10s %12u %12llu %12llu %9.1f%% %12llu %12llu %9.1f%%\n", name,
                        taking, (unsigned long long)row.plain.warp_steps,
                        (unsigned long long)row.reordered.warp_steps,
                        change(row.plain.warp_steps, row.reordered.warp_steps),
                        (unsigned long long)row.plain.cycles,
                        (unsigned long long)row.reordered.cycles,
                        change(row.plain.cycles, row.reordered.cycles));
        }
    }

    // --- how much work follows it --------------------------------------------

    std::printf(
        "\n  Half the block taking a scattered branch, by how long the branch is\n\n");
    std::printf("  %10s %12s %12s %10s %12s %12s %10s\n", "instrs", "steps", "reordered",
                "change", "cycles", "reordered", "change");

    for (const uint32_t length : {2u, 8u, 32u, 128u}) {
        const Row row{THREADS / 2, "scattered", length,
                      measure(THREADS / 2, Spread::Scattered, length, false),
                      measure(THREADS / 2, Spread::Scattered, length, true)};
        rows.push_back(row);
        std::printf("  %10u %12llu %12llu %9.1f%% %12llu %12llu %9.1f%%\n", length,
                    (unsigned long long)row.plain.warp_steps,
                    (unsigned long long)row.reordered.warp_steps,
                    change(row.plain.warp_steps, row.reordered.warp_steps),
                    (unsigned long long)row.plain.cycles,
                    (unsigned long long)row.reordered.cycles,
                    change(row.plain.cycles, row.reordered.cycles));
    }

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "ser_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "spread,taking,branch_instructions,steps,reordered_steps,cycles,"
           "reordered_cycles,divergence,reordered_divergence\n";
    for (const Row& r : rows) {
        csv << r.spread << ',' << r.taking << ',' << r.length << ',' << r.plain.warp_steps
            << ',' << r.reordered.warp_steps << ',' << r.plain.cycles << ','
            << r.reordered.cycles << ',' << r.plain.divergence << ','
            << r.reordered.divergence << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/ser_bench; do not edit by hand -->\n\n"
       << "## Regrouping the threads before they disagree\n\n"
       << "A block of " << WARPS
       << " warps and a 32-instruction branch, by how the "
          "threads\ntaking it are spread over the warps.\n\n"
       << "| Spread | Taking | Warp steps | reordered | change | Cycles | reordered | "
          "change |\n"
       << "|---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        if (r.length != 32) {
            continue;
        }
        std::snprintf(
            buf, sizeof(buf),
            "| %s | %u | %llu | %llu | **%.1f%%** | %llu | %llu | **%.1f%%** |\n",
            r.spread, r.taking, (unsigned long long)r.plain.warp_steps,
            (unsigned long long)r.reordered.warp_steps,
            change(r.plain.warp_steps, r.reordered.warp_steps),
            (unsigned long long)r.plain.cycles, (unsigned long long)r.reordered.cycles,
            change(r.plain.cycles, r.reordered.cycles));
        md << buf;
    }

    md << "\n## What has to follow it\n\n"
       << "Half the block taking a scattered branch, by how many instructions are "
          "inside it.\n\n"
       << "| Branch | Warp steps | reordered | change | Cycles | reordered | change |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        if (r.length == 32 || r.taking != THREADS / 2) {
            continue;
        }
        std::snprintf(buf, sizeof(buf),
                      "| %u | %llu | %llu | **%.1f%%** | %llu | %llu | **%.1f%%** |\n",
                      r.length, (unsigned long long)r.plain.warp_steps,
                      (unsigned long long)r.reordered.warp_steps,
                      change(r.plain.warp_steps, r.reordered.warp_steps),
                      (unsigned long long)r.plain.cycles,
                      (unsigned long long)r.reordered.cycles,
                      change(r.plain.cycles, r.reordered.cycles));
        md << buf;
    }

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
