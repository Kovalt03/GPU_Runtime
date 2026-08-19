// One instruction against 4,096 multiply-adds.
//
// V_MMA_16X16X16_F32 has the warp compute a whole 16x16x16 product at once,
// where the arithmetic pipe does it one fused multiply-add at a time. The
// comparison is not close, and that is exactly why the interesting figure is not
// the ratio but the price:
//
// The cost of the matrix instruction is a claim, not a measurement — there is no
// hardware here to time, and a table weighted by that claim would only report it
// back. So the question is turned around, as reduction_bench does for the lane
// exchange: how expensive would one of these have to be before the two routes
// came level? Everything else here is counted rather than priced — instructions
// issued, warp steps, cycles.
//
//   ./build/benchmarks/mma_bench                     result/mma.{md,csv}
//   ./build/benchmarks/mma_bench --machine <file>    latencies come from it
//   ./build/benchmarks/mma_bench --out dir           writes into that directory

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

constexpr uint32_t ALL_LANES = 0xFFFFFFFFu;

// Registers: A and B fragments, then the accumulator.
constexpr uint8_t R_A = 0;
constexpr uint8_t R_B = 8;
constexpr uint8_t R_D = 16;

GPUSpec MACHINE;

// tiles products of 16x16x16, accumulated into one fragment. This is the shape
// of a matrix multiply's inner loop: the same accumulator, a new pair of
// operands each time round.
Program matrix_unit(uint32_t tiles)
{
    Program p;
    for (uint32_t i = 0; i < tiles; ++i) {
        p.push_back(make_v_mma_16x16x16_f32(R_D, R_A, R_B, ALL_LANES));
    }
    p.push_back(make_ret());
    return p;
}

// The same work through the arithmetic pipe. Each lane owns eight elements of
// the accumulator and each of those is a dot product of length 16, so a tile is
// 128 fused multiply-adds a lane.
//
// The operands are not gathered: a lane holding an eighth of A cannot see the
// row it needs, and fetching it would be shuffles this comparison is not about.
// What is counted is the multiply-adds themselves, which is the floor the matrix
// instruction has to beat and rather better than the arithmetic route could
// really do.
Program arithmetic_pipe(uint32_t tiles)
{
    Program p;
    for (uint32_t i = 0; i < tiles * MMA_FRAGMENT_REGISTERS * MMA_TILE; ++i) {
        p.push_back(make_v_fma_f32(R_D, R_A, R_B));
    }
    p.push_back(make_ret());
    return p;
}

struct Reading {
    size_t instructions = 0;
    uint64_t warp_steps = 0;
    uint64_t weighted = 0;
    uint64_t cycles = 0;
};

Reading measure(const Program& p)
{
    ThreadBlock block = make_block(1);
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        for (uint32_t i = 0; i < MMA_FRAGMENT_REGISTERS; ++i) {
            block.warps[0].threads[lane].regs[R_A + i] = 1.0f;
            block.warps[0].threads[lane].regs[R_B + i] = 1.0f;
        }
    }

    WarpScheduler scheduler;
    scheduler.set_spec(MACHINE);
    scheduler.set_latency_model(LatencyModel::Modelled);

    std::vector<uint8_t> global(64, 0);
    scheduler.run(p, block, DeviceSpan{global.data(), global.size()});

    return Reading{p.size(), scheduler.stats().warp_steps,
                   scheduler.stats().weighted_lane_ops, scheduler.stats().cycles};
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
    uint32_t tiles = 0;
    Reading fma;
    Reading mma;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "mma";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] a matrix unit against the arithmetic pipe\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf(
        "  %ux%ux%u a tile: %u multiply-adds, %u a lane. The pipe issues one each\n\n",
        MMA_TILE, MMA_TILE, MMA_TILE, MMA_TILE * MMA_TILE * MMA_TILE,
        MMA_FRAGMENT_REGISTERS * MMA_TILE);
    std::printf("  %6s %10s %12s %12s %14s %12s\n", "tiles", "route", "instrs",
                "warp steps", "lane ops", "cycles");

    std::vector<Row> rows;
    for (const uint32_t tiles : {1u, 4u, 16u}) {
        const Row row{tiles, measure(arithmetic_pipe(tiles)),
                      measure(matrix_unit(tiles))};
        rows.push_back(row);
        std::printf("  %6u %10s %12zu %12llu %14s %12llu\n", tiles, "fma",
                    row.fma.instructions, (unsigned long long)row.fma.warp_steps,
                    with_commas(row.fma.weighted).c_str(),
                    (unsigned long long)row.fma.cycles);
        std::printf("  %6s %10s %12zu %12llu %14s %12llu\n", "", "mma",
                    row.mma.instructions, (unsigned long long)row.mma.warp_steps,
                    with_commas(row.mma.weighted).c_str(),
                    (unsigned long long)row.mma.cycles);
    }

    // What the cost table would have to say before the two came level. Everything
    // above is counted; this is the one line that touches the price.
    const uint32_t priced_at = instruction_cost(Opcode::V_MMA_16X16X16_F32);
    // The RET at the end of each program is in both totals and belongs to
    // neither route, so it comes off before the two are compared.
    const uint64_t ret = WARP_SIZE * instruction_cost(Opcode::RET);
    const double break_even = static_cast<double>(rows[0].fma.weighted - ret) / WARP_SIZE;

    std::printf("\n  One tile: %zu instructions against 1, %s lane ops against %s.\n",
                rows[0].fma.instructions - 1, with_commas(rows[0].fma.weighted).c_str(),
                with_commas(rows[0].mma.weighted).c_str());
    std::printf(
        "  Priced at %u, the matrix instruction would have to cost %.0f before the\n"
        "  two came level — which is the multiply-adds it performs, one a lane.\n",
        priced_at, break_even);

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "mma_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "tiles,route,instructions,warp_steps,weighted_lane_ops,cycles\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {{"fma", &r.fma},
                                                                 {"mma", &r.mma}};
        for (const auto& [name, reading] : routes) {
            csv << r.tiles << ',' << name << ',' << reading->instructions << ','
                << reading->warp_steps << ',' << reading->weighted << ','
                << reading->cycles << '\n';
        }
    }

    std::ofstream md(prefix + ".md");
    char buf[512];
    md << "<!-- generated by benchmarks/mma_bench; do not edit by hand -->\n\n"
       << "## A matrix unit against the arithmetic pipe\n\n"
       << MMA_TILE << "x" << MMA_TILE << "x" << MMA_TILE << " a tile, which is "
       << MMA_TILE * MMA_TILE * MMA_TILE << " multiply-adds — "
       << MMA_FRAGMENT_REGISTERS * MMA_TILE << " for each of the warp's " << WARP_SIZE
       << " lanes.\n\n"
       << "| Tiles | Route | Instructions | Warp steps | Lane ops | Cycles |\n"
       << "|---:|---|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {{"fma", &r.fma},
                                                                 {"mma", &r.mma}};
        for (const auto& [name, reading] : routes) {
            std::snprintf(buf, sizeof(buf), "| %u | %s | %zu | %llu | %s | %llu |\n",
                          r.tiles, name, reading->instructions,
                          (unsigned long long)reading->warp_steps,
                          with_commas(reading->weighted).c_str(),
                          (unsigned long long)reading->cycles);
            md << buf;
        }
    }
    std::snprintf(
        buf, sizeof(buf),
        "\nPriced at %u, the matrix instruction would have to cost %.0f a lane "
        "before the\ntwo routes came level — which is exactly the multiply-adds "
        "it performs. The\nratio between those two numbers is the claim being "
        "made about the unit, and\nit is the only number here that is not "
        "counted.\n",
        priced_at, break_even);
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
