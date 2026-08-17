// Summing across a warp, two ways.
//
// Before V_SHUFFLE_F32 the only route was shared memory: every lane stores, the
// block synchronises, every lane loads its partner's value. The exchange does it
// in registers. Both halve the live values each round, so both take log2(32) = 5
// of them, and what differs is what one round costs.
//
// The instruction costs of the warp primitives are placeholders — all 1 in
// isa.cpp, pending this measurement. So weighted_lane_ops cannot be the figure
// that decides the comparison: it would only report the guess back. Counted here
// instead are the things the cost table does not touch — how many instructions
// each program contains and how many warp steps the scheduler issued — and the
// weighted total is turned around to ask how expensive a shuffle would have to
// be before the two routes came level.
//
//   ./build/benchmarks/reduction_bench            benchmarks/result/reduction.{md,csv}
//   ./build/benchmarks/reduction_bench --out dir  writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "isa.hpp"
#include "scheduler.hpp"

namespace {

constexpr uint32_t ALL_LANES = 0xFFFFFFFFu;

// Registers, named once so the two programs cannot drift apart in what they
// hold where.
constexpr uint8_t R_SUM = 0;      // the running total, and the answer in lane 0
constexpr uint8_t R_LANE = 1;     // this lane's index, seeded before the run
constexpr uint8_t R_PARTNER = 2;  // the lane to take from this round
constexpr uint8_t R_OFFSET = 3;
constexpr uint8_t R_WIDTH = 4;    // WARP_SIZE, for the wrap
constexpr uint8_t R_WRAPPED = 5;  // 1.0 when partner ran off the end
constexpr uint8_t R_NEG_WIDTH = 6;
constexpr uint8_t R_TAKEN = 7;  // the partner's value
constexpr uint8_t R_ADDR = 8;   // R_LANE and R_PARTNER as byte addresses

// partner = (lane + offset) mod WARP_SIZE, without a branch.
//
// Both programs need it and both pay the same three instructions for it, so it
// cancels out of the comparison. Branchless on purpose: a divergent wrap would
// put warp splits into a measurement about something else.
void emit_partner(Program& p, uint32_t offset)
{
    p.push_back(make_v_mov_f32(R_OFFSET, static_cast<float>(offset)));
    p.push_back(make_v_add_f32(R_PARTNER, R_LANE, R_OFFSET));
    p.push_back(make_v_cmp_f32(R_WRAPPED, R_PARTNER, R_WIDTH, CmpOp::GE));
    p.push_back(make_v_fma_f32(R_PARTNER, R_WRAPPED, R_NEG_WIDTH));
}

void emit_preamble(Program& p)
{
    p.push_back(make_v_mov_f32(R_WIDTH, static_cast<float>(WARP_SIZE)));
    p.push_back(make_v_mov_f32(R_NEG_WIDTH, -static_cast<float>(WARP_SIZE)));
}

// Round: store mine, wait for everyone, load my partner's, add.
//
// The barrier is what makes this a block-wide operation for a warp-wide answer,
// and this machine charges nothing for it beyond the instruction: warp_steps
// comes out at exactly one per warp per instruction however many warps are in
// the block, so the wait for the slowest one costs nothing here. On hardware it
// is most of what a barrier is. The figures below therefore understate the
// shared route rather than flatter it.
Program shared_memory_reduction()
{
    Program p;
    emit_preamble(p);
    for (uint32_t offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
        emit_partner(p, offset);

        // Four bytes a lane, so the lane index doubles as the address once
        // scaled. R_ADDR is rebuilt each round because R_PARTNER moves.
        p.push_back(make_v_mov_f32(R_ADDR, 4.0f));
        p.push_back(make_v_mul_f32(R_ADDR, R_LANE, R_ADDR));
        p.push_back(make_v_st_shared_f32(R_ADDR, R_SUM, 0.0f));
        p.push_back(make_barrier());

        p.push_back(make_v_mov_f32(R_ADDR, 4.0f));
        p.push_back(make_v_mul_f32(R_ADDR, R_PARTNER, R_ADDR));
        p.push_back(make_v_ld_shared_f32(R_TAKEN, R_ADDR, 0.0f));
        p.push_back(make_v_add_f32(R_SUM, R_SUM, R_TAKEN));

        // A second barrier: without it a fast lane's next store would land in
        // the slot a slow one has yet to read.
        p.push_back(make_barrier());
    }
    p.push_back(make_ret());
    return p;
}

// The same reduction with the exchange in place of the round trip.
Program shuffle_reduction()
{
    Program p;
    emit_preamble(p);
    for (uint32_t offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
        emit_partner(p, offset);
        p.push_back(make_v_shuffle_f32(R_TAKEN, R_SUM, R_PARTNER, ALL_LANES));
        p.push_back(make_v_add_f32(R_SUM, R_SUM, R_TAKEN));
    }
    p.push_back(make_ret());
    return p;
}

struct Reading {
    size_t instructions = 0;
    uint64_t warp_steps = 0;
    uint64_t active_lane_ops = 0;
    uint64_t weighted = 0;
    float answer = 0.0f;
};

// warps of 32 lanes each, every lane holding its own index, summed.
Reading measure(const Program& p, uint32_t warps)
{
    ThreadBlock block = make_block(warps);
    for (uint32_t w = 0; w < warps; ++w) {
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            block.warps[w].threads[lane].regs[R_LANE] = static_cast<float>(lane);
            block.warps[w].threads[lane].regs[R_SUM] = static_cast<float>(lane);
        }
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> global(64, 0);
    scheduler.run(p, block, DeviceSpan{global.data(), global.size()});

    Reading r;
    r.instructions = p.size();
    r.warp_steps = scheduler.stats().warp_steps;
    r.active_lane_ops = scheduler.stats().active_lane_ops;
    r.weighted = scheduler.stats().weighted_lane_ops;
    r.answer = block.warps[0].threads[0].regs[R_SUM];
    return r;
}

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
}

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "reduction";

    const Program shared = shared_memory_reduction();
    const Program shuffled = shuffle_reduction();

    // 0 + 1 + ... + 31, which every lane ends up holding.
    constexpr float EXPECTED = WARP_SIZE * (WARP_SIZE - 1) / 2.0f;

    std::printf("\n[BENCH] summing a warp, shared memory against the exchange\n\n");
    std::printf("  %-8s %14s %12s %12s %14s %10s\n", "warps", "route", "instrs",
                "warp steps", "lane ops", "answer");

    struct Row {
        uint32_t warps;
        Reading shared;
        Reading shuffled;
    };
    std::vector<Row> rows;
    for (const uint32_t warps : {1u, 2u, 4u, 8u}) {
        Row row{warps, measure(shared, warps), measure(shuffled, warps)};
        rows.push_back(row);

        std::printf("  %-8u %14s %12zu %12llu %14llu %10.0f\n", warps, "shared",
                    row.shared.instructions, (unsigned long long)row.shared.warp_steps,
                    (unsigned long long)row.shared.active_lane_ops, row.shared.answer);
        std::printf(
            "  %-8s %14s %12zu %12llu %14llu %10.0f\n", "", "shuffle",
            row.shuffled.instructions, (unsigned long long)row.shuffled.warp_steps,
            (unsigned long long)row.shuffled.active_lane_ops, row.shuffled.answer);
    }

    if (rows[0].shared.answer != EXPECTED || rows[0].shuffled.answer != EXPECTED) {
        std::fprintf(stderr,
                     "\nreduction_bench: expected %.0f — the two routes are "
                     "not computing the same thing\n",
                     EXPECTED);
        return 1;
    }

    const Row& one = rows[0];
    std::printf(
        "\n  One warp: %zu instructions against %zu, %llu warp steps against "
        "%llu.\n",
        one.shared.instructions, one.shuffled.instructions,
        (unsigned long long)one.shared.warp_steps,
        (unsigned long long)one.shuffled.warp_steps);

    // Turned around, so that whatever the cost table currently says about a
    // shuffle does not decide the comparison. Raising its price by one adds a
    // lane op per lane per round to the exchange and nothing to shared memory,
    // so this is where the two would meet.
    const uint64_t shared_weighted = one.shared.weighted;
    const uint64_t shuffle_weighted = one.shuffled.weighted;
    const uint64_t rounds = 5;
    const uint64_t shuffle_lane_ops = rounds * WARP_SIZE;
    const uint32_t priced_at = instruction_cost(Opcode::V_SHUFFLE_F32);
    const double break_even =
        priced_at + static_cast<double>(shared_weighted - shuffle_weighted) /
                        static_cast<double>(shuffle_lane_ops);

    std::printf("\n  Weighted: %llu against %llu (%.1f%%), a shuffle priced at %u.\n",
                (unsigned long long)shared_weighted, (unsigned long long)shuffle_weighted,
                change(shared_weighted, shuffle_weighted), priced_at);
    std::printf("  It would have to cost %.0f before the two came level.\n", break_even);

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "reduction_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "warps,route,instructions,warp_steps,active_lane_ops,weighted_lane_ops\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {
            {"shared", &r.shared}, {"shuffle", &r.shuffled}};
        for (const auto& [name, reading] : routes) {
            csv << r.warps << ',' << name << ',' << reading->instructions << ','
                << reading->warp_steps << ',' << reading->active_lane_ops << ','
                << reading->weighted << '\n';
        }
    }

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/reduction_bench; do not edit by hand -->\n\n"
       << "## Summing a warp\n\n"
       << "| Warps | Route | Instructions | Warp steps | Lane ops |\n"
       << "|---|---|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %u | shared | %zu | %llu | %llu |\n", r.warps,
                      r.shared.instructions, (unsigned long long)r.shared.warp_steps,
                      (unsigned long long)r.shared.active_lane_ops);
        md << buf;
        std::snprintf(buf, sizeof(buf), "| %u | shuffle | %zu | %llu | %llu |\n", r.warps,
                      r.shuffled.instructions, (unsigned long long)r.shuffled.warp_steps,
                      (unsigned long long)r.shuffled.active_lane_ops);
        md << buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "\nPriced at %u, a shuffle would have to reach %.0f before the two "
                  "routes came level.\n",
                  priced_at, break_even);
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
