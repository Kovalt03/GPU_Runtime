// Summing across a warp, four ways.
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
// The two atomic routes came later, with V_ATOM_ADD_GLOBAL_F32, and they answer
// a slightly different question: they leave the total in memory, where the two
// above leave it in a register. That is the form anything summing across warps
// or blocks needs, and it is where the choice between them actually arises —
// one atomic a lane against one a warp, with the reduction in front.
//
//   ./build/benchmarks/reduction_bench            benchmarks/result/reduction.{md,csv}
//   ./build/benchmarks/reduction_bench --out dir  writes into that directory

#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr uint8_t R_TAKEN = 7;     // the partner's value
constexpr uint8_t R_ADDR = 8;      // R_LANE and R_PARTNER as byte addresses
constexpr uint8_t R_SUM_ZERO = 9;  // 0.0, for the lane-0 test

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

// Every lane adds its own value to one counter. Thirty-two lanes on one address
// is thirty-two operations the atomic unit performs one after another, and no
// coalescing can merge them — which is the arithmetic the two routes above exist
// to avoid.
Program atomic_per_lane_reduction()
{
    Program p;
    emit_preamble(p);
    p.push_back(make_v_mov_f32(R_ADDR, 0.0f));
    p.push_back(make_v_atom_add_global_f32(R_TAKEN, R_ADDR, R_SUM));
    p.push_back(make_ret());
    return p;
}

// The exchange first, then one atomic from one lane. The reduction costs five
// rounds in registers and leaves the total in every lane; the atomic then carries
// it out of the warp, once, with nothing to collide with.
Program shuffle_then_atomic_reduction()
{
    Program p;
    emit_preamble(p);
    for (uint32_t offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
        emit_partner(p, offset);
        p.push_back(make_v_shuffle_f32(R_TAKEN, R_SUM, R_PARTNER, ALL_LANES));
        p.push_back(make_v_add_f32(R_SUM, R_SUM, R_TAKEN));
    }

    // Lane 0 alone. The others branch to the RET, so the atomic is issued with
    // one lane active and the collision depth is one.
    //
    // The zero is emitted here rather than in the preamble the four routes
    // share: adding an instruction there would move the two figures the
    // shared-against-exchange comparison already published.
    p.push_back(make_v_mov_f32(R_SUM_ZERO, 0.0f));
    p.push_back(make_v_cmp_f32(R_WRAPPED, R_LANE, R_SUM_ZERO, CmpOp::GT));
    p.push_back(make_bra_div(R_WRAPPED, 4));
    p.push_back(make_v_mov_f32(R_ADDR, 0.0f));
    p.push_back(make_v_atom_add_global_f32(R_TAKEN, R_ADDR, R_SUM));
    p.push_back(make_bra(1));
    p.push_back(make_ret());
    return p;
}

struct Reading {
    size_t instructions = 0;
    uint64_t warp_steps = 0;
    uint64_t active_lane_ops = 0;
    uint64_t weighted = 0;
    uint64_t cycles = 0;
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

    // Cycles, because what an atomic costs is mostly waiting: the collisions are
    // worked through one at a time and the issue count cannot see that. The
    // instruction and step counts are the same under every model.
    scheduler.set_latency_model(LatencyModel::Modelled);
    scheduler.set_memory_model(MemoryModel::Coalesced);

    std::vector<uint8_t> global(64, 0);
    scheduler.run(p, block, DeviceSpan{global.data(), global.size()});

    Reading r;
    r.instructions = p.size();
    r.warp_steps = scheduler.stats().warp_steps;
    r.active_lane_ops = scheduler.stats().active_lane_ops;
    r.weighted = scheduler.stats().weighted_lane_ops;
    r.cycles = scheduler.stats().cycles;

    // The register routes leave the total in every lane; the atomic ones leave it
    // at byte 0. Whichever this program filled is the answer.
    float in_memory = 0.0f;
    std::memcpy(&in_memory, global.data(), sizeof(float));
    r.answer = in_memory != 0.0f ? in_memory : block.warps[0].threads[0].regs[R_SUM];
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
    const Program per_lane = atomic_per_lane_reduction();
    const Program then_atomic = shuffle_then_atomic_reduction();

    // 0 + 1 + ... + 31, which every lane ends up holding.
    constexpr float EXPECTED = WARP_SIZE * (WARP_SIZE - 1) / 2.0f;

    std::printf("\n[BENCH] summing a warp: shared memory, the exchange, and atomics\n\n");
    std::printf("  %-8s %16s %12s %12s %14s %10s %10s\n", "warps", "route", "instrs",
                "warp steps", "lane ops", "cycles", "answer");

    struct Row {
        uint32_t warps;
        Reading shared;
        Reading shuffled;
        Reading per_lane;
        Reading then_atomic;
    };
    std::vector<Row> rows;
    for (const uint32_t warps : {1u, 2u, 4u, 8u}) {
        Row row{warps, measure(shared, warps), measure(shuffled, warps),
                measure(per_lane, warps), measure(then_atomic, warps)};
        rows.push_back(row);

        const std::pair<const char*, const Reading*> routes[] = {
            {"shared", &row.shared},
            {"shuffle", &row.shuffled},
            {"atomic a lane", &row.per_lane},
            {"shuffle+atomic", &row.then_atomic}};
        bool first_row = true;
        for (const auto& [name, r] : routes) {
            std::printf("  %-8s %16s %12zu %12llu %14llu %10llu %10.0f\n",
                        first_row ? std::to_string(warps).c_str() : "", name,
                        r->instructions, (unsigned long long)r->warp_steps,
                        (unsigned long long)r->active_lane_ops,
                        (unsigned long long)r->cycles, r->answer);
            first_row = false;
        }
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
    csv << "warps,route,instructions,warp_steps,active_lane_ops,weighted_lane_ops,"
           "cycles\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {
            {"shared", &r.shared},
            {"shuffle", &r.shuffled},
            {"atomic a lane", &r.per_lane},
            {"shuffle+atomic", &r.then_atomic}};
        for (const auto& [name, reading] : routes) {
            csv << r.warps << ',' << name << ',' << reading->instructions << ','
                << reading->warp_steps << ',' << reading->active_lane_ops << ','
                << reading->weighted << ',' << reading->cycles << '\n';
        }
    }

    std::ofstream md(prefix + ".md");
    char buf[512];
    md << "<!-- generated by benchmarks/reduction_bench; do not edit by hand -->\n\n"
       << "## Summing a warp\n\n"
       << "The two atomic routes leave the total in memory rather than in a "
          "register, so\ntheir answer accumulates across the block's warps where "
          "the other two do not.\nThat is the reason to reach for one, and the "
          "cycles are the price.\n\n"
       << "| Warps | Route | Instructions | Warp steps | Lane ops | Cycles |\n"
       << "|---|---|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {
            {"shared", &r.shared},
            {"shuffle", &r.shuffled},
            {"atomic a lane", &r.per_lane},
            {"shuffle+atomic", &r.then_atomic}};
        for (const auto& [name, reading] : routes) {
            std::snprintf(buf, sizeof(buf), "| %u | %s | %zu | %llu | %llu | %llu |\n",
                          r.warps, name, reading->instructions,
                          (unsigned long long)reading->warp_steps,
                          (unsigned long long)reading->active_lane_ops,
                          (unsigned long long)reading->cycles);
            md << buf;
        }
    }
    std::snprintf(buf, sizeof(buf),
                  "\nPriced at %u, a shuffle would have to reach %.0f before shared "
                  "memory and the\nexchange came level.\n\nOne warp: an atomic a lane "
                  "is %zu instructions against %zu and %llu cycles\nagainst %llu. The "
                  "issue count says it wins and the clock says it loses by\n%.1fx — "
                  "thirty-two lanes on one address are thirty-two operations, worked\n"
                  "through one at a time.\n",
                  priced_at, break_even, rows[0].per_lane.instructions,
                  rows[0].then_atomic.instructions,
                  (unsigned long long)rows[0].per_lane.cycles,
                  (unsigned long long)rows[0].then_atomic.cycles,
                  static_cast<double>(rows[0].per_lane.cycles) /
                      static_cast<double>(rows[0].then_atomic.cycles));
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
