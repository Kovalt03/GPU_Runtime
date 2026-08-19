// What warp divergence costs.
//
// The renderer reports a divergence rate, but a rate on its own says nothing
// about the price. This varies the split deliberately, holding the work
// constant, so the two can be put side by side.
//
//   ./build/benchmarks/divergence_bench
//   ./build/benchmarks/divergence_bench --csv    (writes
//   test/benchmark/output/divergence.csv)

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "runtime.hpp"

namespace {

// 1024 warps: long enough that the clock has something to measure, short enough
// to rerun while iterating.
constexpr uint32_t WARPS = 1024;

// Per path. Both sides run the same count, so the split is the only variable.
constexpr int PATH_LENGTH = 20;

// Timing on a wall clock is noisy, and the first run pays for cold caches and
// page faults. Discard it, then keep the best of the rest — the fastest run is
// the one least disturbed by whatever else the machine was doing.
constexpr int WARMUP_RUNS = 1;
constexpr int TIMED_RUNS = 4;

struct Sample {
    uint32_t diverging_lanes = 0;
    double divergence_rate = 0.0;
    uint64_t warp_steps = 0;
    uint64_t active_lane_ops = 0;
    double giops = 0.0;
};

// Lanes below the threshold take one path, the rest take the other. Both paths
// hold PATH_LENGTH instructions, so any difference in cost comes from the split
// alone and not from one side doing more work.
KernelFunc make_split_kernel(uint32_t diverging_lanes)
{
    return [diverging_lanes](void**) {
        Program p;
        p.push_back(make_v_mov_f32(1, static_cast<float>(diverging_lanes)));
        p.push_back(make_v_cmp_f32(2, REG_GLOBAL_ID_X, 1, CmpOp::LT));

        const size_t branch = p.size();
        p.push_back(make_bra_div(2, 0));

        for (int i = 0; i < PATH_LENGTH; ++i) {
            p.push_back(make_v_add_f32(10, 10, 1));
        }
        const size_t skip = p.size();
        p.push_back(make_bra(0));

        const size_t other_path = p.size();
        for (int i = 0; i < PATH_LENGTH; ++i) {
            p.push_back(make_v_mul_f32(11, 11, 1));
        }

        const size_t join = p.size();
        p[branch] = make_bra_div(2, static_cast<int32_t>(other_path - branch));
        p[skip] = make_bra(static_cast<int32_t>(join - skip));

        p.push_back(make_ret());
        return p;
    };
}

Sample measure(uint32_t diverging_lanes)
{
    Sample s;
    s.diverging_lanes = diverging_lanes;

    // grid along y, block along x: REG_GLOBAL_ID_X then runs 0..31 and is the
    // lane index within the warp. Putting the grid on x instead would make it
    // the global thread id, and the comparison would separate whole warps
    // rather than the lanes inside one — divergence would read as zero.
    const dim3 grid{1, WARPS, 1};
    const dim3 block{WARP_SIZE, 1, 1};

    for (int run = 0; run < WARMUP_RUNS + TIMED_RUNS; ++run) {
        MyGPURuntime rt(1u << 16, 4096);
        rt.myrt_launch(make_split_kernel(diverging_lanes), grid, block, nullptr);

        if (run < WARMUP_RUNS) {
            continue;
        }
        if (rt.throughput_giops() > s.giops) {
            s.giops = rt.throughput_giops();
        }
        // Deterministic, so any run reports the same figures.
        s.divergence_rate = rt.divergence_rate();
        s.warp_steps = rt.stats().warp_steps;
        s.active_lane_ops = rt.stats().active_lane_ops;
    }
    return s;
}

void write_csv(const std::vector<Sample>& samples, const std::string& path)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open " + path + " for writing");
    }
    out << "diverging_lanes,divergence_rate,warp_steps,active_lane_ops,giops\n";
    for (const Sample& s : samples) {
        out << s.diverging_lanes << ',' << s.divergence_rate << ',' << s.warp_steps << ','
            << s.active_lane_ops << ',' << s.giops << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const bool csv = (argc > 1) && std::string(argv[1]) == "--csv";

    std::vector<Sample> samples;
    for (uint32_t lanes : {0u, 1u, 2u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 31u, 32u}) {
        samples.push_back(measure(lanes));
    }

    // The converged baseline every other row is compared against.
    const Sample& base = samples.front();

    std::printf(
        "[BENCH] warp divergence vs cost — %u warps, %d instructions per path\n\n", WARPS,
        PATH_LENGTH);
    std::printf(
        "  lanes | divergence | warp steps | issue | useful ops |  GIOPS | slowdown\n");
    std::printf(
        "  ------|------------|------------|-------|------------|--------|---------\n");
    for (const Sample& s : samples) {
        std::printf(
            "  %2u/%2u |    %6.2f%% | %10llu | %4.2fx | %10llu | %6.3f |   %5.2fx\n",
            s.diverging_lanes, WARP_SIZE, 100.0 * s.divergence_rate,
            static_cast<unsigned long long>(s.warp_steps),
            static_cast<double>(s.warp_steps) / static_cast<double>(base.warp_steps),
            static_cast<unsigned long long>(s.active_lane_ops), s.giops,
            (s.giops > 0.0) ? base.giops / s.giops : 0.0);
    }

    std::printf(
        "\n  One lane taking the branch costs exactly what sixteen do: the two\n");
    std::printf(
        "  sides get issued separately either way, and a warp step buys 32 lane\n");
    std::printf(
        "  slots whether one lane fills them or all of them. Both extremes read\n");
    std::printf("  0%% — divergence is not about branching, but about disagreeing.\n");

    // "issue" counts what the scheduler had to put through and is identical on
    // any machine. GIOPS is a wall clock reading of this simulator, where a
    // masked lane is skipped with a continue and so genuinely costs less;
    // hardware would keep its ALU busy regardless. That is why the issue column
    // is the one that models the real cost, and the GIOPS ratios come out
    // milder than 1.80x.
    std::printf(
        "\n  issue = instructions the scheduler had to put through (deterministic)\n");
    std::printf("  GIOPS = this simulator on this host (indicative, ratios only)\n");

    if (csv) {
        const std::string path = "test/benchmark/output/divergence.csv";
        write_csv(samples, path);
        std::printf("\n  wrote %s\n", path.c_str());
    }
    return 0;
}
