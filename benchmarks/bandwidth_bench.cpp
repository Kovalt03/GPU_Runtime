// What happens when the memory system has a ceiling.
//
// Every figure in this repository before this one was taken with transactions
// counted and never queued: two warps each needing four lines were served as
// though the other were not there. Nothing saturated, because there was nothing
// to saturate — no shared resource and no rate.
//
// BandwidthModel::Modelled gives the memory system one: it delivers a fixed
// number of lines a cycle, and a request arriving while it is busy waits for the
// ones ahead. Latency stops being a constant and becomes a function of load,
// which is the difference between a kernel being latency bound and being
// bandwidth bound.
//
// Two questions, and the second is the one that matters for everything above:
//
//   The mechanism.  How the wait grows as the ceiling comes down.
//   The ceiling.    Whether more SMs still buy what they bought. Every scaling
//                   figure in RESULTS.md was measured without one.
//
//   ./build/benchmarks/bandwidth_bench                     result/bandwidth.{md,csv}
//   ./build/benchmarks/bandwidth_bench --machine <file>    the rest comes from it
//   ./build/benchmarks/bandwidth_bench --out dir           writes into that directory

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
    uint64_t queued = 0;
    uint64_t misses = 0;
};

// A warp that streams: every lane reads a line of its own, and none of them is
// ever read twice. Nothing else in this repository asks memory for so much at
// once, and that is what it takes to reach a ceiling — the renderer below asks
// for a fifth of a line a cycle.
Program streaming_kernel(size_t base, uint32_t reads)
{
    IRBuilder k;
    // Each block reads a region of its own, or the second warp would find the
    // first one's lines resident and ask memory for nothing.
    const Reg<Scalar> at =
        k.add(k.add(k.constant(static_cast<float>(base)),
                    k.mul(k.block_x(), k.constant(static_cast<float>(reads * WARP_SIZE *
                                                                     CACHE_LINE_BYTES)))),
              k.mul(k.thread_x(), k.constant(static_cast<float>(CACHE_LINE_BYTES))));
    Reg<Scalar> sum = k.constant(0.0f);
    for (uint32_t i = 0; i < reads; ++i) {
        k.fma(sum, k.load(at, static_cast<float>(i * WARP_SIZE * CACHE_LINE_BYTES)),
              k.constant(1.0f));
    }
    k.store(k.constant(static_cast<float>(base)), sum);
    return k.build();
}

Reading stream(uint32_t warps, uint32_t lines_a_cycle)
{
    MyGPURuntime rt(1u << 24);
    GPUSpec spec = MACHINE;
    spec.sms.sm_count = 1;
    spec.sms.blocks_per_sm = warps;
    if (lines_a_cycle > 0) {
        spec.memory_lines_a_cycle = lines_a_cycle;
    }
    rt.myrt_set_spec(spec);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);
    rt.myrt_set_bandwidth_model(lines_a_cycle > 0 ? BandwidthModel::Modelled
                                                  : BandwidthModel::Ignored);

    constexpr uint32_t READS = 8;
    void* buffer = rt.myrt_malloc(warps * READS * WARP_SIZE * CACHE_LINE_BYTES);
    const Program prog = streaming_kernel(rt.myrt_device_offset(buffer), READS);
    rt.myrt_launch([prog](void**) { return prog; }, dim3{warps, 1, 1},
                   dim3{WARP_SIZE, 1, 1}, nullptr);

    return Reading{rt.stats().weighted_lane_ops, rt.stats().cycles,
                   rt.stats().stall_steps, rt.stats().memory_queue_cycles,
                   rt.stats().cache_misses};
}

// The caches are scaled down, as cache_bench scales them, and for the same
// reason turned around: at the hardware sizes this scene misses a few hundred
// times and a ceiling has nothing to bite on. **That is itself the finding** —
// bandwidth is a property of the working set, not of the machine alone — and it
// is stated in the section rather than hidden by picking a bigger scene.
constexpr size_t SMALL_L1 = 16;
constexpr size_t SMALL_L2 = 128;

Reading measure(const std::vector<Float3>& world, uint32_t sm_count,
                uint32_t lines_a_cycle, bool small_caches)
{
    MyGPURuntime rt(1u << 27);
    GPUSpec spec = MACHINE;
    spec.sms.sm_count = sm_count;
    spec.sms.blocks_per_sm = 8;
    if (small_caches) {
        spec.l1_lines = SMALL_L1;
        spec.l2_lines = SMALL_L2;
    }
    if (lines_a_cycle > 0) {
        spec.memory_lines_a_cycle = lines_a_cycle;
    }
    rt.myrt_set_spec(spec);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);
    rt.myrt_set_bandwidth_model(lines_a_cycle > 0 ? BandwidthModel::Modelled
                                                  : BandwidthModel::Ignored);

    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, bench_target());
    draw_walk(rt, geometry, frame, bench_target());

    const Reading r{rt.stats().weighted_lane_ops, rt.stats().cycles,
                    rt.stats().stall_steps, rt.stats().memory_queue_cycles,
                    rt.stats().cache_misses};
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
    uint32_t sms = 0;
    uint32_t lines = 0;  // zero when the model is off
    Reading reading;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "bandwidth";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    const std::vector<Float3> world = spread(4);

    std::printf("\n[BENCH] a memory system with a ceiling\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());

    // --- how the wait grows --------------------------------------------------

    std::printf(
        "  Eight warps streaming, every lane on a line of its own — %u lines "
        "asked for\n\n",
        8u * WARP_SIZE * 8u);
    std::printf("  %14s %14s %14s %14s\n", "lines a cycle", "cycles", "queued",
                "vs none");

    std::vector<Row> rows;
    const Reading unlimited = stream(8, 0);
    rows.push_back(Row{8, 0, unlimited});
    std::printf("  %14s %14s %14s %14s\n", "no ceiling",
                with_commas(unlimited.cycles).c_str(), "0", "—");

    for (const uint32_t lines : {32u, 8u, 2u, 1u}) {
        const Reading r = stream(8, lines);
        rows.push_back(Row{8, lines, r});
        char change[32];
        std::snprintf(
            change, sizeof(change), "%.2fx",
            static_cast<double>(r.cycles) / static_cast<double>(unlimited.cycles));
        std::printf("  %14u %14s %14s %14s\n", lines, with_commas(r.cycles).c_str(),
                    with_commas(r.queued).c_str(), change);
    }

    // --- what it does to the scaling above -----------------------------------

    std::printf(
        "\n  And what it does to a renderer. draw_walk, L1 %zu lines so that it "
        "misses at all\n\n",
        SMALL_L1);
    std::printf("  %6s %16s %10s %16s %10s\n", "SMs", "no ceiling", "vs 1 SM",
                "8 lines a cycle", "vs 1 SM");

    std::vector<Row> scaling;
    uint64_t base_free = 0;
    uint64_t base_capped = 0;
    for (const uint32_t sms : {1u, 2u, 4u, 8u, 16u, 32u}) {
        const Reading free_run = measure(world, sms, 0, true);
        const Reading capped = measure(world, sms, 8, true);
        scaling.push_back(Row{sms, 0, free_run});
        scaling.push_back(Row{sms, 8, capped});
        if (sms == 1) {
            base_free = free_run.cycles;
            base_capped = capped.cycles;
        }
        std::printf(
            "  %6u %16s %9.2fx %16s %9.2fx\n", sms, with_commas(free_run.cycles).c_str(),
            static_cast<double>(base_free) / static_cast<double>(free_run.cycles),
            with_commas(capped.cycles).c_str(),
            static_cast<double>(base_capped) / static_cast<double>(capped.cycles));
    }

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "bandwidth_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "table,sms,lines_a_cycle,weighted,cycles,stalls,queued,misses\n";
    const auto rows_to_csv = [&csv](const char* table, const std::vector<Row>& all) {
        for (const Row& r : all) {
            csv << table << ',' << r.sms << ',' << r.lines << ',' << r.reading.weighted
                << ',' << r.reading.cycles << ',' << r.reading.stalls << ','
                << r.reading.queued << ',' << r.reading.misses << '\n';
        }
    };
    rows_to_csv("ceiling", rows);
    rows_to_csv("scaling", scaling);

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/bandwidth_bench; do not edit by hand -->\n\n"
       << "## A memory system with a ceiling\n\n"
       << "Eight warps streaming, every lane on a line of its own and none read "
          "twice.\n\n"
       << "| Lines a cycle | Cycles | queued | vs none |\n|---|---:|---:|---:|\n";
    for (const Row& r : rows) {
        if (r.lines == 0) {
            std::snprintf(buf, sizeof(buf), "| no ceiling | %s | 0 | — |\n",
                          with_commas(r.reading.cycles).c_str());
        } else {
            std::snprintf(buf, sizeof(buf), "| %u | %s | %s | **%.2fx** |\n", r.lines,
                          with_commas(r.reading.cycles).c_str(),
                          with_commas(r.reading.queued).c_str(),
                          static_cast<double>(r.reading.cycles) /
                              static_cast<double>(unlimited.cycles));
        }
        md << buf;
    }

    md << "\n## What it does to more SMs\n\n"
       << "The same sweep *Several SMs* makes, with and without a ceiling of eight "
          "lines a\ncycle. Every scaling figure in this file was taken without "
          "one.\n\n"
       << "| SMs | no ceiling | vs 1 SM | 8 lines a cycle | vs 1 SM |\n"
       << "|---:|---:|---:|---:|---:|\n";
    for (size_t i = 0; i < scaling.size(); i += 2) {
        std::snprintf(buf, sizeof(buf), "| %u | %s | %.2fx | %s | **%.2fx** |\n",
                      scaling[i].sms, with_commas(scaling[i].reading.cycles).c_str(),
                      static_cast<double>(base_free) /
                          static_cast<double>(scaling[i].reading.cycles),
                      with_commas(scaling[i + 1].reading.cycles).c_str(),
                      static_cast<double>(base_capped) /
                          static_cast<double>(scaling[i + 1].reading.cycles));
        md << buf;
    }

    // What the two ask for, which is the whole of why one saturates and the
    // other does not.
    const Reading busiest = scaling[scaling.size() - 2].reading;
    std::printf(
        "\n  Demand: the streaming warps ask for %.2f lines a cycle, the renderer "
        "%.2f.\n  A ceiling binds when it is below what a kernel asks for, and "
        "nothing else.\n",
        static_cast<double>(unlimited.misses) / static_cast<double>(unlimited.cycles),
        static_cast<double>(busiest.misses) / static_cast<double>(busiest.cycles));

    std::snprintf(
        buf, sizeof(buf),
        "\nThe streaming warps ask for %.2f lines a cycle and the renderer for "
        "%.2f. A\nceiling binds when it is below what a kernel asks for, and "
        "nothing else — which is\nwhy every figure in this file stands.\n",
        static_cast<double>(unlimited.misses) / static_cast<double>(unlimited.cycles),
        static_cast<double>(busiest.misses) / static_cast<double>(busiest.cycles));
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
