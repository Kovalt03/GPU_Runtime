// The kernel the last two instructions were waiting for.
//
// V_MMA_16X16X16_F32 had no kernel at all, and cp.async had one whose staged
// tile was read 256 times — so hiding the fetch could not matter there, and
// RESULTS.md said as much and named this as the place to ask again.
//
// A tiled matrix multiply is the other end of both ratios: a staged tile is
// consumed by a fixed, small number of operations. Two flags, four routes, one
// kernel — the same arithmetic and the same staging, and only the inner loop and
// the way the tiles arrive differ.
//
//   ./build/benchmarks/gemm_bench                     result/gemm.{md,csv}
//   ./build/benchmarks/gemm_bench --machine <file>    caches and latencies from it
//   ./build/benchmarks/gemm_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gemm.hpp"
#include "gpu_spec.hpp"
#include "half.hpp"

namespace {

GPUSpec MACHINE;

struct Reading {
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    uint64_t stalls = 0;
};

std::vector<float> ramp(size_t n, int period, float scale)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(static_cast<int>(i % period) - period / 2) * scale;
    }
    return v;
}

struct Route {
    bool matrix_unit = true;
    bool wide_fragments = false;
    bool half_inputs = false;
    TileStaging staging = TileStaging::Synchronous;
};

Reading measure(uint32_t m, uint32_t n, uint32_t k, const Route& how)
{
    const std::vector<float> a = ramp(static_cast<size_t>(m) * k, 7, 0.5f);
    const std::vector<float> b = ramp(static_cast<size_t>(k) * n, 5, 0.25f);

    MyGPURuntime rt(1u << 26);
    rt.myrt_set_spec(MACHINE);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

    // The device never converts: halves arrive packed, as they do on hardware.
    const std::vector<float> a_device = how.half_inputs ? pack_halves(a) : a;
    const std::vector<float> b_device = how.half_inputs ? pack_halves(b) : b;

    void* da = rt.myrt_malloc(a_device.size() * sizeof(float));
    void* db = rt.myrt_malloc(b_device.size() * sizeof(float));
    void* dc = rt.myrt_malloc(static_cast<size_t>(m) * n * sizeof(float));
    rt.myrt_memcpy(da, a_device.data(), a_device.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(db, b_device.data(), b_device.size() * sizeof(float),
                   Direction::HostToDevice);

    GemmArgs args;
    args.a_offset = rt.myrt_device_offset(da);
    args.b_offset = rt.myrt_device_offset(db);
    args.c_offset = rt.myrt_device_offset(dc);
    args.m = m;
    args.n = n;
    args.k = k;
    args.matrix_unit = how.matrix_unit;
    args.wide_fragments = how.wide_fragments;
    args.half_inputs = how.half_inputs;
    args.staging = how.staging;
    run_gemm(rt, args);

    return Reading{rt.stats().weighted_lane_ops, rt.stats().cycles,
                   rt.stats().stall_steps};
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
    uint32_t k = 0;
    Reading fma_sync;
    Reading mma_sync;
    Reading mma_async;
    Reading mma_wide;
    Reading mma_half;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "gemm";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    constexpr uint32_t M = 64;
    constexpr uint32_t N = 256;

    std::printf(
        "\n[BENCH] a tiled matrix multiply, which is what both of them wanted\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf(
        "  C[%ux%u] = A * B, a block of %u warps holding a %ux%u strip in registers\n\n",
        M, N, GEMM_WARPS, GEMM_TILE, GEMM_TILE_N);
    std::printf("  %6s %14s %14s %10s %14s %10s %14s %10s %14s %10s\n", "K", "fma, sync",
                "mma, sync", "change", "mma, staged", "change", "+wide frags", "change",
                "+halves", "change");

    std::vector<Row> rows;
    for (const uint32_t k : {32u, 64u, 128u}) {
        const Row row{
            k,
            measure(M, N, k, Route{false, false, false, TileStaging::Synchronous}),
            measure(M, N, k, Route{true, false, false, TileStaging::Synchronous}),
            measure(M, N, k, Route{true, false, false, TileStaging::AsyncDoubleBuffered}),
            measure(M, N, k, Route{true, true, false, TileStaging::AsyncDoubleBuffered}),
            measure(M, N, k, Route{true, false, true, TileStaging::AsyncDoubleBuffered})};
        rows.push_back(row);
        std::printf("  %6u %14s %14s %9.1f%% %14s %9.1f%% %14s %9.1f%% %14s %9.1f%%\n", k,
                    with_commas(row.fma_sync.cycles).c_str(),
                    with_commas(row.mma_sync.cycles).c_str(),
                    change(row.fma_sync.cycles, row.mma_sync.cycles),
                    with_commas(row.mma_async.cycles).c_str(),
                    change(row.mma_sync.cycles, row.mma_async.cycles),
                    with_commas(row.mma_wide.cycles).c_str(),
                    change(row.mma_async.cycles, row.mma_wide.cycles),
                    with_commas(row.mma_half.cycles).c_str(),
                    change(row.mma_wide.cycles, row.mma_half.cycles));
    }

    std::printf("\n  Issued work, same rows: fma %s, mma %s, staged ahead %s, wide %s\n",
                with_commas(rows.back().fma_sync.weighted).c_str(),
                with_commas(rows.back().mma_sync.weighted).c_str(),
                with_commas(rows.back().mma_async.weighted).c_str(),
                with_commas(rows.back().mma_wide.weighted).c_str());
    std::printf(
        "  The last two are equal: a fragment is eight floats however it is "
        "asked for.\n  What the wide load removes is the seven waits.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "gemm_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "m,n,k,route,weighted_lane_ops,cycles,stalls\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {
            {"fma,sync", &r.fma_sync},
            {"mma,sync", &r.mma_sync},
            {"mma,async", &r.mma_async},
            {"mma,async,wide", &r.mma_wide},
            {"mma,async,half", &r.mma_half}};
        for (const auto& [name, reading] : routes) {
            csv << M << ',' << N << ',' << r.k << ',' << name << ',' << reading->weighted
                << ',' << reading->cycles << ',' << reading->stalls << '\n';
        }
    }

    std::ofstream md(prefix + ".md");
    char buf[400];
    md << "<!-- generated by benchmarks/gemm_bench; do not edit by hand -->\n\n"
       << "## A tiled matrix multiply\n\n"
       << "C[" << M << "x" << N << "] = A * B, a block of " << GEMM_WARPS
       << " warps holding a " << GEMM_TILE << "x" << GEMM_TILE_N
       << " strip of C in\nregisters for the whole K loop. Cycles.\n\n"
       << "| K | fma, sync | mma, sync | change | mma, staged ahead | change | "
          "+ wide fragments | change | + halves | change |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf),
                      "| %u | %s | %s | **%.1f%%** | %s | **%.1f%%** | %s | "
                      "**%.1f%%** | %s | **%.1f%%** |\n",
                      r.k, with_commas(r.fma_sync.cycles).c_str(),
                      with_commas(r.mma_sync.cycles).c_str(),
                      change(r.fma_sync.cycles, r.mma_sync.cycles),
                      with_commas(r.mma_async.cycles).c_str(),
                      change(r.mma_sync.cycles, r.mma_async.cycles),
                      with_commas(r.mma_wide.cycles).c_str(),
                      change(r.mma_async.cycles, r.mma_wide.cycles),
                      with_commas(r.mma_half.cycles).c_str(),
                      change(r.mma_wide.cycles, r.mma_half.cycles));
        md << buf;
    }
    std::snprintf(
        buf, sizeof(buf),
        "\nIssued work at K = %u: %s, %s, %s, %s and %s. The middle three are the "
        "same\nkernel: a fragment is eight floats however it is asked for, and what "
        "the wide\nload removes is the seven waits. The last is four floats and a "
        "multiply priced\nat half.\n",
        rows.back().k, with_commas(rows.back().fma_sync.weighted).c_str(),
        with_commas(rows.back().mma_sync.weighted).c_str(),
        with_commas(rows.back().mma_async.weighted).c_str(),
        with_commas(rows.back().mma_wide.weighted).c_str(),
        with_commas(rows.back().mma_half.weighted).c_str());
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
