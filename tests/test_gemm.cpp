#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "gemm.hpp"

// The tiled matrix multiply, which exists to give V_MMA_16X16X16_F32 and
// cp.async a kernel. Four routes — the matrix unit or the arithmetic pipe,
// staging synchronously or a chunk ahead — and all four have to agree with a
// host reference and with each other.

namespace {

std::vector<float> ramp(size_t n, int period, float scale)
{
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        // Nothing symmetric and nothing constant: a transposed operand or a
        // fragment laid down in the wrong order passes against either.
        v[i] = static_cast<float>(static_cast<int>(i % period) - period / 2) * scale;
    }
    return v;
}

struct Measurement {
    std::vector<float> c;
    uint64_t weighted = 0;
    uint64_t cycles = 0;
};

struct GemmOptions {
    bool matrix_unit = true;
    bool wide_fragments = false;
    TileStaging staging = TileStaging::Synchronous;
};

Measurement multiply(uint32_t m, uint32_t n, uint32_t k, const GemmOptions& how)
{
    const std::vector<float> a = ramp(static_cast<size_t>(m) * k, 7, 0.5f);
    const std::vector<float> b = ramp(static_cast<size_t>(k) * n, 5, 0.25f);

    MyGPURuntime rt(1u << 24);

    // The cost models the figures in benchmarks/RESULTS.md are taken under. The
    // flat one charges every lane of a staging load in full, which drowns the
    // difference between the two inner loops in the fetch they share.
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

    void* da = rt.myrt_malloc(a.size() * sizeof(float));
    void* db = rt.myrt_malloc(b.size() * sizeof(float));
    void* dc = rt.myrt_malloc(static_cast<size_t>(m) * n * sizeof(float));
    rt.myrt_memcpy(da, a.data(), a.size() * sizeof(float), Direction::HostToDevice);
    rt.myrt_memcpy(db, b.data(), b.size() * sizeof(float), Direction::HostToDevice);

    GemmArgs args;
    args.a_offset = rt.myrt_device_offset(da);
    args.b_offset = rt.myrt_device_offset(db);
    args.c_offset = rt.myrt_device_offset(dc);
    args.m = m;
    args.n = n;
    args.k = k;
    args.matrix_unit = how.matrix_unit;
    args.wide_fragments = how.wide_fragments;
    args.staging = how.staging;
    run_gemm(rt, args);

    Measurement run;
    run.c.resize(static_cast<size_t>(m) * n);
    rt.myrt_memcpy(run.c.data(), dc, run.c.size() * sizeof(float),
                   Direction::DeviceToHost);
    run.weighted = rt.stats().weighted_lane_ops;
    run.cycles = rt.stats().cycles;
    return run;
}

void agrees(const std::vector<float>& got, const std::vector<float>& want,
            const char* what)
{
    ASSERT_EQ(got.size(), want.size()) << what;
    for (size_t i = 0; i < want.size(); ++i) {
        // Exactly: the two orders of summation are the same one, k ascending,
        // so there is no rounding to allow for. A tolerance here would hide a
        // fragment laid down in the wrong place.
        EXPECT_FLOAT_EQ(got[i], want[i]) << what << " element " << i;
    }
}

}  // namespace

TEST(Gemm, TheMatrixUnitComputesWhatTheHostDoes)
{
    constexpr uint32_t M = 32, N = 128, K = 32;
    const std::vector<float> want =
        gemm_reference(ramp(M * K, 7, 0.5f), ramp(K * N, 5, 0.25f), M, N, K);
    agrees(multiply(M, N, K, GemmOptions{}).c, want, "matrix unit");
}

TEST(Gemm, EveryRouteAgreesWithEveryOther)
{
    // The four are one kernel with two flags, and the flags are supposed to
    // change what it costs rather than what it computes.
    constexpr uint32_t M = 32, N = 128, K = 32;
    const std::vector<float> want =
        gemm_reference(ramp(M * K, 7, 0.5f), ramp(K * N, 5, 0.25f), M, N, K);

    agrees(
        multiply(M, N, K, GemmOptions{true, false, TileStaging::AsyncDoubleBuffered}).c,
        want, "matrix unit, staged ahead");
    agrees(multiply(M, N, K, GemmOptions{false, false, TileStaging::Synchronous}).c, want,
           "arithmetic");
    agrees(
        multiply(M, N, K, GemmOptions{false, false, TileStaging::AsyncDoubleBuffered}).c,
        want, "arithmetic, staged ahead");
}

TEST(Gemm, OneKStepIsAsMuchAKernelAsSeveral)
{
    // K = 16 runs the loop once, so the double-buffered form issues a fill for a
    // step that does not exist and has to wait for it without walking it.
    constexpr uint32_t M = 16, N = 64, K = 16;
    const std::vector<float> want =
        gemm_reference(ramp(M * K, 7, 0.5f), ramp(K * N, 5, 0.25f), M, N, K);
    agrees(
        multiply(M, N, K, GemmOptions{true, false, TileStaging::AsyncDoubleBuffered}).c,
        want, "one k-step");
}

TEST(Gemm, TheMatrixUnitIssuesAFractionOfTheWork)
{
    constexpr uint32_t M = 32, N = 128, K = 32;
    const Measurement mma = multiply(M, N, K, GemmOptions{});
    const Measurement fma =
        multiply(M, N, K, GemmOptions{false, false, TileStaging::Synchronous});

    EXPECT_LT(mma.weighted, fma.weighted / 2)
        << "one instruction against 128 multiply-adds a lane";
    EXPECT_LT(mma.cycles, fma.cycles);
}

TEST(Gemm, StagingAheadPaysHereWhereItDidNotInTheRenderer)
{
    // The measurement cp.async was waiting for. A staged tile is consumed by a
    // fixed, small number of operations, so the fetch is a real fraction of the
    // work — the opposite of a tile read by 256 threads.
    constexpr uint32_t M = 32, N = 128, K = 64;
    const Measurement sync = multiply(M, N, K, GemmOptions{});
    const Measurement async =
        multiply(M, N, K, GemmOptions{true, false, TileStaging::AsyncDoubleBuffered});

    EXPECT_LT(async.cycles, sync.cycles * 9 / 10) << "at least a tenth of the cycles";
}

TEST(Gemm, AShapeWithATailIsRefused)
{
    MyGPURuntime rt(1u << 20);
    GemmArgs args;
    args.m = 16;
    args.n = 32;  // not a whole number of 64-column strips
    args.k = 16;
    EXPECT_THROW(run_gemm(rt, args), std::runtime_error);

    args.n = 64;
    args.k = 24;  // not a whole number of tiles
    EXPECT_THROW(run_gemm(rt, args), std::runtime_error);
}

TEST(Gemm, AWideFragmentLoadComputesTheSameProduct)
{
    // Eight loads or one: the registers have to come back holding the same
    // fragment, in the same order, or the multiply reads a transposed operand.
    constexpr uint32_t M = 32, N = 128, K = 32;
    const std::vector<float> want =
        gemm_reference(ramp(M * K, 7, 0.5f), ramp(K * N, 5, 0.25f), M, N, K);

    GemmOptions wide;
    wide.matrix_unit = true;
    wide.wide_fragments = true;
    agrees(multiply(M, N, K, wide).c, want, "wide fragment load");

    wide.staging = TileStaging::AsyncDoubleBuffered;
    agrees(multiply(M, N, K, wide).c, want, "wide fragment load, staged ahead");
}

TEST(Gemm, AWideFragmentLoadBuysTimeRatherThanCapacity)
{
    // The distinction this instruction is priced on. Eight floats cost eight
    // floats' worth of issue capacity however they are asked for — shared memory
    // has banks rather than lines, so there is no transaction to save. What it
    // saves is the seven waits it does not take.
    constexpr uint32_t M = 32, N = 128, K = 64;
    GemmOptions narrow;
    narrow.matrix_unit = true;
    GemmOptions wide = narrow;
    wide.wide_fragments = true;

    const Measurement eight = multiply(M, N, K, narrow);
    const Measurement one = multiply(M, N, K, wide);

    EXPECT_LT(one.cycles, eight.cycles * 3 / 4) << "the waiting is what it removes";
    EXPECT_EQ(one.weighted, eight.weighted)
        << "and the capacity is the same to the lane-op: eight floats are eight "
           "floats however they are asked for";
}
