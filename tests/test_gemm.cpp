#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "gemm.hpp"
#include "half.hpp"

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
    bool half_inputs = false;
    TileStaging staging = TileStaging::Synchronous;
};

// The operands every test uses unless it hands over its own. Multiples of a half
// and a quarter, so that the single-precision routes can be checked exactly —
// the half-precision one needs numbers that are not representable and says so.
std::vector<float> default_a(uint32_t m, uint32_t k)
{
    return ramp(static_cast<size_t>(m) * k, 7, 0.5f);
}

std::vector<float> default_b(uint32_t k, uint32_t n)
{
    return ramp(static_cast<size_t>(k) * n, 5, 0.25f);
}

Measurement multiply(uint32_t m, uint32_t n, uint32_t k, const GemmOptions& how,
                     const std::vector<float>* operand_a = nullptr,
                     const std::vector<float>* operand_b = nullptr)
{
    const std::vector<float> a = operand_a != nullptr ? *operand_a : default_a(m, k);
    const std::vector<float> b = operand_b != nullptr ? *operand_b : default_b(k, n);

    // The device never converts: a kernel multiplying halves reads them already
    // packed, so putting them there is the caller's job — here and on hardware.
    const std::vector<float> a_device = how.half_inputs ? pack_halves(a) : a;
    const std::vector<float> b_device = how.half_inputs ? pack_halves(b) : b;

    MyGPURuntime rt(1u << 24);

    // The cost models the figures in benchmarks/RESULTS.md are taken under. The
    // flat one charges every lane of a staging load in full, which drowns the
    // difference between the two inner loops in the fetch they share.
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

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
        gemm_reference(default_a(M, K), default_b(K, N), M, N, K);
    agrees(multiply(M, N, K, GemmOptions{}).c, want, "matrix unit");
}

TEST(Gemm, EveryRouteAgreesWithEveryOther)
{
    // The four are one kernel with two flags, and the flags are supposed to
    // change what it costs rather than what it computes.
    constexpr uint32_t M = 32, N = 128, K = 32;
    const std::vector<float> want =
        gemm_reference(default_a(M, K), default_b(K, N), M, N, K);

    agrees(multiply(M, N, K,
                    GemmOptions{true, false, false, TileStaging::AsyncDoubleBuffered})
               .c,
           want, "matrix unit, staged ahead");
    agrees(
        multiply(M, N, K, GemmOptions{false, false, false, TileStaging::Synchronous}).c,
        want, "arithmetic");
    agrees(multiply(M, N, K,
                    GemmOptions{false, false, false, TileStaging::AsyncDoubleBuffered})
               .c,
           want, "arithmetic, staged ahead");
}

TEST(Gemm, OneKStepIsAsMuchAKernelAsSeveral)
{
    // K = 16 runs the loop once, so the double-buffered form issues a fill for a
    // step that does not exist and has to wait for it without walking it.
    constexpr uint32_t M = 16, N = 64, K = 16;
    const std::vector<float> want =
        gemm_reference(default_a(M, K), default_b(K, N), M, N, K);
    agrees(multiply(M, N, K,
                    GemmOptions{true, false, false, TileStaging::AsyncDoubleBuffered})
               .c,
           want, "one k-step");
}

TEST(Gemm, TheMatrixUnitIssuesAFractionOfTheWork)
{
    constexpr uint32_t M = 32, N = 128, K = 32;
    const Measurement mma = multiply(M, N, K, GemmOptions{});
    const Measurement fma =
        multiply(M, N, K, GemmOptions{false, false, false, TileStaging::Synchronous});

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
    const Measurement async = multiply(
        M, N, K, GemmOptions{true, false, false, TileStaging::AsyncDoubleBuffered});

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
        gemm_reference(default_a(M, K), default_b(K, N), M, N, K);

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

// ---------------------------------------------------------------------------
// Half precision — the width is saved on the inputs and kept on the sum
// ---------------------------------------------------------------------------

TEST(Gemm, HalfInputsComputeTheRoundedProductExactly)
{
    // Not the reference's answer, and not approximately the reference's answer:
    // exactly the answer the reference gives when its inputs are rounded first.
    // That is what the narrow type costs, stated as an equality rather than a
    // tolerance — a tolerance would pass a kernel that lost a fragment as well.
    constexpr uint32_t M = 32, N = 128, K = 32;
    const std::vector<float> a = ramp(M * K, 7, 0.1f);
    const std::vector<float> b = ramp(K * N, 5, 0.3f);

    GemmOptions half;
    half.half_inputs = true;
    const std::vector<float> got = multiply(M, N, K, half, &a, &b).c;

    agrees(got, gemm_reference(rounded_to_half(a), rounded_to_half(b), M, N, K),
           "half inputs against the rounded reference");

    // And it is a different answer from the exact one, or the test above would
    // be measuring nothing: 0.1 and 0.3 are not half-precision numbers.
    const std::vector<float> exact = gemm_reference(a, b, M, N, K);
    bool differs = false;
    for (size_t i = 0; i < exact.size(); ++i) {
        differs = differs || got[i] != exact[i];
    }
    EXPECT_TRUE(differs) << "the inputs were representable, so nothing was rounded";
}

TEST(Gemm, HalfInputsCostLessToStageAndToMultiply)
{
    constexpr uint32_t M = 32, N = 128, K = 64;
    GemmOptions wide;
    wide.wide_fragments = true;
    GemmOptions half;
    half.half_inputs = true;

    const Measurement single = multiply(M, N, K, wide);
    const Measurement narrow = multiply(M, N, K, half);

    // Half the bytes staged, half the registers an operand, and an instruction
    // priced at half.
    EXPECT_LT(narrow.weighted, single.weighted * 3 / 4);
    EXPECT_LT(narrow.cycles, single.cycles);
}

TEST(Gemm, HalfPrecisionRoundsTheWayHardwareDoes)
{
    // The conversion is written out in half.cpp rather than left to an
    // intrinsic, so it is worth checking the corners it was written for.
    EXPECT_EQ(f16_to_f32(f32_to_f16(2.5f)), 2.5f) << "exactly representable";
    EXPECT_NE(f16_to_f32(f32_to_f16(0.1f)), 0.1f) << "not representable";
    EXPECT_NEAR(f16_to_f32(f32_to_f16(0.1f)), 0.1f, 1e-4f);

    EXPECT_TRUE(std::isinf(f16_to_f32(f32_to_f16(65600.0f)))) << "saturates";
    EXPECT_EQ(f16_to_f32(f32_to_f16(1e-8f)), 0.0f) << "underflows to zero";
    EXPECT_LT(f16_to_f32(f32_to_f16(-0.7f)), 0.0f) << "keeps its sign";

    // Two to a register, low first, which is the order the F16 instructions
    // unpack them in.
    uint16_t low = 0;
    uint16_t high = 0;
    unpack_f16x2(pack_f16x2(f32_to_f16(1.0f), f32_to_f16(2.0f)), low, high);
    EXPECT_EQ(f16_to_f32(low), 1.0f);
    EXPECT_EQ(f16_to_f32(high), 2.0f);
}
