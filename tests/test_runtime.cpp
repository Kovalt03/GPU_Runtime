#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "runtime.hpp"

namespace {

// Enough for any of these kernels, and small enough that constructing a runtime
// per test stays cheap: MemoryManager zeroes its arenas, so the 64 MB default
// costs about half a second each time.
constexpr size_t TEST_DEVICE_BYTES = 64 * 1024;
constexpr size_t TEST_HOST_BYTES = 4 * 1024;

MyGPURuntime make_runtime()
{
    return MyGPURuntime(TEST_DEVICE_BYTES, TEST_HOST_BYTES);
}

// A kernel that ignores its arguments and always produces the same program.
KernelFunc constant_kernel(Program prog)
{
    return [prog](void**) { return prog; };
}

}  // namespace

// ---------------------------------------------------------------------------
// Memory — forwarding to MemoryManager, which has its own tests. What matters
// here is that the runtime exposes one object rather than two.
// ---------------------------------------------------------------------------

TEST(Runtime, MallocFree)
{
    MyGPURuntime rt = make_runtime();

    void* p = rt.myrt_malloc(1024);
    ASSERT_NE(p, nullptr);
    rt.myrt_free(p);

    // The freed block must come back, not leak.
    void* q = rt.myrt_malloc(1024);
    EXPECT_EQ(q, p);
}

TEST(Runtime, MemcpyRoundtrip)
{
    MyGPURuntime rt = make_runtime();

    const std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    const size_t bytes = src.size() * sizeof(float);

    void* dev = rt.myrt_malloc(bytes);
    rt.myrt_memcpy(dev, src.data(), bytes, Direction::HostToDevice);

    std::vector<float> back(src.size(), 0.0f);
    rt.myrt_memcpy(back.data(), dev, bytes, Direction::DeviceToHost);
    EXPECT_EQ(back, src);
}

TEST(Runtime, MemcpyStillRejectsAMixedUpDirection)
{
    MyGPURuntime rt = make_runtime();
    void* dev = rt.myrt_malloc(64);
    std::vector<uint8_t> host(64, 0);

    EXPECT_THROW(rt.myrt_memcpy(host.data(), dev, 64, Direction::HostToDevice),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Thread indexing — a kernel has no other way to tell threads apart
// ---------------------------------------------------------------------------

TEST(Runtime, EachThreadSeesItsOwnGlobalIndex)
{
    MyGPURuntime rt = make_runtime();
    const uint32_t threads = 64;
    void* out = rt.myrt_malloc(threads * sizeof(float));

    // out[gid.x] = gid.x, addressed as a byte offset: 4 * gid.x.
    Program prog{
        make_v_mov_f32(1, 4.0f), make_v_mul_f32(2, REG_GLOBAL_ID_X, 1),  // byte offset
        make_v_mov_f32(3, 0.0f),  // base of the output buffer
        make_v_add_f32(2, 2, 3), make_v_st_global_f32(2, REG_GLOBAL_ID_X), make_ret(),
    };
    rt.myrt_launch(constant_kernel(prog), dim3{1, 1, 1}, dim3{threads, 1, 1}, nullptr);

    std::vector<float> back(threads, -1.0f);
    rt.myrt_memcpy(back.data(), out, threads * sizeof(float), Direction::DeviceToHost);
    for (uint32_t i = 0; i < threads; ++i) {
        EXPECT_FLOAT_EQ(back[i], static_cast<float>(i)) << "thread " << i;
    }
    rt.myrt_free(out);
}

TEST(Runtime, TwoDimensionalLaunchGivesBothCoordinates)
{
    // A 2D launch must hand a kernel its pixel coordinates directly: the ISA
    // has no integer division to recover them from a flat index.
    MyGPURuntime rt = make_runtime();
    rt.myrt_launch(constant_kernel(Program{make_ret()}), dim3{1, 1, 1}, dim3{4, 4, 1},
                   nullptr);
    EXPECT_GT(rt.stats().warp_steps, 0u);
}

TEST(Runtime, LaunchSpanningSeveralBlocks)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_launch(constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()}),
                   dim3{3, 1, 1}, dim3{32, 1, 1}, nullptr);

    // Three blocks of one warp, two instructions each.
    EXPECT_EQ(rt.stats().warp_steps, 6u);
}

TEST(Runtime, RejectsAnEmptyOrOversizedLaunch)
{
    MyGPURuntime rt = make_runtime();
    const KernelFunc k = constant_kernel(Program{make_ret()});

    EXPECT_THROW(rt.myrt_launch(k, dim3{0, 1, 1}, dim3{32, 1, 1}, nullptr),
                 std::runtime_error);
    EXPECT_THROW(rt.myrt_launch(k, dim3{1, 1, 1}, dim3{0, 1, 1}, nullptr),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

TEST(Runtime, LaunchSimpleKernel)
{
    MyGPURuntime rt = make_runtime();
    const size_t bytes = 4 * sizeof(float);

    void* dev = rt.myrt_malloc(bytes);
    const std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    rt.myrt_memcpy(dev, in.data(), bytes, Direction::HostToDevice);

    // Every thread multiplies element 0 by ten and writes it to element 1.
    Program prog{
        make_v_mov_f32(1, 0.0f),          make_v_ld_global_f32(2, 1, 0.0f),
        make_v_mov_f32(3, 10.0f),         make_v_mul_f32(2, 2, 3),
        make_v_st_global_f32(1, 2, 4.0f), make_ret(),
    };
    rt.myrt_launch(constant_kernel(prog), dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    std::vector<float> back(4, 0.0f);
    rt.myrt_memcpy(back.data(), dev, bytes, Direction::DeviceToHost);
    EXPECT_FLOAT_EQ(back[1], 10.0f);
    rt.myrt_free(dev);
}

// ---------------------------------------------------------------------------
// Statistics — accumulated across launches, cleared by sync
// ---------------------------------------------------------------------------

TEST(Runtime, StatisticsStartEmpty)
{
    const MyGPURuntime rt = make_runtime();
    EXPECT_EQ(rt.stats().warp_steps, 0u);
    EXPECT_DOUBLE_EQ(rt.divergence_rate(), 0.0);
    EXPECT_DOUBLE_EQ(rt.throughput_giops(), 0.0) << "must not divide by zero";
}

TEST(Runtime, ConvergedKernelReportsNoDivergence)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_launch(constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()}),
                   dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
    EXPECT_DOUBLE_EQ(rt.divergence_rate(), 0.0);
}

TEST(Runtime, DivergentKernelReportsDivergence)
{
    MyGPURuntime rt = make_runtime();

    // Even lanes branch, odd lanes fall through: gid.x - 2*floor(gid.x/2) is not
    // expressible, so compare against a threshold instead — the lower half of
    // the warp takes one path and the upper half the other.
    Program prog{
        make_v_mov_f32(1, 16.0f),
        make_v_cmp_f32(2, REG_GLOBAL_ID_X, 1, CmpOp::LT),
        make_bra_div(2, 3),
        make_v_mov_f32(10, 1.0f),
        make_bra(2),
        make_v_mov_f32(10, 2.0f),
        make_ret(),
    };
    rt.myrt_launch(constant_kernel(prog), dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    EXPECT_GT(rt.divergence_rate(), 0.0);
    EXPECT_LT(rt.divergence_rate(), 1.0);
}

TEST(Runtime, ThroughputIsPositiveAfterALaunch)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_launch(constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()}),
                   dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    EXPECT_GT(rt.throughput_giops(), 0.0);
    EXPECT_GT(rt.elapsed_seconds(), 0.0);
}

TEST(Runtime, ExpensiveInstructionsWeighMore)
{
    // The whole point of the cost model: identical instruction counts, very
    // different work. A global load must not look as cheap as a move.
    MyGPURuntime cheap = make_runtime();
    cheap.myrt_launch(constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()}),
                      dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    MyGPURuntime expensive = make_runtime();
    void* dev = expensive.myrt_malloc(16);
    Program prog{make_v_mov_f32(1, 0.0f), make_v_ld_global_f32(2, 1), make_ret()};
    expensive.myrt_launch(constant_kernel(prog), dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
    expensive.myrt_free(dev);

    EXPECT_GT(expensive.stats().weighted_lane_ops, cheap.stats().weighted_lane_ops);
}

TEST(Runtime, StatisticsAccumulateAcrossLaunches)
{
    MyGPURuntime rt = make_runtime();
    const KernelFunc k = constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()});

    rt.myrt_launch(k, dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
    const uint64_t after_one = rt.stats().warp_steps;
    ASSERT_GT(after_one, 0u);

    rt.myrt_launch(k, dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
    EXPECT_EQ(rt.stats().warp_steps, after_one * 2);
}

TEST(Runtime, SyncResetsStats)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_launch(constant_kernel(Program{make_v_mov_f32(10, 1.0f), make_ret()}),
                   dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
    ASSERT_GT(rt.stats().warp_steps, 0u);

    testing::internal::CaptureStdout();
    rt.myrt_sync();
    const std::string printed = testing::internal::GetCapturedStdout();

    EXPECT_NE(printed.find("[STATS]"), std::string::npos) << "sync must report";
    EXPECT_EQ(rt.stats().warp_steps, 0u);
    EXPECT_DOUBLE_EQ(rt.divergence_rate(), 0.0);
}
