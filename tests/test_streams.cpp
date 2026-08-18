#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "runtime.hpp"

// Streams and indirect launches — what a queue buys once the machine is wide
// enough to run two things at once, and what a launch looks like when the host
// never learns its size.

namespace {

constexpr size_t TEST_DEVICE_BYTES = 64 * 1024;
constexpr size_t TEST_HOST_BYTES = 4 * 1024;

MyGPURuntime make_runtime()
{
    return MyGPURuntime(TEST_DEVICE_BYTES, TEST_HOST_BYTES);
}

KernelFunc constant_kernel(Program prog)
{
    return [prog](void**) { return prog; };
}

// out[gid.x] = value, with out at byte offset base.
Program store_kernel(size_t base, float value)
{
    return Program{
        make_v_mov_f32(1, 4.0f),
        make_v_mul_f32(2, REG_GLOBAL_ID_X, 1),
        make_v_mov_f32(3, static_cast<float>(base)),
        make_v_add_f32(2, 2, 3),
        make_v_mov_f32(4, value),
        make_v_st_global_f32(2, 4),
        make_ret(),
    };
}

// out[gid.x] += 1. Reads what another launch wrote, which is what makes the
// order of two launches observable.
Program increment_kernel(size_t base)
{
    return Program{
        make_v_mov_f32(1, 4.0f),
        make_v_mul_f32(2, REG_GLOBAL_ID_X, 1),
        make_v_mov_f32(3, static_cast<float>(base)),
        make_v_add_f32(2, 2, 3),
        make_v_ld_global_f32(5, 2),
        make_v_mov_f32(6, 1.0f),
        make_v_add_f32(5, 5, 6),
        make_v_st_global_f32(2, 5),
        make_ret(),
    };
}

// A chain of dependent loads from one address. Under LatencyModel::Modelled each
// one waits for the last, so a warp on its own leaves the SM idle almost all the
// time — the room another stream's warp is there to use.
Program dependent_loads(size_t base, int count)
{
    Program prog{make_v_mov_f32(1, static_cast<float>(base))};
    for (int i = 0; i < count; ++i) {
        prog.push_back(make_v_ld_global_f32(2, 1));
    }
    prog.push_back(make_ret());
    return prog;
}

std::vector<float> read_back(MyGPURuntime& rt, void* dev, size_t count)
{
    std::vector<float> back(count, -1.0f);
    rt.myrt_memcpy(back.data(), dev, count * sizeof(float), Direction::DeviceToHost);
    return back;
}

// Two SMs, so that two streams have somewhere to be at once. The default machine
// is one SM holding one block, where a stream changes nothing.
SMConfig two_sms()
{
    SMConfig cfg;
    cfg.sm_count = 2;
    cfg.blocks_per_sm = 1;
    return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// Queueing — what "async" means when the machine is a function call
// ---------------------------------------------------------------------------

TEST(Streams, AQueuedLaunchWaitsForTheDrain)
{
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(32 * sizeof(float));
    const size_t base = rt.myrt_device_offset(out);

    rt.myrt_launch_async(constant_kernel(store_kernel(base, 7.0f)),
                         LaunchConfig{dim3{1, 1, 1}, dim3{32, 1, 1}}, nullptr);
    EXPECT_EQ(rt.stats().warp_steps, 0u) << "queued work must not have run yet";

    rt.myrt_wait();
    EXPECT_GT(rt.stats().warp_steps, 0u);
    EXPECT_FLOAT_EQ(read_back(rt, out, 32)[0], 7.0f);
}

TEST(Streams, ASynchronousLaunchDrainsWhatIsAlreadyQueued)
{
    // A launch that returns having run means the machine is idle behind it. If
    // queued work outlived it, that would not be true.
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(32 * sizeof(float));
    const size_t base = rt.myrt_device_offset(out);
    const StreamId other = rt.myrt_stream_create();

    rt.myrt_launch_async(constant_kernel(store_kernel(base, 3.0f)),
                         LaunchConfig{dim3{1, 1, 1}, dim3{32, 1, 1}}, nullptr, other);
    rt.myrt_launch(constant_kernel(store_kernel(base + 128, 4.0f)), dim3{1, 1, 1},
                   dim3{32, 1, 1}, nullptr);

    EXPECT_FLOAT_EQ(read_back(rt, out, 32)[0], 3.0f);
    EXPECT_FLOAT_EQ(read_back(rt, out, 64)[32], 4.0f);
}

TEST(Streams, LaunchesOnOneStreamRunInOrder)
{
    // The second reads what the first wrote. Out of order, a block of the
    // increment could read a cell the store had not reached.
    MyGPURuntime rt = make_runtime();
    rt.myrt_set_sm_config(two_sms());

    const uint32_t threads = 256;
    void* out = rt.myrt_malloc(threads * sizeof(float));
    const size_t base = rt.myrt_device_offset(out);

    const LaunchConfig config{dim3{8, 1, 1}, dim3{32, 1, 1}};
    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), config, nullptr);
    rt.myrt_launch_async(constant_kernel(increment_kernel(base)), config, nullptr);
    rt.myrt_sync(false);

    const std::vector<float> back = read_back(rt, out, threads);
    for (uint32_t i = 0; i < threads; ++i) {
        EXPECT_FLOAT_EQ(back[i], 2.0f) << "thread " << i;
    }
}

TEST(Streams, AnUnknownStreamIsRejected)
{
    MyGPURuntime rt = make_runtime();
    EXPECT_THROW(
        rt.myrt_launch_async(constant_kernel(store_kernel(0, 1.0f)),
                             LaunchConfig{dim3{1, 1, 1}, dim3{32, 1, 1}}, nullptr, 9),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// Overlap — the whole reason for a second queue
// ---------------------------------------------------------------------------

TEST(Streams, TwoStreamsFillAMachineOneGridCannot)
{
    // One block cannot use two SMs. Two one-block launches can, if nothing says
    // they have to wait for each other.
    MyGPURuntime rt = make_runtime();
    rt.myrt_set_sm_config(two_sms());

    void* out = rt.myrt_malloc(64 * sizeof(float));
    const size_t base = rt.myrt_device_offset(out);
    const LaunchConfig config{dim3{1, 1, 1}, dim3{32, 1, 1}};

    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), config, nullptr);
    rt.myrt_launch_async(constant_kernel(store_kernel(base + 128, 2.0f)), config,
                         nullptr);
    rt.myrt_wait();
    const uint64_t ordered = rt.stats().cycles;
    rt.myrt_sync(false);  // clears, so that the second reading stands alone

    const StreamId other = rt.myrt_stream_create();
    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), config, nullptr);
    rt.myrt_launch_async(constant_kernel(store_kernel(base + 128, 2.0f)), config, nullptr,
                         other);
    rt.myrt_wait();
    const uint64_t concurrent = rt.stats().cycles;

    EXPECT_LT(concurrent, ordered);
    EXPECT_EQ(concurrent, ordered / 2) << "the two grids retire side by side";
}

TEST(Streams, AStreamCoversTheOtherStreamsWaiting)
{
    // One SM, two blocks resident, and every instruction waiting on the last. The
    // second stream issues into the gaps the first leaves, which is what
    // occupancy buys — here bought across kernels rather than within one.
    MyGPURuntime rt = make_runtime();
    SMConfig cfg;
    cfg.sm_count = 1;
    cfg.blocks_per_sm = 2;
    rt.myrt_set_sm_config(cfg);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

    void* buf = rt.myrt_malloc(128);
    const size_t base = rt.myrt_device_offset(buf);
    const Program prog = dependent_loads(base, 16);
    const LaunchConfig config{dim3{1, 1, 1}, dim3{32, 1, 1}};

    rt.myrt_launch_async(constant_kernel(prog), config, nullptr);
    rt.myrt_launch_async(constant_kernel(prog), config, nullptr);
    rt.myrt_wait();
    const uint64_t ordered = rt.stats().cycles;
    rt.myrt_sync(false);

    const StreamId other = rt.myrt_stream_create();
    rt.myrt_launch_async(constant_kernel(prog), config, nullptr);
    rt.myrt_launch_async(constant_kernel(prog), config, nullptr, other);
    rt.myrt_wait();

    EXPECT_LT(rt.stats().cycles, ordered);
    EXPECT_LT(rt.stats().stall_steps, ordered)
        << "the waiting is covered, not merely moved";
}

// ---------------------------------------------------------------------------
// Statistics — work partitions, time overlaps
// ---------------------------------------------------------------------------

TEST(Streams, WorkCountersPartitionAcrossStreams)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_set_sm_config(two_sms());

    void* out = rt.myrt_malloc(2048);
    const size_t base = rt.myrt_device_offset(out);
    const StreamId other = rt.myrt_stream_create();

    // Two grids of different sizes, so that an equal split cannot pass by luck.
    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)),
                         LaunchConfig{dim3{4, 1, 1}, dim3{32, 1, 1}}, nullptr);
    rt.myrt_launch_async(constant_kernel(increment_kernel(base + 1024)),
                         LaunchConfig{dim3{1, 1, 1}, dim3{32, 1, 1}}, nullptr, other);
    rt.myrt_wait();

    const SchedulerStats& a = rt.myrt_stream_stats(DEFAULT_STREAM);
    const SchedulerStats& b = rt.myrt_stream_stats(other);
    EXPECT_EQ(a.warp_steps + b.warp_steps, rt.stats().warp_steps);
    EXPECT_EQ(a.active_lane_ops + b.active_lane_ops, rt.stats().active_lane_ops);
    EXPECT_EQ(a.weighted_lane_ops + b.weighted_lane_ops, rt.stats().weighted_lane_ops);
    EXPECT_GT(a.warp_steps, b.warp_steps) << "four blocks against one";
}

TEST(Streams, AStreamIsChargedWhatItWouldHaveCostAlone)
{
    // Attribution is exact, not apportioned: the work a stream did concurrently
    // is to the instruction what it did on an empty machine.
    MyGPURuntime rt = make_runtime();
    rt.myrt_set_sm_config(two_sms());

    void* out = rt.myrt_malloc(2048);
    const size_t base = rt.myrt_device_offset(out);
    const LaunchConfig small{dim3{1, 1, 1}, dim3{32, 1, 1}};
    const LaunchConfig large{dim3{4, 1, 1}, dim3{32, 1, 1}};

    rt.myrt_launch(constant_kernel(store_kernel(base, 1.0f)), large.grid, large.block,
                   nullptr);
    const uint64_t alone = rt.stats().warp_steps;
    rt.myrt_sync(false);  // clears both the totals and the stream's share

    const StreamId other = rt.myrt_stream_create();
    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), large, nullptr);
    rt.myrt_launch_async(constant_kernel(increment_kernel(base + 1024)), small, nullptr,
                         other);
    rt.myrt_wait();

    EXPECT_EQ(rt.myrt_stream_stats(DEFAULT_STREAM).warp_steps, alone);
}

TEST(Streams, OverlappingStreamsAreBothChargedForTheCyclesTheyShared)
{
    MyGPURuntime rt = make_runtime();
    rt.myrt_set_sm_config(two_sms());

    void* out = rt.myrt_malloc(1024);
    const size_t base = rt.myrt_device_offset(out);
    const StreamId other = rt.myrt_stream_create();
    const LaunchConfig config{dim3{1, 1, 1}, dim3{32, 1, 1}};

    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), config, nullptr);
    rt.myrt_launch_async(constant_kernel(store_kernel(base + 512, 2.0f)), config, nullptr,
                         other);
    rt.myrt_wait();

    const uint64_t a = rt.myrt_stream_stats(DEFAULT_STREAM).cycles;
    const uint64_t b = rt.myrt_stream_stats(other).cycles;
    EXPECT_GT(a + b, rt.stats().cycles)
        << "time is not divided between streams; both were there";
    EXPECT_EQ(a, rt.stats().cycles);
    EXPECT_EQ(b, rt.stats().cycles) << "the two ran side by side the whole way";
}

TEST(Streams, AStreamsTotalsStandAfterAWaitAndAreClearedByASync)
{
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(256);
    const size_t base = rt.myrt_device_offset(out);
    const LaunchConfig config{dim3{1, 1, 1}, dim3{32, 1, 1}};

    rt.myrt_launch_async(constant_kernel(store_kernel(base, 1.0f)), config, nullptr);
    rt.myrt_wait();
    EXPECT_GT(rt.myrt_stream_stats(DEFAULT_STREAM).warp_steps, 0u);

    rt.myrt_sync(false);
    EXPECT_EQ(rt.myrt_stream_stats(DEFAULT_STREAM).warp_steps, 0u)
        << "a stream's share is cleared with the totals it belongs to";
}

// ---------------------------------------------------------------------------
// Indirect launches — a grid the host never sees
// ---------------------------------------------------------------------------

TEST(Streams, AnIndirectLaunchReadsItsGridFromDeviceMemory)
{
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(128 * sizeof(float));
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    const float dims[3] = {3.0f, 1.0f, 1.0f};
    rt.myrt_memcpy(grid, dims, sizeof(dims), Direction::HostToDevice);

    IndirectLaunchConfig config;
    config.grid_offset = rt.myrt_device_offset(grid);
    config.block = dim3{32, 1, 1};
    rt.myrt_launch_indirect(
        constant_kernel(store_kernel(rt.myrt_device_offset(out), 5.0f)), config, nullptr);
    rt.myrt_sync(false);

    const std::vector<float> back = read_back(rt, out, 128);
    for (uint32_t i = 0; i < 96; ++i) {
        EXPECT_FLOAT_EQ(back[i], 5.0f) << "thread " << i;
    }
    EXPECT_FLOAT_EQ(back[96], 0.0f) << "a fourth block would have written here";
}

TEST(Streams, TheKernelBeforeItDecidesAnIndirectLaunchsGrid)
{
    // The premise of GPU-driven rendering: the size of one launch is a result the
    // device computed, and the host is never told it.
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(128 * sizeof(float));
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    const size_t grid_offset = rt.myrt_device_offset(grid);

    const float dims[3] = {0.0f, 0.0f, 0.0f};
    rt.myrt_memcpy(grid, dims, sizeof(dims), Direction::HostToDevice);

    // One thread writes the grid the launch after it will run at.
    const Program writer{
        make_v_mov_f32(1, static_cast<float>(grid_offset)),
        make_v_mov_f32(2, 2.0f),
        make_v_st_global_f32(1, 2),
        make_v_mov_f32(3, 1.0f),
        make_v_st_global_f32(1, 3, 4.0f),
        make_v_st_global_f32(1, 3, 8.0f),
        make_ret(),
    };
    rt.myrt_launch_async(constant_kernel(writer),
                         LaunchConfig{dim3{1, 1, 1}, dim3{1, 1, 1}}, nullptr);

    IndirectLaunchConfig config;
    config.grid_offset = grid_offset;
    config.block = dim3{32, 1, 1};
    rt.myrt_launch_indirect(
        constant_kernel(store_kernel(rt.myrt_device_offset(out), 9.0f)), config, nullptr);
    rt.myrt_sync(false);

    const std::vector<float> back = read_back(rt, out, 128);
    EXPECT_FLOAT_EQ(back[63], 9.0f) << "two blocks ran";
    EXPECT_FLOAT_EQ(back[64], 0.0f) << "and no more than two";
}

TEST(Streams, AnEmptyIndirectGridRunsNothing)
{
    // Nothing to draw is a result a culling pass is entitled to reach.
    MyGPURuntime rt = make_runtime();
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    const float dims[3] = {0.0f, 1.0f, 1.0f};
    rt.myrt_memcpy(grid, dims, sizeof(dims), Direction::HostToDevice);

    IndirectLaunchConfig config;
    config.grid_offset = rt.myrt_device_offset(grid);
    config.block = dim3{32, 1, 1};
    rt.myrt_launch_indirect(constant_kernel(store_kernel(0, 1.0f)), config, nullptr);
    rt.myrt_wait();

    EXPECT_EQ(rt.stats().warp_steps, 0u);
    EXPECT_EQ(rt.stats().cycles, 0u);
}

TEST(Streams, AnIndirectGridMustBeWholeNumbers)
{
    MyGPURuntime rt = make_runtime();
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    const float dims[3] = {1.5f, 1.0f, 1.0f};
    rt.myrt_memcpy(grid, dims, sizeof(dims), Direction::HostToDevice);

    IndirectLaunchConfig config;
    config.grid_offset = rt.myrt_device_offset(grid);
    config.block = dim3{32, 1, 1};
    rt.myrt_launch_indirect(constant_kernel(store_kernel(0, 1.0f)), config, nullptr);
    EXPECT_THROW(rt.myrt_wait(), std::runtime_error);
}

TEST(Streams, AnIndirectGridMustLieInDeviceMemory)
{
    MyGPURuntime rt = make_runtime();
    IndirectLaunchConfig config;
    config.grid_offset = TEST_DEVICE_BYTES - sizeof(float);
    config.block = dim3{32, 1, 1};
    rt.myrt_launch_indirect(constant_kernel(store_kernel(0, 1.0f)), config, nullptr);
    EXPECT_THROW(rt.myrt_wait(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Clusters — blocks placed together, and addressing each other's shared memory
// ---------------------------------------------------------------------------

TEST(Streams, ABlockReadsItsNeighboursSharedMemory)
{
    // What distributed shared memory is for: one block stages something and the
    // others read it across rather than fetching it again.
    MyGPURuntime rt = make_runtime();
    SMConfig cfg;
    cfg.sm_count = 4;
    cfg.blocks_per_sm = 1;
    rt.myrt_set_sm_config(cfg);

    void* out = rt.myrt_malloc(4 * sizeof(float));
    const size_t base = rt.myrt_device_offset(out);

    // Rank 0 writes its own shared memory; everyone reads rank 0's and stores it.
    const Program prog{
        make_v_mov_f32(1, 0.0f),  // shared address 0
        make_v_mov_f32(2, 7.0f),  // the value
        make_v_cmp_f32(3, REG_CLUSTER_RANK, 1, CmpOp::GT),
        make_bra_div(3, 2),  // ranks above 0 skip the store
        make_v_st_shared_f32(1, 2),
        make_barrier_cluster(),
        make_v_mov_f32(4, 0.0f),         // rank 0
        make_v_ld_cluster_f32(5, 1, 4),  // read rank 0's shared memory
        make_v_mov_f32(6, 4.0f),
        make_v_mul_f32(7, REG_CLUSTER_RANK, 6),
        make_v_mov_f32(8, static_cast<float>(base)),
        make_v_add_f32(7, 7, 8),
        make_v_st_global_f32(7, 5),
        make_ret(),
    };

    LaunchConfig config{dim3{4, 1, 1}, dim3{32, 1, 1}};
    config.cluster_size = 4;
    rt.myrt_launch_async(constant_kernel(prog), config, nullptr);
    rt.myrt_wait();

    const std::vector<float> back = read_back(rt, out, 4);
    for (uint32_t rank = 0; rank < 4; ++rank) {
        EXPECT_FLOAT_EQ(back[rank], 7.0f) << "rank " << rank << " read nothing";
    }
}

TEST(Streams, ARankOutsideTheClusterIsRefused)
{
    MyGPURuntime rt = make_runtime();
    const Program prog{
        make_v_mov_f32(1, 0.0f),
        make_v_mov_f32(2, 3.0f),  // rank 3 of a cluster of one
        make_v_ld_cluster_f32(3, 1, 2),
        make_ret(),
    };
    rt.myrt_launch_async(constant_kernel(prog),
                         LaunchConfig{dim3{1, 1, 1}, dim3{32, 1, 1}}, nullptr);
    EXPECT_THROW(rt.myrt_wait(), std::runtime_error);
}

TEST(Streams, ALoneBlockIsAClusterOfOne)
{
    // A kernel written for clusters runs unchanged without one: rank 0 is itself.
    MyGPURuntime rt = make_runtime();
    void* out = rt.myrt_malloc(sizeof(float));
    const size_t base = rt.myrt_device_offset(out);

    rt.myrt_launch(constant_kernel(Program{
                       make_v_mov_f32(1, 0.0f),
                       make_v_mov_f32(2, 5.0f),
                       make_v_st_shared_f32(1, 2),
                       make_barrier_cluster(),
                       make_v_ld_cluster_f32(3, 1, 1),
                       make_v_mov_f32(4, static_cast<float>(base)),
                       make_v_st_global_f32(4, 3),
                       make_ret(),
                   }),
                   dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    EXPECT_FLOAT_EQ(read_back(rt, out, 1)[0], 5.0f);
}

TEST(Streams, AClusterTooBigForTheMachineIsRefused)
{
    // The blocks have to be co-resident, so a cluster the machine cannot hold is
    // a launch that cannot start rather than one that runs a piece at a time.
    MyGPURuntime rt = make_runtime();
    LaunchConfig config{dim3{8, 1, 1}, dim3{32, 1, 1}};
    config.cluster_size = 8;  // the default machine holds one block
    rt.myrt_launch_async(constant_kernel(Program{make_ret()}), config, nullptr);
    EXPECT_THROW(rt.myrt_wait(), std::runtime_error);
}
