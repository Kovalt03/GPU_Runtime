#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "isa.hpp"        // Program, for KernelFunc
#include "memory.hpp"     // MemoryManager, Direction
#include "scheduler.hpp"  // WarpScheduler, SchedulerStats

// Launch geometry, named after CUDA's dim3 so the resemblance is deliberate.
struct dim3 {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;

    uint32_t volume() const
    {
        return x * y * z;
    }
};

// What a launch declares about itself beyond its shape.
//
// shared_bytes is how much of a block's scratchpad the kernel will use. It is
// not a bound — ThreadBlock carries the whole scratchpad either way — but one of
// the three things residency is the smallest of, and the reason a kernel that
// stages through shared memory fits fewer blocks on an SM. CUDA states it the
// same way, at the launch rather than in the kernel.
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_bytes = 0;

    // How many blocks are placed together and can read each other's shared
    // memory. One is what everything before this declared, and it means no
    // cluster at all rather than a cluster of one — though a kernel written for
    // clusters still runs, reading rank 0 and finding itself.
    //
    // The blocks of a cluster are placed together or not at all, and none of them
    // is freed until all of them have retired. That is what makes a neighbour's
    // shared memory safe to address, and it is also what makes a cluster larger
    // than the machine can hold a launch that cannot start.
    uint32_t cluster_size = 1;
};

// Where an indirect launch reads its grid, and everything else a launch declares.
//
// The grid is three consecutive floats in device memory — x, y, z — rather than
// a dim3 the host filled in, which is the whole difference: a kernel can write
// them, so the size of one launch can be decided by the one before it. Floats
// because the ISA has no integer register to write them from, and whole numbers
// are exact to 2^24.
//
// A grid of zero is legal here where it is an error in LaunchConfig. Nothing to
// draw is a result a culling pass is entitled to reach, and a launch that
// refused it would make the host check what it just asked the device to decide.
struct IndirectLaunchConfig {
    size_t grid_offset = 0;
    dim3 block;
    size_t shared_bytes = 0;
};

// Which queue a launch waits in. Launches sharing one run in order; launches in
// different ones are unordered, and the machine interleaves their blocks.
//
// A plain integer rather than an opaque handle: there is no per-stream state to
// hold, ordering being decided at the drain from the order things were enqueued.
using StreamId = uint32_t;

// The stream a launch uses when it names none, and the one every synchronous
// launch uses.
inline constexpr StreamId DEFAULT_STREAM = 0;

// A kernel is a function that produces an instruction sequence, mirroring the
// compile step a real toolchain performs ahead of a launch. It is called once
// per launch, not once per thread: all threads run the same Program, and what
// differs between them is the register state seeded below.
using KernelFunc = std::function<Program(void**)>;

// --- thread indexing --------------------------------------------------------
// A kernel has no other way to tell which thread it is running as. The launch
// seeds these three registers in every thread with its global coordinates —
// blockIdx * blockDim + threadIdx, in CUDA terms — before execution starts.
//
// Three coordinates rather than one flat index because the ISA has no integer
// division: a 256x256 image would otherwise need idx % 256 to recover a pixel
// column, which V_RCP_F32 cannot express exactly.
//
// They sit at the top of the register file so that kernels can allocate upward
// from r0 without colliding — Möller-Trumbore already claims r0..r42.
inline constexpr uint8_t REG_GLOBAL_ID_X = 253;
inline constexpr uint8_t REG_GLOBAL_ID_Y = 254;
inline constexpr uint8_t REG_GLOBAL_ID_Z = 255;

// Which block of its cluster this one is, seeded like the block ids below.
//
// A rank cannot be recovered from a block id without an integer division, and a
// kernel reading a neighbour's shared memory needs to know which neighbour it is
// itself. Zero for a launch that declared no cluster.
inline constexpr uint8_t REG_CLUSTER_RANK = 249;

// Which block a thread belongs to — CUDA's blockIdx, and the other half of the
// indexing model. A global coordinate cannot be turned back into one, the ISA
// having no integer division, so anything a block owns collectively has to be
// found through these.
//
// Reserved below the global ids, which lowers the allocator's ceiling to 250.
// Kernels here use around sixty registers, so the room costs nothing.
inline constexpr uint8_t REG_BLOCK_ID_X = 250;
inline constexpr uint8_t REG_BLOCK_ID_Y = 251;
inline constexpr uint8_t REG_BLOCK_ID_Z = 252;

class MyGPURuntime {
public:
    // 64 MB of device memory by default; host allocations are rarely needed,
    // since myrt_memcpy accepts any ordinary address on the host side.
    explicit MyGPURuntime(size_t device_mem_size = 64u * 1024 * 1024,
                          size_t host_mem_size = 1024 * 1024);

    // --- memory -------------------------------------------------------------
    void* myrt_malloc(size_t size);
    void myrt_free(void* ptr);
    void myrt_memcpy(void* dst, const void* src, size_t size, Direction dir);

    // Byte offset of a device allocation from the base of device memory, which
    // is the form a kernel addresses it in. Nothing in the ISA can dereference
    // a host pointer, so this is how a buffer's address reaches a register.
    size_t myrt_device_offset(const void* ptr) const;

    // How much of the device arena is unallocated. cudaMemGetInfo's half of the
    // pair, and what says whether releasing a buffer actually returned it —
    // MemoryManager coalesces on free, so a run that hands everything back ends
    // exactly where it started rather than merely close to it.
    size_t myrt_device_free_bytes() const;

    // --- execution ----------------------------------------------------------
    // Runs kernel(args) over grid x block threads. Execution is synchronous:
    // the call returns once every thread has retired. Throws
    // std::runtime_error if the launch geometry is empty or if a block holds
    // more threads than the simulator will place in one ThreadBlock.
    void myrt_launch(KernelFunc kernel, dim3 grid, dim3 block, void** args);

    // The same, with what else a launch declares about itself. Kept apart from
    // the four-argument form because every kernel and test calls that one, and a
    // defaulted parameter there would have made a shared-memory claim easy to
    // forget rather than easy to see.
    void myrt_launch(KernelFunc kernel, const LaunchConfig& config, void** args);

    // Queues a launch and returns at once. Nothing runs until a sync or a
    // synchronous launch drains the queue.
    //
    // args need not outlive the call: the kernel is built here, and a Program
    // carries its constants. What must outlive it is the device memory the
    // program addresses.
    //
    // Two launches on one stream run in order — the second's first block waits
    // for the first's last. Two on different streams do not, and that is what
    // buys anything: a grid too small to fill the machine leaves slots the other
    // stream's blocks take.
    void myrt_launch_async(KernelFunc kernel, const LaunchConfig& config, void** args,
                           StreamId stream = DEFAULT_STREAM);

    // The same, with the grid read from device memory when the launch reaches the
    // machine rather than when it is queued.
    //
    // The delay is the point. A kernel earlier in the same stream can write those
    // three floats, so a culling pass can decide how much work the pass after it
    // does without the host ever learning the number — the premise GPU-driven
    // rendering is built on.
    void myrt_launch_indirect(KernelFunc kernel, const IndirectLaunchConfig& config,
                              void** args, StreamId stream = DEFAULT_STREAM);

    // A fresh stream id. Ordered against nothing, including the default stream.
    StreamId myrt_stream_create();

    // What one stream's launches came to, since construction or the last
    // myrt_sync. Two halves, and they behave differently: the work counters
    // partition the total exactly, while cycles and stall_steps are what that
    // stream saw of the machine's time — two streams both see a cycle they
    // overlapped in, so those add up to more than the total. The surplus is the
    // overlap.
    const SchedulerStats& myrt_stream_stats(StreamId stream) const;

    // Runs whatever is queued and returns. Every counter stands afterwards,
    // which is what a caller wanting to read them does instead of syncing: a
    // queued launch has nowhere else to be run, and myrt_sync clears the
    // statistics it just finished gathering.
    void myrt_wait();

    // The machine the next launch runs on. Defaults to one SM holding one block,
    // which is what every figure in benchmarks/ was taken on.
    void myrt_set_sm_config(const SMConfig& config);
    void myrt_set_spec(const GPUSpec& spec);

    // What a benchmark prints at the head of its table. A figure whose machine is
    // not stated beside it cannot be reproduced.
    const GPUSpec& myrt_spec() const;

    // Runs whatever is queued, then reports the statistics gathered since the
    // last call and clears them, which is what makes it the natural place to end
    // a kernel run.
    // report=false still waits and still clears the counters, but prints
    // nothing. For a sync in the middle of a draw, which ends a pass rather
    // than a kernel run — and for a benchmark making fifteen of them.
    void myrt_sync(bool report = true);

    // --- statistics ---------------------------------------------------------
    // Accumulated across every launch since construction or the last
    // myrt_sync(). The Scheduler only ever knows about the launch it is
    // running, so the totals live here.
    double divergence_rate() const;

    // Simulated instructions per second, cost-weighted, in billions.
    //
    // This measures how fast the *simulator* retires work on the host, not how
    // fast the simulated GPU would be — that would need a cycle model rather
    // than a wall clock. It is still the number that moves when a kernel is
    // made to diverge less, which is what the benchmark compares.
    double throughput_giops() const;

    const SchedulerStats& stats() const
    {
        return stats_;
    }

    // How a warp picks which of its lanes to issue when they disagree. Defaults
    // to WarpPolicy::LowestPc, which is what every figure in benchmarks/ was
    // taken under — a caller changing it is measuring the policy itself.
    void myrt_set_warp_policy(WarpPolicy policy);

    // How a global access is charged. Defaults to MemoryModel::Flat, which is
    // what every figure in benchmarks/ was taken under.
    void myrt_set_memory_model(MemoryModel model);

    // Whether a result waits. Defaults to LatencyModel::Ignored, likewise.
    void myrt_set_latency_model(LatencyModel model);

    // Cache capacities in lines, for a caller that means to reach one. The
    // hardware sizes are far larger than any scene here fills, so a benchmark
    // wanting to show what happens at capacity has to scale them down and say so.
    void myrt_set_cache_lines(size_t l1, size_t l2);

    double elapsed_seconds() const
    {
        return elapsed_seconds_;
    }

private:
    // Held by pointer so the runtime stays movable even though MemoryManager
    // deliberately is not — its buffers back every pointer it has handed out.
    std::unique_ptr<MemoryManager> mem_;
    std::unique_ptr<WarpScheduler> scheduler_;

    SchedulerStats stats_;
    double elapsed_seconds_ = 0.0;

    // A launch that has been built but not yet run. The Program is held by value
    // because the caller's args are gone by now; the grid may still be unknown,
    // which is what indirect means.
    struct QueuedLaunch {
        Program program;
        dim3 grid;
        dim3 block;
        size_t shared_bytes = 0;
        uint32_t cluster_size = 1;
        StreamId stream = DEFAULT_STREAM;
        bool indirect = false;
        size_t grid_offset = 0;
    };

    std::vector<QueuedLaunch> queue_;
    StreamId next_stream_ = DEFAULT_STREAM;

    // Indexed by stream id, which is why the ids are dense. Cleared with the
    // totals, by myrt_sync.
    std::vector<SchedulerStats> stream_stats_{1};

    void enqueue(QueuedLaunch launch);

    // Everything queued, on the machine at once. One timing around the whole of
    // it: overlapping launches share cycles, and timing them separately would
    // count the same wall-clock second once a stream.
    void drain();

    // Seeds one block's threads with the coordinates a kernel reads.
    void seed_block(ThreadBlock& tb, const QueuedLaunch& launch, uint32_t block_id,
                    uint32_t warps) const;

    // Reads three floats from device memory. Throws if they are not whole,
    // non-negative numbers, or if the offset is not three floats inside the
    // device arena.
    dim3 read_grid(size_t offset) const;

    // "[STATS] divergence: X.X%, throughput: X.X GIOPS"
    void print_stats() const;
};
