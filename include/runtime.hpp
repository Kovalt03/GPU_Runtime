#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

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
};

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

    // The machine the next launch runs on. Defaults to one SM holding one block,
    // which is what every figure in benchmarks/ was taken on.
    void myrt_set_sm_config(const SMConfig& config);
    void myrt_set_spec(const GPUSpec& spec);

    // What a benchmark prints at the head of its table. A figure whose machine is
    // not stated beside it cannot be reproduced.
    const GPUSpec& myrt_spec() const;

    // Nothing to wait for in a synchronous simulator; this reports the
    // statistics gathered since the last call and clears them, which is what
    // makes it the natural place to end a kernel run.
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

    // "[STATS] divergence: X.X%, throughput: X.X GIOPS"
    void print_stats() const;
};
