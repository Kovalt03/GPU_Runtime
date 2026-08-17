#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

// thread.hpp for ThreadBlock/Thread/WARP_SIZE/make_block, which runtime.hpp
// itself does not need.
#include "runtime.hpp"
#include "thread.hpp"

MyGPURuntime::MyGPURuntime(size_t device_mem_size, size_t host_mem_size)
    : mem_(std::make_unique<MemoryManager>(host_mem_size, device_mem_size)),
      scheduler_(std::make_unique<WarpScheduler>())
{
}

// Forwarding, so that a kernel author sees one object rather than two. Every
// bounds check and both address spaces stay in MemoryManager; repeating any of
// it here would give two places to disagree.
//
// Device memory, as with cudaMalloc — host allocations are not exposed, since
// any ordinary address works as the host side of a transfer.
void* MyGPURuntime::myrt_malloc(size_t size)
{
    return mem_->device_alloc(size);
}

void MyGPURuntime::myrt_free(void* ptr)
{
    mem_->device_free(ptr);
}

void MyGPURuntime::myrt_memcpy(void* dst, const void* src, size_t size, Direction dir)
{
    mem_->memcpy(dst, src, size, dir);
}

size_t MyGPURuntime::myrt_device_offset(const void* ptr) const
{
    return mem_->device_offset(ptr);
}

void MyGPURuntime::myrt_launch(KernelFunc kernel, dim3 grid, dim3 block, void** args)
{
    if (grid.volume() == 0 || block.volume() == 0) {
        throw std::runtime_error("myrt_launch: grid and block must both be non-empty");
    }

    // Called once for the whole launch, not once per thread: every thread runs
    // these same instructions, and only their registers differ.
    const Program program = kernel(args);

    const uint32_t threads_per_block = block.volume();
    const uint32_t warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;

    for (uint32_t bz = 0; bz < grid.z; bz++) {
        for (uint32_t by = 0; by < grid.y; by++) {
            for (uint32_t bx = 0; bx < grid.x; bx++) {
                // Blocks are independent — own warps, own shared memory — so
                // each is built fresh. The id is the flat index, x fastest.
                const uint32_t block_id = (bz * grid.y + by) * grid.x + bx;
                ThreadBlock tb = make_block(warps, block_id);

                for (uint32_t t = 0; t < warps * WARP_SIZE; t++) {
                    Thread& th = tb.warps[t / WARP_SIZE].threads[t % WARP_SIZE];
                    // Rounding the warp count up leaves a tail past the end of
                    // the launch. Retiring those lanes is what stops them from
                    // running with coordinate 0 and rewriting thread 0's output.
                    if (t >= threads_per_block) {
                        th.active = false;
                        continue;
                    }
                    const uint32_t tx = t % block.x;
                    const uint32_t ty = (t / block.x) % block.y;
                    const uint32_t tz = t / (block.x * block.y);

                    th.regs[REG_GLOBAL_ID_X] = float(bx * block.x + tx);
                    th.regs[REG_GLOBAL_ID_Y] = float(by * block.y + ty);
                    th.regs[REG_GLOBAL_ID_Z] = float(bz * block.z + tz);

                    // Same for every thread of the block, and not recoverable
                    // from the coordinates above without an integer divide.
                    th.regs[REG_BLOCK_ID_X] = float(bx);
                    th.regs[REG_BLOCK_ID_Y] = float(by);
                    th.regs[REG_BLOCK_ID_Z] = float(bz);
                }
                // The Scheduler accumulates, so clear it between blocks and fold
                // the totals in here: statistics span every launch since the last
                // myrt_sync().
                scheduler_->reset_stats();

                // steady_clock, not high_resolution_clock, which is permitted to
                // be non-monotonic and could yield a negative interval.
                const auto t0 = std::chrono::steady_clock::now();
                scheduler_->run(program, tb,
                                DeviceSpan{mem_->device_base(), mem_->device_size()});
                const auto t1 = std::chrono::steady_clock::now();

                // Both accumulate. Assigning the elapsed time instead would
                // leave throughput dividing every block's work by one block's
                // time.
                elapsed_seconds_ += std::chrono::duration<double>(t1 - t0).count();
                stats_ += scheduler_->stats();
            }
        }
    }
}

void MyGPURuntime::myrt_sync(bool report)
{
    // Nothing to wait for: execution is synchronous. Reporting and clearing is
    // what makes this the natural end of a kernel run. Both fields reset
    // together, or the next launch would divide its work by carried-over time.
    //
    // Counters only. The scheduler's caches survive, as L2 does across kernel
    // launches on hardware — draw_walk and its neighbours call this between their
    // two passes, and clearing here would stop pass 2 from finding anything pass 1
    // had read.
    if (report) {
        print_stats();
    }
    stats_ = SchedulerStats{};
    elapsed_seconds_ = 0.0;
}

// Totals spanning several launches answer the same question a single run's do,
// so the arithmetic stays in SchedulerStats.
void MyGPURuntime::myrt_set_warp_policy(WarpPolicy policy)
{
    scheduler_->set_policy(policy);
}

void MyGPURuntime::myrt_set_memory_model(MemoryModel model)
{
    scheduler_->set_memory_model(model);
}

void MyGPURuntime::myrt_set_latency_model(LatencyModel model)
{
    scheduler_->set_latency_model(model);
}

void MyGPURuntime::myrt_set_cache_lines(size_t l1, size_t l2)
{
    scheduler_->set_cache_lines(l1, l2);
}

double MyGPURuntime::divergence_rate() const
{
    return stats_.divergence_rate();
}

double MyGPURuntime::throughput_giops() const
{
    // A kernel can finish inside the clock's resolution, leaving elapsed at
    // zero. Report nothing rather than infinity.
    if (elapsed_seconds_ <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(stats_.weighted_lane_ops) / elapsed_seconds_ / 1e9;
}

void MyGPURuntime::print_stats() const
{
    std::printf("[STATS] divergence: %.1f%%, throughput: %.2f GIOPS\n",
                100.0 * divergence_rate(), throughput_giops());
}
