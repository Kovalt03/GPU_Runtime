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

    // An upload replaces bytes the caches hold tags for, and they cannot see it
    // happen: nothing went through a warp. Told nothing, they would report a hit
    // on a line whose contents nobody had read — and since a released buffer is
    // reallocated at the same address, that is the ordinary case rather than a
    // corner of one. A read back changes no device byte and needs no such call.
    if (dir == Direction::HostToDevice) {
        scheduler_->invalidate_range(mem_->device_offset(dst), size);
    }
}

size_t MyGPURuntime::myrt_device_offset(const void* ptr) const
{
    return mem_->device_offset(ptr);
}

size_t MyGPURuntime::myrt_device_free_bytes() const
{
    return mem_->device_free_bytes();
}

void MyGPURuntime::myrt_launch(KernelFunc kernel, dim3 grid, dim3 block, void** args)
{
    myrt_launch(kernel, LaunchConfig{grid, block}, args);
}

void MyGPURuntime::myrt_set_sm_config(const SMConfig& config)
{
    scheduler_->set_sm_config(config);
}

void MyGPURuntime::myrt_set_spec(const GPUSpec& spec)
{
    scheduler_->set_spec(spec);
}

const GPUSpec& MyGPURuntime::myrt_spec() const
{
    return scheduler_->spec();
}

void MyGPURuntime::seed_block(ThreadBlock& tb, const QueuedLaunch& launch,
                              uint32_t block_id, uint32_t warps) const
{
    const dim3 grid = launch.grid;
    const dim3 block = launch.block;
    const uint32_t threads_per_block = block.volume();

    const uint32_t bx = block_id % grid.x;
    const uint32_t by = (block_id / grid.x) % grid.y;
    const uint32_t bz = block_id / (grid.x * grid.y);

    tb = make_block(warps, block_id);
    for (uint32_t t = 0; t < warps * WARP_SIZE; t++) {
        Thread& th = tb.warps[t / WARP_SIZE].threads[t % WARP_SIZE];
        // Rounding the warp count up leaves a tail past the end of the launch.
        // Retiring those lanes is what stops them from running with coordinate 0
        // and rewriting thread 0's output.
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

        // Same for every thread of the block, and not recoverable from the
        // coordinates above without an integer divide.
        th.regs[REG_BLOCK_ID_X] = float(bx);
        th.regs[REG_BLOCK_ID_Y] = float(by);
        th.regs[REG_BLOCK_ID_Z] = float(bz);

        // Which block of its cluster this is. Not recoverable from the block id
        // without an integer division, and a kernel reading a neighbour's shared
        // memory has to know which neighbour it is itself.
        th.regs[REG_CLUSTER_RANK] =
            float(launch.cluster_size <= 1 ? 0 : block_id % launch.cluster_size);
    }
}

dim3 MyGPURuntime::read_grid(size_t offset) const
{
    if (offset % sizeof(float) != 0 || offset + 3 * sizeof(float) > mem_->device_size()) {
        throw std::runtime_error(
            "myrt_launch_indirect: the grid must be three aligned floats inside "
            "device memory");
    }
    const float* p = reinterpret_cast<const float*>(mem_->device_base() + offset);

    // The same decoder the ISA's addresses go through, so that a grid written by
    // a kernel is read under the rule the kernel wrote it under. A dimension is
    // one number and cannot be half of one.
    dim3 grid;
    grid.x = static_cast<uint32_t>(decode_address(p[0], "indirect grid x"));
    grid.y = static_cast<uint32_t>(decode_address(p[1], "indirect grid y"));
    grid.z = static_cast<uint32_t>(decode_address(p[2], "indirect grid z"));
    return grid;
}

void MyGPURuntime::myrt_launch(KernelFunc kernel, const LaunchConfig& config, void** args)
{
    // Queued and drained at once, which is what makes it synchronous. Anything
    // already waiting goes with it: a launch that returns having run means the
    // machine is idle behind it, and leaving another stream's blocks queued would
    // make that untrue.
    myrt_launch_async(std::move(kernel), config, args, DEFAULT_STREAM);
    drain();
}

void MyGPURuntime::myrt_launch_async(KernelFunc kernel, const LaunchConfig& config,
                                     void** args, StreamId stream)
{
    if (config.grid.volume() == 0 || config.block.volume() == 0) {
        throw std::runtime_error("myrt_launch: grid and block must both be non-empty");
    }
    QueuedLaunch launch;
    launch.grid = config.grid;
    launch.block = config.block;
    launch.shared_bytes = config.shared_bytes;
    launch.cluster_size = config.cluster_size;
    launch.stream = stream;

    // Called once for the whole launch, not once per thread: every thread runs
    // these same instructions, and only their registers differ. Called here
    // rather than at the drain so that args may die with the calling scope.
    launch.program = kernel(args);
    enqueue(std::move(launch));
}

void MyGPURuntime::myrt_launch_indirect(KernelFunc kernel,
                                        const IndirectLaunchConfig& config, void** args,
                                        StreamId stream)
{
    if (config.block.volume() == 0) {
        throw std::runtime_error("myrt_launch_indirect: block must be non-empty");
    }
    QueuedLaunch launch;
    launch.block = config.block;
    launch.shared_bytes = config.shared_bytes;
    launch.stream = stream;
    launch.indirect = true;
    launch.grid_offset = config.grid_offset;
    launch.program = kernel(args);
    enqueue(std::move(launch));
}

void MyGPURuntime::enqueue(QueuedLaunch launch)
{
    if (launch.stream >= stream_stats_.size()) {
        throw std::runtime_error("myrt_launch: no such stream");
    }
    queue_.push_back(std::move(launch));
}

void MyGPURuntime::myrt_wait()
{
    drain();
}

StreamId MyGPURuntime::myrt_stream_create()
{
    stream_stats_.emplace_back();
    return ++next_stream_;
}

const SchedulerStats& MyGPURuntime::myrt_stream_stats(StreamId stream) const
{
    if (stream >= stream_stats_.size()) {
        throw std::runtime_error("myrt_stream_stats: no such stream");
    }
    return stream_stats_[stream];
}

void MyGPURuntime::drain()
{
    if (queue_.empty()) {
        return;
    }

    // Where each launch has got to. Held apart from the queue because a builder
    // is called by the scheduler, whenever a slot comes free, and has to remember
    // between calls what it has already handed out.
    struct Cursor {
        uint32_t next = 0;
        uint32_t blocks = 0;
        uint32_t warps = 0;
        bool resolved = false;
    };
    std::vector<Cursor> cursors(queue_.size());

    std::vector<GridLaunch> launches(queue_.size());
    for (size_t i = 0; i < queue_.size(); ++i) {
        launches[i].program = &queue_[i].program;
        launches[i].shared_bytes = queue_[i].shared_bytes;
        launches[i].stream = queue_[i].stream;
        launches[i].cluster_size = queue_[i].cluster_size;

        // Built as the SMs ask for them rather than all at once. A warp is 32 KB,
        // so a 256x256 launch would otherwise materialise tens of megabytes of
        // blocks before the first instruction issued.
        launches[i].next_block = [this, i, &cursors](ThreadBlock& tb) {
            QueuedLaunch& launch = queue_[i];
            Cursor& cursor = cursors[i];
            if (!cursor.resolved) {
                // First asked for when the launch reaches the machine, which for
                // an indirect one is after whatever wrote its grid has retired.
                if (launch.indirect) {
                    launch.grid = read_grid(launch.grid_offset);
                }
                cursor.blocks = launch.grid.volume();
                cursor.warps = (launch.block.volume() + WARP_SIZE - 1) / WARP_SIZE;
                cursor.resolved = true;
            }
            if (cursor.next >= cursor.blocks) {
                return false;
            }
            seed_block(tb, launch, cursor.next++, cursor.warps);
            return true;
        };
    }

    // Once for the drain, where it used to be once a block. Blocks overlap when
    // there is more than one SM, so their clocks cannot be added — and the
    // counters that can be added are added here, in one place.
    scheduler_->reset_stats();

    // Attribution costs a snapshot a warp step, so it is asked for only when
    // there is something to attribute. One launch owns everything the drain did.
    std::vector<SchedulerStats> per_launch;
    const bool split = queue_.size() > 1;

    // steady_clock, not high_resolution_clock, which is permitted to be
    // non-monotonic and could yield a negative interval.
    const auto t0 = std::chrono::steady_clock::now();
    scheduler_->run_streams(launches,
                            DeviceSpan{mem_->device_base(), mem_->device_size()},
                            split ? &per_launch : nullptr);
    const auto t1 = std::chrono::steady_clock::now();

    elapsed_seconds_ += std::chrono::duration<double>(t1 - t0).count();
    stats_ += scheduler_->stats();
    if (split) {
        for (size_t i = 0; i < queue_.size(); ++i) {
            stream_stats_[queue_[i].stream] += per_launch[i];
        }
    } else {
        stream_stats_[queue_[0].stream] += scheduler_->stats();
    }

    queue_.clear();
}

void MyGPURuntime::myrt_sync(bool report)
{
    // Everything queued runs here, which is the only thing a sync waits for.
    // Reporting and clearing is what makes this the natural end of a kernel run.
    // Both fields reset together, or the next launch would divide its work by
    // carried-over time.
    //
    // Counters only. The scheduler's caches survive, as L2 does across kernel
    // launches on hardware — draw_walk and its neighbours call this between their
    // two passes, and clearing here would stop pass 2 from finding anything pass 1
    // had read.
    drain();
    if (report) {
        print_stats();
    }
    stats_ = SchedulerStats{};
    elapsed_seconds_ = 0.0;
    stream_stats_.assign(stream_stats_.size(), SchedulerStats{});
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

void MyGPURuntime::myrt_set_bandwidth_model(BandwidthModel model)
{
    scheduler_->set_bandwidth_model(model);
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
