#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>

#include "isa.hpp"
#include "thread.hpp"

// Device memory a kernel may address. Register values are byte offsets from
// base, matching what MemoryManager::device_alloc hands out.
struct DeviceSpan {
    uint8_t* base = nullptr;
    size_t size = 0;
};

// Two raw counters; the rest is derived so the numbers cannot drift apart.
// A step issues one instruction to a whole warp and costs WARP_SIZE lane slots
// whether one lane participates or all 32 — which is why divergence is costly.
struct SchedulerStats {
    uint64_t warp_steps = 0;       // instructions issued to a warp
    uint64_t active_lane_ops = 0;  // lane-instructions that actually ran

    // active_lane_ops weighted by instruction_cost, so that throughput readings
    // distinguish a kernel full of global loads from one full of adds.
    uint64_t weighted_lane_ops = 0;

    uint64_t lane_slots() const
    {
        return warp_steps * WARP_SIZE;
    }

    uint64_t masked_lane_slots() const
    {
        return lane_slots() - active_lane_ops;
    }

    // Wasted fraction of issued capacity, 0.0 to 1.0. Zero when nothing ran.
    double divergence_rate() const;

    // A Scheduler only ever knows about the run it is performing, so totals
    // spanning several launches are folded together by the caller.
    SchedulerStats& operator+=(const SchedulerStats& other)
    {
        warp_steps += other.warp_steps;
        active_lane_ops += other.active_lane_ops;
        weighted_lane_ops += other.weighted_lane_ops;
        return *this;
    }
};

// Executes a Program over the warps of one ThreadBlock.
//
// Divergence is modelled by min-PC reconvergence: a step issues the lowest pc
// among the warp's still-active threads, and lanes sitting elsewhere are masked
// off for that step. Threads that took different branches therefore cost extra
// steps, and rejoin on their own once they reach the same instruction again.
//
// The alternative, an explicit reconvergence stack, needs the compiler to mark
// where control flow rejoins. Min-PC derives that from the pcs themselves, which
// keeps the ISA free of a post-dominator annotation it has no way to compute.
class WarpScheduler {
public:
    // Runs until every thread has retired. Throws std::runtime_error on a bad
    // register index, an unaligned or out-of-range address, or a pc that leaves
    // the program. The block must outlive the call.
    void run(const Program& program, ThreadBlock& block, DeviceSpan global);

    const SchedulerStats& stats() const
    {
        return stats_;
    }

    double divergence_rate() const
    {
        return stats_.divergence_rate();
    }

    void reset_stats();

private:
    std::queue<Warp*> ready_queue_;  // pointers: a Warp is ~32 KB
    SchedulerStats stats_;

    // Returns false when the warp will not take another turn — retired, or now
    // waiting at a barrier. run() tells the two apart by Warp::at_barrier.
    bool step_warp(const Program& program, Warp& warp, ThreadBlock& block,
                   DeviceSpan global);

    // Every warp of the block has arrived, so let them all go.
    void release_barrier(ThreadBlock& block);

    // thread.pc has already been advanced past instr; branches overwrite it,
    // computing their target from instr_pc rather than from the advanced value.
    void execute(const Instruction& instr, uint32_t instr_pc, Thread& thread,
                 ThreadBlock& block, DeviceSpan global);
};

// Exposed for testing: each encodes a rule whose violation would corrupt memory
// rather than fail loudly.

// Turns a register value into a byte offset. Rejects negatives, NaN, and
// anything that is not a whole number.
size_t decode_address(float value, const char* what);

// Checks that reg .. reg + count - 1 all exist: a VEC3 at src0 = 254 does not.
void require_register_range(uint32_t reg, uint32_t count, const char* what);

// VEC4 and wider start at a register index that is a multiple of 4. The rule
// exists so that a future V_LD_GLOBAL_MAT4_F32 can map an aligned register
// block onto an aligned address; enforcing it on the ALU too keeps one
// convention rather than two.
void require_register_alignment(uint32_t reg, uint32_t alignment, const char* what);
