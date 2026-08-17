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

    // Turns spent with no warp able to issue, every resident one waiting on a
    // result. The number occupancy exists to drive down, and zero while
    // instruction_latency returns zero.
    uint64_t stall_steps = 0;

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
        stall_steps += other.stall_steps;
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
// Which live pc a warp issues when its lanes disagree.
//
// Every lane already carries its own pc and there is no reconvergence stack, so
// the machine is Volta-shaped in its data. What it lacks is Volta's guarantee,
// and that lives entirely in this choice.
enum class WarpPolicy {
    // The lowest live pc, always. Lanes that ran ahead wait, so a warp
    // reconverges the moment its pcs coincide and it costs nothing to arrange.
    // The price is that a lane can be starved by one that is merely behind it —
    // LowestPcFirstStrandsALaneWaitingOnAHigherOne is that case.
    LowestPc,

    // Every live pc gets its turn, so no lane can be starved by another's
    // position in the program. This is what independent thread scheduling buys
    // and what it costs: reconvergence stops being automatic, and a kernel that
    // needs its lanes together has to say so.
    Independent,
};

// How a global access is charged. A warp issues one load and its 32 lanes name
// 32 addresses; what that costs depends on how many separate pieces of memory
// they land in, which is the first question asked of any real kernel.
enum class MemoryModel {
    // Every lane pays instruction_cost whatever it asked for, so a warp reading
    // one address costs what a warp reading 32 scattered ones does.
    Flat,

    // Charged by the distinct cache lines the warp touches.
    //
    // Not a refinement of Flat but a different answer: the raster kernels read
    // their triangle warp-uniformly, which is one line rather than 32 lanes, and
    // that alone takes about 93% off every route. It also reverses shared-memory
    // staging, which existed to stop 32 lanes issuing the same load and turns
    // out to have been saving something that cost one line anyway.
    Coalesced,
};

// Bytes fetched together. 128 is what NVIDIA moves for a warp-wide access, and
// at four bytes a float that is a warp's 32 lanes exactly.
inline constexpr uint32_t CACHE_LINE_BYTES = 128;

class WarpScheduler {
public:
    // How many warp steps one block may take before run() gives up. Issuing the
    // lowest live pc first means a lane can wait on one that will never be
    // reached, so not every block that fails to retire is a mistaken kernel —
    // see LowestPcFirstStrandsALaneWaitingOnAHigherOne.
    //
    // The heaviest kernel here takes 706 steps for a block at 256x256, so this
    // leaves four orders of magnitude before anything that works could meet it.
    static constexpr uint64_t DEFAULT_STEP_BUDGET = 1ull << 24;

    // Lowered by tests that mean to reach the budget: at the default, doing so
    // takes tens of seconds.
    void set_step_budget(uint64_t steps)
    {
        step_budget_ = steps;
    }

    // Defaults to LowestPc, which is what every measurement in benchmarks/ was
    // taken under.
    void set_policy(WarpPolicy policy)
    {
        policy_ = policy;
    }

    // Defaults to Flat, likewise.
    void set_memory_model(MemoryModel model)
    {
        memory_ = model;
    }

    // Runs until every thread has retired. Throws std::runtime_error on a bad
    // register index, an unaligned or out-of-range address, a pc that leaves
    // the program, or the step budget elapsing with the block unfinished. The
    // block must outlive the call.
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
    uint64_t step_budget_ = DEFAULT_STEP_BUDGET;
    WarpPolicy policy_ = WarpPolicy::LowestPc;
    MemoryModel memory_ = MemoryModel::Flat;

    // Separate from instruction_cost because the answer is a property of the 32
    // addresses rather than of the opcode.
    uint64_t global_access_cost(const Warp& warp, const Instruction& instr) const;

    // Which pc to issue under Independent. Mutable because fairness needs
    // somewhere to remember what it last did — Warp::last_issued_pc.
    uint32_t select_independent_pc(Warp& warp) const;

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
