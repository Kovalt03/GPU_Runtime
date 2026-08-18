#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <queue>
#include <unordered_map>
#include <vector>

#include "gpu_spec.hpp"
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
    // result. The number occupancy exists to drive down, and zero under
    // LatencyModel::Ignored.
    uint64_t stall_steps = 0;

    // Elapsed time, as against instructions issued — equal while nothing waits.
    // divergence_rate stays on warp_steps once they part: stalling and divergence
    // are different waste, and one rate for both would say which to fix about
    // neither.
    uint64_t cycles = 0;

    // Where the lines a warp asked for were found. Kept apart because the levels
    // cost different amounts, and because an L2 hit is what says L1 evicted.
    uint64_t l1_hits = 0;
    uint64_t l2_hits = 0;
    uint64_t cache_misses = 0;

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

    // What happened between two readings. Used to charge a step's work to the
    // launch that issued it, where taking the difference of the whole struct is
    // what keeps a counter added later from being left out of the attribution.
    SchedulerStats& operator-=(const SchedulerStats& other)
    {
        warp_steps -= other.warp_steps;
        active_lane_ops -= other.active_lane_ops;
        weighted_lane_ops -= other.weighted_lane_ops;
        stall_steps -= other.stall_steps;
        cycles -= other.cycles;
        l1_hits -= other.l1_hits;
        l2_hits -= other.l2_hits;
        cache_misses -= other.cache_misses;
        return *this;
    }

    // A Scheduler only ever knows about the run it is performing, so totals
    // spanning several launches are folded together by the caller.
    SchedulerStats& operator+=(const SchedulerStats& other)
    {
        warp_steps += other.warp_steps;
        active_lane_ops += other.active_lane_ops;
        weighted_lane_ops += other.weighted_lane_ops;
        stall_steps += other.stall_steps;
        cycles += other.cycles;
        l1_hits += other.l1_hits;
        l2_hits += other.l2_hits;
        cache_misses += other.cache_misses;
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

    // As Coalesced, and then asking whether each line had to be fetched at all.
    // Counting transactions says nothing about repetition: every warp of the
    // naive walk re-reads the same triangle, and Coalesced charges each of them
    // full price for a line the one before it just touched.
    Cached,
};

// Whether a result is available at once or after a delay. Independent of
// MemoryModel, which decides what an access costs in issue capacity rather than
// how long the warp waits to use the answer.
enum class LatencyModel {
    // Ready the instant it is issued, so no warp waits and resident warps neither
    // help nor hinder each other.
    Ignored,

    // instruction_latency decides when a warp may issue again, so warps cover one
    // another's waiting — the reason they are batched, and what occupancy
    // measures.
    Modelled,
};

// A set of resident cache lines with least-recently-used eviction.
//
// Tags only, no data: a load reads device memory either way, so copies would hand
// back the same numbers. What a cache changes is the price, and for that it need
// only know what is resident. Nor is there dirty state — nothing reads through
// the cache, so no write can be seen before it reaches memory.
//
// A map into a list rather than a scan, because L2 holds tens of thousands of
// lines and a warp-heavy kernel looks up millions of times.
class LineCache {
public:
    explicit LineCache(size_t capacity) : capacity_(capacity) {}

    // Marks the line as most recently used, and says whether it was already
    // resident. Lookup and install in one call, because a caller that did them
    // separately could do them in the wrong order.
    bool touch(size_t line);

    // Drops a line if it is resident. What a host upload does to the copies:
    // holding tags rather than data, the cache cannot tell that the bytes under
    // a line have been replaced, and would report a hit on something nobody had
    // read yet.
    void invalidate(size_t line);

    // Emptied when a block starts, for a cache that belongs to one block.
    void clear();

private:
    size_t capacity_;
    std::list<size_t> order_;  // least recently used first
    std::unordered_map<size_t, std::list<size_t>::iterator> resident_;
};

// What a warp's access to global memory came to.
//
// Cost adds across the lines, capacity being spent on each. Latency takes the
// worst of them, the lines being fetched together — a warp waits for the slowest,
// not for the sum.
struct GlobalAccess {
    uint64_t cost = 0;
    uint32_t latency = 0;
};

// A source of blocks for the SMs to pull from.
//
// The runtime knows how to build block i — seeding its coordinates is its job —
// and the scheduler knows when a slot is free. Handing over a callback rather
// than a vector is what keeps a 256x256 launch from materialising 2,048 blocks
// at 32 KB a warp before the first one runs.
//
// Returns false when the grid is exhausted.
using NextBlock = std::function<bool(ThreadBlock&)>;

// One grid waiting for a machine, and which queue it waits in.
//
// The program is held by pointer rather than by value because a launch is
// queued, not copied: the caller keeps the Program it built and this names it.
// It must outlive run_streams.
struct GridLaunch {
    const Program* program = nullptr;
    NextBlock next_block;

    // What the kernel declares it will use of a block's scratchpad. Per launch
    // rather than per machine, since two kernels sharing an SM need not agree
    // about it.
    size_t shared_bytes = 0;

    // Launches carrying the same id run in order: none of this one's blocks is
    // placed until every block of the one before it has retired. Different ids
    // are unordered with respect to each other, which is the whole of what a
    // stream is — the machine is free to interleave them, and does.
    uint32_t stream = 0;
};

class WarpScheduler {
public:
    // How many cycles a *launch* may take before run_grid gives up.
    //
    // Cycles rather than issued instructions, because the two part under
    // LatencyModel::Modelled and a warp can then burn time without issuing
    // anything — a bound on issues alone would never be reached by one that only
    // ever waits.
    //
    // Not every launch that fails to retire is a mistaken kernel: issuing the
    // lowest live pc first means a lane can wait on one that will never be
    // reached. See LowestPcFirstStrandsALaneWaitingOnAHigherOne.
    //
    // It used to bound one block, and blocks ran one at a time. Now it bounds the
    // launch's wall clock, which on one SM is every block's time added up — so a
    // grid of sixty-four blocks meets a per-block bound sixty-four times sooner
    // than it should. Raised by four bits to cover that, which still leaves the
    // heaviest thing measured here (about 130 million cycles) two bits of room.
    static constexpr uint64_t DEFAULT_CYCLE_BUDGET = 1ull << 28;

    // Lowered by tests that mean to reach the budget: at the default, doing so
    // takes tens of seconds.
    void set_cycle_budget(uint64_t cycles)
    {
        cycle_budget_ = cycles;
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

    // Defaults to Ignored, likewise.
    void set_latency_model(LatencyModel model)
    {
        latency_ = model;
    }

    // Shrunk by tests that mean to reach a capacity: filling L2 for real would
    // take about 175,000 triangles, so its eviction is otherwise a path no scene
    // reaches. Both caches are emptied, a capacity change making what is resident
    // meaningless.
    void set_cache_lines(size_t l1, size_t l2)
    {
        spec_.l1_lines = l1;
        spec_.l2_lines = l2;
        l1s_.clear();
        l2_ = LineCache(l2);
    }

    // Forgets every line the byte range touches, at both levels.
    //
    // A host upload writes device memory without going through a warp, so the
    // caches never see it. Holding tags and no data, they would go on reporting
    // hits for lines whose contents had been replaced — and a buffer released
    // and reallocated lands at the same address, which is exactly when that
    // matters. Hardware has the same problem and answers it the same way.
    void invalidate_range(size_t offset, size_t bytes);

    // Runs until every thread has retired. Throws std::runtime_error on a bad
    // register index, an unaligned or out-of-range address, a pc that leaves
    // the program, or the cycle budget elapsing with the block unfinished. The
    // block must outlive the call.
    void run(const Program& program, ThreadBlock& block, DeviceSpan global);

    // A whole grid, with the SMs pulling blocks as their slots come free.
    //
    // The cycle count this leaves is the launch's wall clock rather than the sum
    // of its blocks': with several SMs the blocks overlap, and adding their
    // clocks would count the same cycle once per SM. At the default one SM
    // holding one block the two are the same number, which is what keeps every
    // earlier figure standing.
    //
    // shared_bytes is what the kernel declares it will use of a block's
    // scratchpad, and one of the three things residency is the smallest of. It
    // is not a bound on what the kernel may address — ThreadBlock carries the
    // whole scratchpad either way — but a statement about how many blocks can
    // sit on an SM at once, which is what it decides on hardware.
    void run_grid(const Program& program, DeviceSpan global, size_t shared_bytes,
                  const NextBlock& next_block);

    // Several grids at once, ordered within a stream and free across streams.
    //
    // The blocks of concurrent launches compete for the same slots, warp slots
    // and shared memory, which is where a stream earns anything: a grid too
    // small to fill the machine leaves room another kernel's blocks can take.
    //
    // per_launch, when given, is filled with one entry a launch, in the order
    // they were handed in. What it holds divides in two:
    //
    //   work   warp_steps, the lane counts and the cache counts. Charged to
    //          whichever launch the warp belonged to, so they partition the
    //          totals exactly — the parts add up to stats().
    //   time   cycles and stall_steps. Charged to every launch resident while
    //          they passed, so two overlapping launches are both charged for
    //          the same cycle and the parts add up to *more* than the whole.
    //          The surplus is the overlap, and measuring it is the point.
    void run_streams(const std::vector<GridLaunch>& launches, DeviceSpan global,
                     std::vector<SchedulerStats>* per_launch = nullptr);

    // The machine this scheduler runs on. Defaults to the one every figure in
    // benchmarks/ was taken on: one SM holding one block.
    void set_sm_config(const SMConfig& config)
    {
        spec_.sms = config;
    }

    void set_spec(const GPUSpec& spec)
    {
        spec_ = spec;
        l1s_.clear();
        l2_ = LineCache(spec_.l2_lines);
    }

    const GPUSpec& spec() const
    {
        return spec_;
    }

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
    // Where run() leaves the block it was handed, so that it can be given back:
    // the residency loop owns its blocks, and run()'s caller owns theirs.
    std::unique_ptr<ThreadBlock> last_block_;
    SchedulerStats stats_;
    uint64_t cycle_budget_ = DEFAULT_CYCLE_BUDGET;
    WarpPolicy policy_ = WarpPolicy::LowestPc;
    MemoryModel memory_ = MemoryModel::Flat;
    LatencyModel latency_ = LatencyModel::Ignored;

    GPUSpec spec_;

    // L1 belongs to an SM, so there is one a piece and a launch empties them all.
    // Blocks that share an SM therefore share its L1 — which is the point, and
    // the reason this stopped being cleared between blocks.
    //
    // L2 belongs to the device and outlives a launch, which it does by living
    // here: the runtime holds one scheduler for its lifetime.
    std::vector<LineCache> l1s_;
    LineCache* active_l1_ = nullptr;  // the SM currently issuing
    LineCache l2_{L2_LINES};

    // The latency of whatever step_warp last issued, so that run_grid need not
    // decide it a second time — and so that a global load's answer, which comes
    // from the memory model rather than from instruction_latency, reaches it.
    uint32_t issued_latency_ = 0;

    // Separate from instruction_cost and instruction_latency because the answer is
    // a property of the 32 addresses rather than of the opcode.
    GlobalAccess global_access(const Warp& warp, const Instruction& instr);

    // What one line came to, given where it was found. Mutable because a lookup
    // changes what is resident.
    GlobalAccess cache_lookup(size_t line, const Instruction& instr);

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
