#pragma once

#include <array>
#include <cstdint>
#include <vector>

// SIMT execution state. These are plain data: all execution logic lives in the
// Scheduler (stage 4), which reads and mutates these structures.

// Fixed at 32 to match real NVIDIA hardware, and so that a warp's activity fits
// exactly into one uint32_t.
inline constexpr uint32_t WARP_SIZE = 32;

// PTX exposes at most 255 registers; 256 keeps the index a full uint8_t range,
// which is what Instruction's operand fields hold.
inline constexpr uint32_t REGS_PER_THREAD = 256;

// 4096 floats = 16 KB, the per-SM shared memory floor on the hardware modelled.
inline constexpr uint32_t SHARED_MEM_FLOATS = 4096;

struct Thread {
    std::array<float, REGS_PER_THREAD> regs{};  // register file, float-only
    uint32_t pc = 0;                            // index into a Program
    bool active = true;                         // false once RET has run
};

// ~33 KB (32 threads x ~1 KB of registers). Keep warps in a container, never
// as a local — a stack-allocated Warp risks overflow.
//
// Thread::pc is the source of truth. The two fields below are recomputed by the
// Scheduler on every step: pc is the instruction being issued, and active_mask
// says which lanes are sitting on it.
// How many warp masks S_BALLOT and friends have to work with.
//
// Four rather than one so a kernel can hold a ballot while taking another —
// with one, any use of a second ballot would have to spill, and there is
// nowhere to spill a mask to. Small because these are the warp's registers, not
// a lane's: real hardware counts its predicate registers in single digits too.
inline constexpr uint32_t WARP_MASK_REGISTERS = 4;

struct Warp {
    std::array<Thread, WARP_SIZE> threads{};
    uint32_t pc = 0;
    uint32_t active_mask = 0;

    // Where a ballot lands. Separate from the lane register file because a
    // 32-bit mask does not fit a float: the file is exact only to 2^24, so the
    // top eight lanes would be lost. This is the same reason hardware keeps
    // predicate registers apart from the vector ones.
    std::array<uint32_t, WARP_MASK_REGISTERS> masks{};

    // A waiting warp leaves the scheduler's queue entirely, which is what makes
    // the wait cost nothing but the turns it does not take.
    bool at_barrier = false;

    // The cycle this warp may next issue on, under LatencyModel::Modelled. Zero
    // throughout while latency is ignored, which is what makes that model's
    // scheduling identical to the one every measurement was taken under.
    uint64_t ready_at = 0;

    // Where WarpPolicy::Independent left off, so the next turn can go to a
    // different pc. Unused under LowestPc, which needs no memory of its own —
    // the lowest live pc is a property of the warp's state alone.
    //
    // On the warp rather than on the scheduler: warps take turns, so a single
    // cursor in the scheduler would be advanced by whichever warp ran last and
    // would tell this one nothing about its own lanes.
    uint32_t last_issued_pc = 0;
};

// Named ThreadBlock rather than Block because memory.hpp already defines a
// Block (an allocator span), and the Scheduler includes both headers. This is
// also CUDA's own term for the structure.
struct ThreadBlock {
    std::vector<Warp> warps;
    std::array<float, SHARED_MEM_FLOATS> shared_mem{};  // shared by all warps here
    uint32_t block_id = 0;
};

// --- activeMask helpers -----------------------------------------------------
// Precondition: lane < WARP_SIZE. Shifting a uint32_t by 32 or more is
// undefined behaviour, so these are not checked on what is a per-lane hot path.

inline bool is_active(const Warp& warp, uint32_t lane)
{
    return ((warp.active_mask >> lane) & 1u) != 0u;
}

inline void activate(Warp& warp, uint32_t lane)
{
    warp.active_mask |= (1u << lane);
}

inline void deactivate(Warp& warp, uint32_t lane)
{
    warp.active_mask &= ~(1u << lane);
}

// Lanes currently participating. A step costs a full warp slot however few
// lanes are on, so WARP_SIZE minus this is the work that went to waste — which
// is what the Scheduler's divergence statistics accumulate.
uint32_t active_lane_count(const Warp& warp);

// Mask with the low lane_count bits set. Not (1u << lane_count) - 1: that is
// undefined behaviour at lane_count == WARP_SIZE, which is the common case.
uint32_t lane_mask(uint32_t lane_count);

// --- construction -----------------------------------------------------------

// A warp with lane_count lanes enabled, every thread zeroed and at pc 0.
Warp make_warp(uint32_t lane_count = WARP_SIZE);

// A block of warp_count fully-enabled warps with shared memory zeroed.
ThreadBlock make_block(uint32_t warp_count, uint32_t block_id = 0);
