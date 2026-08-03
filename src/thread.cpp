#include <stdexcept>
#include <string>

#include "thread.hpp"

uint32_t active_lane_count(const Warp& warp)
{
    // mask & (mask - 1) clears the lowest set bit, so this costs one pass per
    // active lane. std::popcount would say it directly, but it is C++20.
    uint32_t count = 0;
    for (uint32_t mask = warp.active_mask; mask != 0u; mask &= mask - 1u) {
        ++count;
    }
    return count;
}

uint32_t lane_mask(uint32_t lane_count)
{
    // 1u << 32 is undefined behaviour, not zero — and a full warp is the
    // default for every warp created, so this is the common path.
    if (lane_count >= WARP_SIZE) {
        return ~0u;
    }
    return (1u << lane_count) - 1u;
}

Warp make_warp(uint32_t lane_count)
{
    // Clamping would hand back a warp whose mask disagrees with threads.size(),
    // which then under-reports divergence rather than failing.
    if (lane_count > WARP_SIZE) {
        throw std::runtime_error("make_warp: lane_count exceeds WARP_SIZE (" +
                                 std::to_string(lane_count) + ")");
    }

    // Default member initialisers already zero the threads, so only the mask is left.
    // Note that Thread::active and the mask answer different questions:
    // RET clears active for good, while the mask is recomputed every step.
    Warp warp;
    warp.active_mask = lane_mask(lane_count);
    return warp;
}

ThreadBlock make_block(uint32_t warp_count, uint32_t block_id)
{
    ThreadBlock block;
    block.block_id = block_id;

    // resize default-constructs in place. Building one warp and copying it would
    // move about 32 KB per warp for nothing.
    block.warps.resize(warp_count);
    for (Warp& warp : block.warps) {
        warp.active_mask = lane_mask(WARP_SIZE);
    }

    // shared_mem is zeroed by its default member initialiser.
    return block;
}
