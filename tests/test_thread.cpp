#include <stdexcept>

#include <gtest/gtest.h>

#include "thread.hpp"

// ---------------------------------------------------------------------------
// Structural contracts
// ---------------------------------------------------------------------------

TEST(Thread, WarpHoldsExactlyThirtyTwoLanes)
{
    // WARP_SIZE is what makes a uint32_t mask exactly cover a warp. If the two
    // ever disagree, every mask helper silently addresses the wrong lanes.
    EXPECT_EQ(WARP_SIZE, 32u);
    EXPECT_EQ(Warp{}.threads.size(), WARP_SIZE);
    EXPECT_EQ(sizeof(Warp{}.active_mask) * 8, WARP_SIZE);
}

TEST(Thread, RegisterIndexFitsInInstructionOperand)
{
    // Instruction stores register indices in uint8_t fields, so the register
    // file must not exceed what those can address.
    EXPECT_EQ(REGS_PER_THREAD, 256u);
    EXPECT_EQ(Thread{}.regs.size(), REGS_PER_THREAD);
}

TEST(Thread, DefaultThreadIsZeroedAndRunnable)
{
    const Thread t{};
    EXPECT_EQ(t.pc, 0u);
    EXPECT_TRUE(t.active);
    for (float r : t.regs) {
        ASSERT_EQ(r, 0.0f) << "registers must start zeroed";
    }
}

// ---------------------------------------------------------------------------
// lane_mask — the full-width case is the one that breaks
// ---------------------------------------------------------------------------

TEST(Thread, LaneMaskCoversFullWarp)
{
    // (1u << 32) - 1 is undefined behaviour, and WARP_SIZE is the default for
    // every warp created, so this is the common path rather than an edge case.
    EXPECT_EQ(lane_mask(WARP_SIZE), 0xFFFFFFFFu);
}

TEST(Thread, LaneMaskPartialWidths)
{
    EXPECT_EQ(lane_mask(0), 0x00000000u);
    EXPECT_EQ(lane_mask(1), 0x00000001u);
    EXPECT_EQ(lane_mask(4), 0x0000000Fu);
    EXPECT_EQ(lane_mask(16), 0x0000FFFFu);
    EXPECT_EQ(lane_mask(31), 0x7FFFFFFFu);
}

// ---------------------------------------------------------------------------
// activeMask helpers — lane 31 is where off-by-one shifting shows up
// ---------------------------------------------------------------------------

TEST(Thread, ActivateAndDeactivateSingleLanes)
{
    Warp w = make_warp(WARP_SIZE);
    ASSERT_EQ(w.active_mask, 0xFFFFFFFFu);

    deactivate(w, 0);
    EXPECT_FALSE(is_active(w, 0));
    EXPECT_TRUE(is_active(w, 1)) << "neighbouring lanes must be untouched";

    deactivate(w, 31);
    EXPECT_FALSE(is_active(w, 31));
    EXPECT_TRUE(is_active(w, 30));
    EXPECT_EQ(w.active_mask, 0x7FFFFFFEu);

    activate(w, 0);
    activate(w, 31);
    EXPECT_EQ(w.active_mask, 0xFFFFFFFFu);
}

TEST(Thread, ActivateIsIdempotent)
{
    Warp w = make_warp(0);
    EXPECT_EQ(w.active_mask, 0u);

    activate(w, 5);
    activate(w, 5);
    EXPECT_EQ(w.active_mask, 1u << 5);

    deactivate(w, 5);
    deactivate(w, 5);
    EXPECT_EQ(w.active_mask, 0u);
}

TEST(Thread, ActiveLaneCount)
{
    Warp w = make_warp(WARP_SIZE);
    EXPECT_EQ(active_lane_count(w), WARP_SIZE);

    deactivate(w, 3);
    deactivate(w, 17);
    EXPECT_EQ(active_lane_count(w), WARP_SIZE - 2);

    w.active_mask = 0;
    EXPECT_EQ(active_lane_count(w), 0u);

    // The all-ones mask must not be mistaken for a sign bit anywhere.
    w.active_mask = 0xFFFFFFFFu;
    EXPECT_EQ(active_lane_count(w), 32u);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(Thread, MakeWarpEnablesRequestedLanes)
{
    const Warp full = make_warp(WARP_SIZE);
    EXPECT_EQ(full.active_mask, 0xFFFFFFFFu);
    EXPECT_EQ(full.pc, 0u);
    EXPECT_EQ(active_lane_count(full), WARP_SIZE);

    const Warp partial = make_warp(8);
    EXPECT_EQ(partial.active_mask, 0x000000FFu);
    EXPECT_EQ(active_lane_count(partial), 8u);

    // A partially filled warp still owns all 32 thread slots; only the mask
    // says which of them run. This is what a tail warp looks like when the
    // launch size is not a multiple of 32.
    EXPECT_EQ(partial.threads.size(), WARP_SIZE);
}

TEST(Thread, MakeWarpDefaultsToFullWidth)
{
    EXPECT_EQ(make_warp().active_mask, make_warp(WARP_SIZE).active_mask);
}

TEST(Thread, MakeWarpRejectsOversizedLaneCount)
{
    // Clamping would produce a mask that disagrees with threads.size() and then
    // quietly under-report divergence, so the request is refused instead.
    EXPECT_THROW(make_warp(WARP_SIZE + 1), std::runtime_error);
    EXPECT_NO_THROW(make_warp(WARP_SIZE));
}

TEST(Thread, MakeBlockCreatesIndependentWarps)
{
    ThreadBlock block = make_block(4, 7);
    ASSERT_EQ(block.warps.size(), 4u);
    EXPECT_EQ(block.block_id, 7u);

    for (const Warp& w : block.warps) {
        EXPECT_EQ(w.active_mask, 0xFFFFFFFFu);
        EXPECT_EQ(w.pc, 0u);
    }

    // Warps must not alias: writing one lane's register must not disturb
    // another warp.
    block.warps[0].threads[0].regs[0] = 1.0f;
    EXPECT_EQ(block.warps[1].threads[0].regs[0], 0.0f);
}

TEST(Thread, MakeBlockZeroesSharedMemory)
{
    const ThreadBlock block = make_block(1, 0);
    EXPECT_EQ(block.shared_mem.size(), SHARED_MEM_FLOATS);
    for (float v : block.shared_mem) {
        ASSERT_EQ(v, 0.0f) << "shared memory must start zeroed";
    }
}
