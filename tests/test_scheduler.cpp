#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "scheduler.hpp"

namespace {

constexpr size_t GLOBAL_BYTES = 1024;

// A one-warp block plus a scratch device buffer, which is what most of these
// tests need. The buffer is a member so its address stays valid for the span.
struct Fixture {
    ThreadBlock block = make_block(1, 0);
    std::vector<uint8_t> global = std::vector<uint8_t>(GLOBAL_BYTES, 0);
    WarpScheduler sched;

    DeviceSpan span()
    {
        return DeviceSpan{global.data(), global.size()};
    }

    Warp& warp()
    {
        return block.warps[0];
    }

    Thread& lane(uint32_t i)
    {
        return block.warps[0].threads[i];
    }

    void run(const Program& prog)
    {
        sched.run(prog, block, span());
    }

    // Writes a float into the device buffer at a byte offset, standing in for
    // a MemoryManager::memcpy the kernel would normally be handed.
    void poke(size_t byte_offset, float value)
    {
        std::memcpy(global.data() + byte_offset, &value, sizeof(float));
    }

    float peek(size_t byte_offset) const
    {
        float out = 0.0f;
        std::memcpy(&out, global.data() + byte_offset, sizeof(float));
        return out;
    }
};

// Sets one register to the same value in every lane.
void broadcast(Warp& warp, uint8_t reg, float value)
{
    for (Thread& t : warp.threads) {
        t.regs[reg] = value;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Address and register validation
// ---------------------------------------------------------------------------

TEST(Scheduler, DecodeAddressAcceptsWholeOffsets)
{
    EXPECT_EQ(decode_address(0.0f, "t"), 0u);
    EXPECT_EQ(decode_address(4.0f, "t"), 4u);
    EXPECT_EQ(decode_address(1024.0f, "t"), 1024u);
}

TEST(Scheduler, DecodeAddressRejectsNonsense)
{
    // A truncated or wrapped address corrupts memory silently, so none of
    // these may be quietly rounded into something plausible.
    EXPECT_THROW(decode_address(-1.0f, "t"), std::runtime_error);
    EXPECT_THROW(decode_address(4.5f, "t"), std::runtime_error);
    EXPECT_THROW(decode_address(std::nanf(""), "t"), std::runtime_error);
    EXPECT_THROW(decode_address(INFINITY, "t"), std::runtime_error);
}

TEST(Scheduler, RegisterRangeRejectsVec3AtTheEnd)
{
    EXPECT_NO_THROW(require_register_range(0, 3, "t"));
    EXPECT_NO_THROW(require_register_range(253, 3, "t"));
    EXPECT_NO_THROW(require_register_range(255, 1, "t"));

    // A VEC3 starting at 254 would read reg 256, one past the file.
    EXPECT_THROW(require_register_range(254, 3, "t"), std::runtime_error);
    EXPECT_THROW(require_register_range(255, 3, "t"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------

TEST(Scheduler, ScalarArithmetic)
{
    Fixture f;
    broadcast(f.warp(), 1, 3.0f);
    broadcast(f.warp(), 2, 4.0f);

    f.run(Program{
        make_v_mul_f32(10, 1, 2),  // 12
        make_v_add_f32(11, 1, 2),  // 7
        make_v_sub_f32(12, 2, 1),  // 1
        make_v_min_f32(13, 1, 2),  // 3
        make_v_max_f32(14, 1, 2),  // 4
        make_v_sqrt_f32(15, 2),    // 2
        make_ret(),
    });

    const Thread& t = f.lane(0);
    EXPECT_FLOAT_EQ(t.regs[10], 12.0f);
    EXPECT_FLOAT_EQ(t.regs[11], 7.0f);
    EXPECT_FLOAT_EQ(t.regs[12], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[13], 3.0f);
    EXPECT_FLOAT_EQ(t.regs[14], 4.0f);
    EXPECT_FLOAT_EQ(t.regs[15], 2.0f);
}

TEST(Scheduler, ReciprocalAndFmaAccumulate)
{
    Fixture f;
    broadcast(f.warp(), 1, 4.0f);
    broadcast(f.warp(), 2, 5.0f);
    broadcast(f.warp(), 20, 1.0f);  // FMA accumulates into dst

    f.run(Program{
        make_v_rcp_f32(10, 1),     // 0.25
        make_v_fma_f32(20, 1, 2),  // 1 + 4*5 = 21
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 0.25f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[20], 21.0f);
}

TEST(Scheduler, EveryLaneComputesItsOwnData)
{
    // The point of SIMT: one instruction, 32 different results.
    Fixture f;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        f.lane(i).regs[1] = static_cast<float>(i);
    }
    broadcast(f.warp(), 2, 2.0f);

    f.run(Program{make_v_mul_f32(10, 1, 2), make_ret()});

    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        EXPECT_FLOAT_EQ(f.lane(i).regs[10], static_cast<float>(i) * 2.0f) << "lane " << i;
    }
}

// ---------------------------------------------------------------------------
// VEC3
// ---------------------------------------------------------------------------

TEST(Scheduler, Vec3AddSubScale)
{
    Fixture f;
    for (Thread& t : f.warp().threads) {
        t.regs[0] = 1.0f;
        t.regs[1] = 2.0f;
        t.regs[2] = 3.0f;
        t.regs[3] = 10.0f;
        t.regs[4] = 20.0f;
        t.regs[5] = 30.0f;
        t.regs[6] = 2.0f;  // scalar for scale
    }

    f.run(Program{
        make_v_add_vec3_f32(10, 0, 3),
        make_v_sub_vec3_f32(13, 3, 0),
        make_v_scale_vec3_f32(16, 0, 6),
        make_ret(),
    });

    const Thread& t = f.lane(0);
    EXPECT_FLOAT_EQ(t.regs[10], 11.0f);
    EXPECT_FLOAT_EQ(t.regs[11], 22.0f);
    EXPECT_FLOAT_EQ(t.regs[12], 33.0f);
    EXPECT_FLOAT_EQ(t.regs[13], 9.0f);
    EXPECT_FLOAT_EQ(t.regs[14], 18.0f);
    EXPECT_FLOAT_EQ(t.regs[15], 27.0f);
    EXPECT_FLOAT_EQ(t.regs[16], 2.0f);
    EXPECT_FLOAT_EQ(t.regs[17], 4.0f);
    EXPECT_FLOAT_EQ(t.regs[18], 6.0f);
}

TEST(Scheduler, Vec3DotAndCross)
{
    Fixture f;
    for (Thread& t : f.warp().threads) {
        t.regs[0] = 1.0f;
        t.regs[1] = 0.0f;
        t.regs[2] = 0.0f;  // x axis
        t.regs[3] = 0.0f;
        t.regs[4] = 1.0f;
        t.regs[5] = 0.0f;  // y axis
    }

    f.run(Program{
        make_v_dot_vec3_f32(10, 0, 3),    // orthogonal -> 0
        make_v_dot_vec3_f32(11, 0, 0),    // with itself -> 1
        make_v_cross_vec3_f32(12, 0, 3),  // x cross y  -> z
        make_ret(),
    });

    const Thread& t = f.lane(0);
    EXPECT_FLOAT_EQ(t.regs[10], 0.0f);
    EXPECT_FLOAT_EQ(t.regs[11], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[12], 0.0f);
    EXPECT_FLOAT_EQ(t.regs[13], 0.0f);
    EXPECT_FLOAT_EQ(t.regs[14], 1.0f);
}

TEST(Scheduler, Vec3NormalizeYieldsUnitLength)
{
    Fixture f;
    for (Thread& t : f.warp().threads) {
        t.regs[0] = 3.0f;
        t.regs[1] = 4.0f;
        t.regs[2] = 0.0f;  // length 5
    }

    f.run(Program{make_v_norm_vec3_f32(10, 0), make_ret()});

    const Thread& t = f.lane(0);
    const float len = std::sqrt(t.regs[10] * t.regs[10] + t.regs[11] * t.regs[11] +
                                t.regs[12] * t.regs[12]);
    EXPECT_NEAR(len, 1.0f, 1e-5f);
    EXPECT_NEAR(t.regs[10], 0.6f, 1e-5f);
    EXPECT_NEAR(t.regs[11], 0.8f, 1e-5f);
}

TEST(Scheduler, Vec3OperandNearEndOfRegisterFileThrows)
{
    Fixture f;
    EXPECT_THROW(f.run(Program{make_v_add_vec3_f32(254, 0, 3), make_ret()}),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

TEST(Scheduler, CompareProducesBooleanFloats)
{
    Fixture f;
    broadcast(f.warp(), 1, 2.0f);
    broadcast(f.warp(), 2, 5.0f);

    f.run(Program{
        make_v_cmp_f32(10, 1, 2, CmpOp::LT),   // 2 < 5  -> 1
        make_v_cmp_f32(11, 1, 2, CmpOp::GT),   // 2 > 5  -> 0
        make_v_cmp_f32(12, 1, 1, CmpOp::EQ),   // 2 == 2 -> 1
        make_v_cmp_f32(13, 1, 2, CmpOp::NEQ),  // 2 != 5 -> 1
        make_v_cmp_f32(14, 1, 2, CmpOp::LE),   // 2 <= 5 -> 1
        make_v_cmp_f32(15, 1, 2, CmpOp::GE),   // 2 >= 5 -> 0
        make_ret(),
    });

    const Thread& t = f.lane(0);
    EXPECT_FLOAT_EQ(t.regs[10], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[11], 0.0f);
    EXPECT_FLOAT_EQ(t.regs[12], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[13], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[14], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[15], 0.0f);
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

TEST(Scheduler, LoadAndStoreGlobal)
{
    Fixture f;
    f.poke(64, 7.5f);
    broadcast(f.warp(), 1, 64.0f);   // base address in a register
    broadcast(f.warp(), 2, 128.0f);  // destination address

    f.run(Program{
        make_v_ld_global_f32(10, 1),        // regs[10] = global[64]
        make_v_st_global_f32(2, 10),        // global[128] = regs[10]
        make_v_ld_global_f32(11, 1, 0.0f),  // explicit zero offset
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 7.5f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[11], 7.5f);
    EXPECT_FLOAT_EQ(f.peek(128), 7.5f);
}

TEST(Scheduler, GlobalOffsetImmediateIsAddedToTheRegister)
{
    Fixture f;
    f.poke(64, 1.0f);
    f.poke(68, 2.0f);
    broadcast(f.warp(), 1, 64.0f);

    f.run(Program{
        make_v_ld_global_f32(10, 1, 0.0f),
        make_v_ld_global_f32(11, 1, 4.0f),
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 1.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[11], 2.0f);
}

TEST(Scheduler, LoadAndStoreShared)
{
    Fixture f;
    broadcast(f.warp(), 1, 0.0f);
    broadcast(f.warp(), 2, 42.0f);

    f.run(Program{
        make_v_st_shared_f32(1, 2),   // shared[0] = 42
        make_v_ld_shared_f32(10, 1),  // regs[10] = shared[0]
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 42.0f);
    EXPECT_FLOAT_EQ(f.block.shared_mem[0], 42.0f);
}

TEST(Scheduler, MemoryAccessOutsideTheSpanThrows)
{
    Fixture f;
    broadcast(f.warp(), 1, static_cast<float>(GLOBAL_BYTES));
    EXPECT_THROW(f.run(Program{make_v_ld_global_f32(10, 1), make_ret()}),
                 std::runtime_error);

    Fixture g;
    // Starts inside but the four bytes of the float run past the end.
    broadcast(g.warp(), 1, static_cast<float>(GLOBAL_BYTES - 2));
    EXPECT_THROW(g.run(Program{make_v_ld_global_f32(10, 1), make_ret()}),
                 std::runtime_error);
}

TEST(Scheduler, UnalignedMemoryAccessThrows)
{
    Fixture f;
    broadcast(f.warp(), 1, 2.0f);  // not a multiple of sizeof(float)
    EXPECT_THROW(f.run(Program{make_v_ld_global_f32(10, 1), make_ret()}),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

TEST(Scheduler, UnconditionalBranchSkipsInstructions)
{
    Fixture f;
    broadcast(f.warp(), 10, 0.0f);

    f.run(Program{
        make_bra(2),                // jump over the next instruction
        make_v_mov_f32(10, 99.0f),  // must not run
        make_v_mov_f32(11, 5.0f),
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 0.0f) << "branch did not skip";
    EXPECT_FLOAT_EQ(f.lane(0).regs[11], 5.0f);
}

TEST(Scheduler, ConvergedProgramWastesNothing)
{
    Fixture f;
    f.run(Program{make_v_mov_f32(10, 1.0f), make_ret()});

    // Every lane agreed at every step, so no slot was masked off.
    EXPECT_EQ(f.sched.stats().masked_lane_slots(), 0u);
    EXPECT_DOUBLE_EQ(f.sched.divergence_rate(), 0.0);
    EXPECT_EQ(f.sched.stats().active_lane_ops, f.sched.stats().lane_slots());
}

TEST(Scheduler, DivergentBranchSplitsTheWarp)
{
    Fixture f;
    // Even lanes take the branch, odd lanes fall through.
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        f.lane(i).regs[1] = (i % 2 == 0) ? 1.0f : 0.0f;
    }

    f.run(Program{
        make_bra_div(1, 3),        // taken -> instruction 3
        make_v_mov_f32(10, 1.0f),  // odd lanes only
        make_bra(2),               // odd lanes jump to RET
        make_v_mov_f32(10, 2.0f),  // even lanes only
        make_ret(),
    });

    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        EXPECT_FLOAT_EQ(f.lane(i).regs[10], (i % 2 == 0) ? 2.0f : 1.0f) << "lane " << i;
    }

    // The two sides had to be issued separately, so some capacity was wasted.
    EXPECT_GT(f.sched.stats().masked_lane_slots(), 0u);
    EXPECT_GT(f.sched.divergence_rate(), 0.0);
    EXPECT_LT(f.sched.divergence_rate(), 1.0);
}

TEST(Scheduler, ThreadsReconvergeAfterDiverging)
{
    Fixture f;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        f.lane(i).regs[1] = (i % 2 == 0) ? 1.0f : 0.0f;
    }

    // Both sides reach instruction 3, which must then be issued once for the
    // whole warp rather than twice.
    const Program prog{
        make_bra_div(1, 2),        // 0: taken -> 2
        make_v_mov_f32(10, 1.0f),  // 1: odd only
        make_v_mov_f32(11, 7.0f),  // 2: both, once reconverged
        make_ret(),                // 3
    };
    f.run(prog);

    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        EXPECT_FLOAT_EQ(f.lane(i).regs[11], 7.0f) << "lane " << i;
    }

    // 4 steps if reconvergence works (0, 1, 2, 3); more if it does not.
    EXPECT_EQ(f.sched.stats().warp_steps, 4u);
}

TEST(Scheduler, LoopWithBackwardBranch)
{
    Fixture f;
    broadcast(f.warp(), 1, 0.0f);  // counter
    broadcast(f.warp(), 2, 1.0f);  // increment
    broadcast(f.warp(), 3, 3.0f);  // limit

    f.run(Program{
        make_v_add_f32(1, 1, 2),             // 0: ++counter
        make_v_cmp_f32(4, 1, 3, CmpOp::LT),  // 1: counter < 3 ?
        make_bra_div(4, -2),                 // 2: jump back to 0 while true
        make_ret(),                          // 3
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[1], 3.0f) << "loop ran the wrong number of times";
}

// ---------------------------------------------------------------------------
// Termination and statistics
// ---------------------------------------------------------------------------

TEST(Scheduler, RetRetiresEveryThread)
{
    Fixture f;
    f.run(Program{make_ret()});

    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        EXPECT_FALSE(f.lane(i).active) << "lane " << i << " still running";
    }
}

TEST(Scheduler, RunsEveryWarpInTheBlock)
{
    ThreadBlock block = make_block(3, 0);
    std::vector<uint8_t> global(GLOBAL_BYTES, 0);
    WarpScheduler sched;

    sched.run(Program{make_v_mov_f32(10, 9.0f), make_ret()}, block,
              DeviceSpan{global.data(), global.size()});

    for (const Warp& w : block.warps) {
        for (const Thread& t : w.threads) {
            EXPECT_FLOAT_EQ(t.regs[10], 9.0f);
            EXPECT_FALSE(t.active);
        }
    }
    // 3 warps x 2 instructions.
    EXPECT_EQ(sched.stats().warp_steps, 6u);
}

TEST(Scheduler, EmptyStatsAreWellDefined)
{
    WarpScheduler sched;
    EXPECT_EQ(sched.stats().warp_steps, 0u);
    EXPECT_EQ(sched.stats().lane_slots(), 0u);
    EXPECT_EQ(sched.stats().masked_lane_slots(), 0u);
    EXPECT_DOUBLE_EQ(sched.divergence_rate(), 0.0) << "must not divide by zero";
}

TEST(Scheduler, ResetStatsClearsCounters)
{
    Fixture f;
    f.run(Program{make_v_mov_f32(10, 1.0f), make_ret()});
    ASSERT_GT(f.sched.stats().warp_steps, 0u);

    f.sched.reset_stats();
    EXPECT_EQ(f.sched.stats().warp_steps, 0u);
    EXPECT_EQ(f.sched.stats().active_lane_ops, 0u);
    EXPECT_DOUBLE_EQ(f.sched.divergence_rate(), 0.0);
}

TEST(Scheduler, ProgramCounterLeavingTheProgramThrows)
{
    Fixture f;
    // Falls off the end: no RET, so pc walks past the last instruction.
    EXPECT_THROW(f.run(Program{make_v_mov_f32(10, 1.0f)}), std::runtime_error);
}
