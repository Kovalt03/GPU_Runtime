#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "runtime.hpp"  // REG_GLOBAL_ID_X, seeded by a launch
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
// MAT4
// ---------------------------------------------------------------------------

namespace {

// Loads a row-major mat4 into 16 consecutive registers of every lane.
void load_mat4(Warp& warp, uint8_t reg, const float (&m)[16])
{
    for (Thread& t : warp.threads) {
        for (uint32_t i = 0; i < 16; ++i) {
            t.regs[reg + i] = m[i];
        }
    }
}

void load_vec4(Warp& warp, uint8_t reg, float x, float y, float z, float w)
{
    for (Thread& t : warp.threads) {
        t.regs[reg + 0] = x;
        t.regs[reg + 1] = y;
        t.regs[reg + 2] = z;
        t.regs[reg + 3] = w;
    }
}

}  // namespace

TEST(Scheduler, MatvecIdentityLeavesTheVectorAlone)
{
    Fixture f;
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    load_mat4(f.warp(), 16, identity);
    load_vec4(f.warp(), 4, 1.0f, 2.0f, 3.0f, 1.0f);

    f.run(Program{make_v_matvec_mat4_f32(8, 16, 4), make_ret()});

    const Thread& t = f.lane(0);
    EXPECT_FLOAT_EQ(t.regs[8], 1.0f);
    EXPECT_FLOAT_EQ(t.regs[9], 2.0f);
    EXPECT_FLOAT_EQ(t.regs[10], 3.0f);
    EXPECT_FLOAT_EQ(t.regs[11], 1.0f);
}

TEST(Scheduler, MatvecTranslatesAPoint)
{
    // Row-major translation: the offsets sit in the last column, so they only
    // take effect when w is 1 — which is what separates a point from a
    // direction.
    Fixture f;
    const float translate[16] = {1, 0, 0, 10, 0, 1, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1};
    load_mat4(f.warp(), 16, translate);

    load_vec4(f.warp(), 4, 1.0f, 2.0f, 3.0f, 1.0f);  // a point
    f.run(Program{make_v_matvec_mat4_f32(8, 16, 4), make_ret()});
    EXPECT_FLOAT_EQ(f.lane(0).regs[8], 11.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[9], 22.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 33.0f);

    Fixture g;
    load_mat4(g.warp(), 16, translate);
    load_vec4(g.warp(), 4, 1.0f, 2.0f, 3.0f, 0.0f);  // a direction
    g.run(Program{make_v_matvec_mat4_f32(8, 16, 4), make_ret()});
    EXPECT_FLOAT_EQ(g.lane(0).regs[8], 1.0f) << "a direction must not translate";
    EXPECT_FLOAT_EQ(g.lane(0).regs[9], 2.0f);
    EXPECT_FLOAT_EQ(g.lane(0).regs[10], 3.0f);
}

TEST(Scheduler, MatvecSurvivesOverlappingOperands)
{
    // Every output component reads every input component, so writing dst[0]
    // before the rest are read would corrupt them.
    Fixture f;
    const float scale2[16] = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    load_mat4(f.warp(), 16, scale2);
    load_vec4(f.warp(), 4, 1.0f, 2.0f, 3.0f, 1.0f);

    f.run(Program{make_v_matvec_mat4_f32(4, 16, 4), make_ret()});  // dst == src1

    EXPECT_FLOAT_EQ(f.lane(0).regs[4], 2.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[5], 4.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[6], 6.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[7], 1.0f);
}

TEST(Scheduler, MatvecRejectsUnalignedOperands)
{
    // VEC4 and wider start on a multiple of 4, so that a later
    // V_LD_GLOBAL_MAT4_F32 can map an aligned block onto an aligned address.
    Fixture f;
    EXPECT_THROW(f.run(Program{make_v_matvec_mat4_f32(1, 16, 4), make_ret()}),
                 std::runtime_error);
    Fixture g;
    EXPECT_THROW(g.run(Program{make_v_matvec_mat4_f32(8, 17, 4), make_ret()}),
                 std::runtime_error);
    Fixture h;
    EXPECT_THROW(h.run(Program{make_v_matvec_mat4_f32(8, 16, 5), make_ret()}),
                 std::runtime_error);
}

TEST(Scheduler, MatvecRejectsAMatrixPastTheEndOfTheFile)
{
    // A mat4 claims 16 registers, so the last place one fits is r240.
    Fixture f;
    EXPECT_NO_THROW(f.run(Program{make_v_matvec_mat4_f32(0, 240, 4), make_ret()}));
    Fixture g;
    EXPECT_THROW(g.run(Program{make_v_matvec_mat4_f32(0, 244, 4), make_ret()}),
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

// ---------------------------------------------------------------------------
// BARRIER — the one opcode the scheduler handles above the lane loop
// ---------------------------------------------------------------------------

TEST(Scheduler, BarrierMakesOneWarpsWritesVisibleToAnother)
{
    // What the whole thing is for. Warp 0 writes shared memory, warp 1 reads
    // it, and only the barrier between them makes the order hold.
    //
    // The writer is padded so that round-robin puts the reader at its load long
    // before the writer reaches its store. Without that the interleaving
    // happens to order them correctly on its own and the test proves nothing —
    // it passed against a barrier that did nothing at all until this was added.
    ThreadBlock block = make_block(2);

    Program p;
    p.push_back(make_v_mov_f32(0, 0.0f));                           // 0: address
    p.push_back(make_v_mov_f32(1, 7.0f));                           // 1: value
    p.push_back(make_v_mov_f32(2, 32.0f));                          // 2: boundary
    p.push_back(make_v_cmp_f32(3, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 3
    p.push_back(make_bra_div(3, 8));                                // 4: readers -> 12
    for (int i = 0; i < 5; ++i) {                                   // 5..9: padding
        p.push_back(make_v_mov_f32(5, 0.0f));
    }
    p.push_back(make_v_st_shared_f32(0, 1, 0.0f));  // 10: the write
    p.push_back(make_bra(1));                       // 11: -> 12
    p.push_back(make_barrier());                    // 12
    p.push_back(make_v_ld_shared_f32(4, 0, 0.0f));  // 13: the read
    p.push_back(make_ret());                        // 14

    for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            block.warps[w].threads[lane].regs[REG_GLOBAL_ID_X] =
                static_cast<float>(w * WARP_SIZE + lane);
        }
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(64, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_FLOAT_EQ(block.warps[1].threads[0].regs[4], 7.0f)
        << "the reader has to see the writer's value";
}

TEST(Scheduler, BarrierLetsEveryWarpThrough)
{
    // Two warps, a barrier, then both keep going. Neither should be left
    // waiting when run() returns.
    ThreadBlock block = make_block(2);

    Program p;
    p.push_back(make_barrier());
    p.push_back(make_v_mov_f32(0, 5.0f));
    p.push_back(make_ret());

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    for (const Warp& warp : block.warps) {
        EXPECT_FALSE(warp.at_barrier) << "nobody is left waiting";
        EXPECT_FLOAT_EQ(warp.threads[0].regs[0], 5.0f) << "and everyone ran on";
    }
}

TEST(Scheduler, BarrierRejectsBeingReachedUnderDivergence)
{
    // Half the lanes branch past the barrier, so the warp arrives with only the
    // other half and those lanes would be synchronised against threads that
    // never come. CUDA leaves this to hang or go quietly wrong; simulating the
    // rule is worth nothing unless it is enforced.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(0, 16.0f));
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 0, CmpOp::GE));
    p.push_back(make_bra_div(1, 2));  // lanes 16..31 jump over the barrier
    p.push_back(make_barrier());
    p.push_back(make_ret());

    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[REG_GLOBAL_ID_X] = static_cast<float>(lane);
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);
}

TEST(Scheduler, BarrierDoesNotWaitForAWarpThatHasRetired)
{
    // Warp 1 leaves before the barrier; warp 0 must not be stranded there.
    ThreadBlock block = make_block(2);

    Program p;
    p.push_back(make_v_mov_f32(0, 32.0f));
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 0, CmpOp::GE));
    p.push_back(make_bra_div(1, 3));  // warp 1 leaves at once
    p.push_back(make_barrier());
    p.push_back(make_v_mov_f32(2, 9.0f));
    p.push_back(make_ret());

    for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            block.warps[w].threads[lane].regs[REG_GLOBAL_ID_X] =
                static_cast<float>(w * WARP_SIZE + lane);
        }
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[2], 9.0f)
        << "warp 0 must not be stranded";
}

TEST(Scheduler, LowestPcFirstStrandsALaneWaitingOnAHigherOne)
{
    // The deadlock independent thread scheduling exists to fix, reachable from
    // a kernel that reads as correct.
    //
    // Lane 0 spins on a flag in memory; every other lane jumps forward to the
    // store that would set it. The spin sits at a lower pc than the store, and
    // this scheduler always issues the lowest live pc, so the writer is never
    // reached — the readiest lane starves the one it is waiting for.
    //
    // Volta gives each thread its own pc, which this machine already does, and
    // then schedules them independently, which it does not. What is missing is
    // a policy, not a data structure.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(0, 1.0f));                           // 0
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 0, CmpOp::GE));  // 1: lane >= 1
    p.push_back(make_bra_div(1, 6));                                // 2: to the writer
    p.push_back(make_v_mov_f32(2, 0.0f));                           // 3: spin, addr 0
    p.push_back(make_v_ld_global_f32(3, 2, 0.0f));                  // 4:   flag = mem[0]
    p.push_back(make_v_cmp_f32(4, 3, 2, CmpOp::EQ));                // 5:   still zero?
    p.push_back(make_bra_div(4, -3));                               // 6:   back to 4
    p.push_back(make_ret());                                        // 7
    p.push_back(make_v_mov_f32(5, 0.0f));                           // 8: writer, addr 0
    p.push_back(make_v_mov_f32(6, 1.0f));                           // 9
    p.push_back(make_v_st_global_f32(5, 6, 0.0f));                  // 10: mem[0] = 1
    p.push_back(make_ret());                                        // 11

    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[REG_GLOBAL_ID_X] = static_cast<float>(lane);
    }

    WarpScheduler scheduler;
    // Small enough that asserting the hang costs milliseconds. The spin issues
    // one instruction a turn and would otherwise run to the default budget.
    scheduler.set_step_budget(10000);

    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);

    // Which is the deadlock rather than merely something throwing: the spinning
    // lane is still inside its loop, the writer has not issued its first
    // instruction, and the flag it was waiting on was never set. A budget that
    // fired for any other reason would not leave the block in this state.
    EXPECT_EQ(block.warps[0].threads[0].pc, 4u) << "lane 0 left its spin";
    EXPECT_EQ(block.warps[0].threads[1].pc, 8u) << "lane 1 ran past the store";
    float flag = -1.0f;
    std::memcpy(&flag, memory.data(), sizeof(float));
    EXPECT_FLOAT_EQ(flag, 0.0f) << "the store ran, so nothing was starved";
}

TEST(Scheduler, TheStepBudgetDoesNotStandInTheWayOfAKernelThatFinishes)
{
    // The budget is worth nothing if a working kernel can meet it. A block that
    // retires normally must not depend on how much room it was given, so the
    // tightest budget that lets this one through is well under what any kernel
    // here uses.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(0, 7.0f));
    p.push_back(make_ret());

    WarpScheduler scheduler;
    scheduler.set_step_budget(8);

    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[0], 7.0f);
}
