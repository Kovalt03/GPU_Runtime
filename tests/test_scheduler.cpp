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

// Every lane of a warp, which is what a converged S_ instruction declares.
// Spelled out at each call site rather than defaulted, because naming a lane
// that is not there is the mistake the mask exists to expose.
constexpr uint32_t ALL_LANES = 0xFFFFFFFFu;
constexpr uint32_t LOW_HALF = 0x0000FFFFu;

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

// A warp-internal producer-consumer, which ordinary CUDA writes and which the
// two scheduling policies disagree about entirely.
//
// Lane 0 spins on a flag in memory; every other lane jumps forward to the store
// that sets it. The spin sits at a lower pc than the store, so the two policies
// disagree about it entirely — which is why both tests build it from here rather
// than from two copies that could drift.
Program producer_consumer_program()
{
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
    return p;
}

// A kernel that ignores its arguments and always produces the same program,
// which is what a launch needs when the point is the machine and not the work.
KernelFunc constant_kernel(Program prog)
{
    return [prog](void**) { return prog; };
}

void seed_lane_ids(ThreadBlock& block)
{
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[REG_GLOBAL_ID_X] = static_cast<float>(lane);
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
    const Program p = producer_consumer_program();
    seed_lane_ids(block);

    WarpScheduler scheduler;
    // Small enough that asserting the hang costs milliseconds. The spin issues
    // one instruction a turn and would otherwise run to the default budget.
    scheduler.set_cycle_budget(10000);

    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);

    // Which is the deadlock rather than merely something throwing: the spinning
    // lane is still inside its loop, the writer has not issued its first
    // instruction, and the flag it was waiting on was never set. A budget that
    // fired for any other reason would not leave the block in this state.
    // Inside the spin (4 load, 5 compare, 6 branch back) rather than at a named
    // instruction: which of the three the budget interrupts is a matter of what
    // cycle it fired on, and the claim here is that the lane never got out.
    EXPECT_GE(block.warps[0].threads[0].pc, 4u) << "lane 0 left its spin";
    EXPECT_LE(block.warps[0].threads[0].pc, 6u) << "lane 0 left its spin";
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
    scheduler.set_cycle_budget(8);

    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[0], 7.0f);
}

// Enabled by [m1]. Independent has to run what LowestPc strands, and leave every
// existing measurement where it is — a policy that changed the issued work would
// invalidate benchmarks/RESULTS.md rather than extend it.
// Enabled by [m1], alongside the test below. One warp cannot catch a cursor
// kept on the scheduler instead of on the warp: with nothing else taking turns,
// a shared cursor and a private one behave identically. Two warps, each with a
// lane to starve, is the smallest scene where they part — the other warp's turn
// advances a shared cursor and this warp's next choice skips past the pc it was
// supposed to visit.
TEST(Scheduler, IndependentSchedulingIsFairWithinEachWarpSeparately)
{
    ThreadBlock block = make_block(2);
    const Program p = producer_consumer_program();
    for (uint32_t w = 0; w < 2; ++w) {
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            block.warps[w].threads[lane].regs[REG_GLOBAL_ID_X] = static_cast<float>(lane);
        }
    }

    WarpScheduler scheduler;
    scheduler.set_policy(WarpPolicy::Independent);
    scheduler.set_cycle_budget(10000);

    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    float flag = -1.0f;
    std::memcpy(&flag, memory.data(), sizeof(float));
    EXPECT_FLOAT_EQ(flag, 1.0f) << "a warp was starved by another warp's turns";
}

TEST(Scheduler, IndependentSchedulingRunsWhatLowestPcStrands)
{
    ThreadBlock block = make_block(1);
    const Program p = producer_consumer_program();
    seed_lane_ids(block);

    WarpScheduler scheduler;
    scheduler.set_policy(WarpPolicy::Independent);
    scheduler.set_cycle_budget(10000);

    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    float flag = -1.0f;
    std::memcpy(&flag, memory.data(), sizeof(float));
    EXPECT_FLOAT_EQ(flag, 1.0f) << "the writer never issued";
}

// Lanes 0..15 vote yes and 16..31 vote no, so S_ANY and S_ALL disagree. A scene
// where every lane votes the same way makes them indistinguishable, and would
// pass against an implementation that read neither the mask nor the predicate.
Program voting_program()
{
    Program p;
    p.push_back(make_v_mov_f32(2, 16.0f));
    p.push_back(make_v_cmp_f32(5, REG_GLOBAL_ID_X, 2, CmpOp::LT));  // r5 = lane < 16
    p.push_back(make_s_ballot(0, 5, ALL_LANES));                    // m0 = ballot(r5)
    p.push_back(make_s_any(6, 0, ALL_LANES));                       // r6 = any(m0)
    p.push_back(make_s_all(7, 0, ALL_LANES));                       // r7 = all(m0)
    p.push_back(make_ret());
    return p;
}

TEST(Scheduler, BallotReportsWhichLanesVoted)
{
    ThreadBlock block = make_block(1);
    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(voting_program(), block, DeviceSpan{memory.data(), memory.size()});

    // The predicate, not the active mask. An implementation that balloted on
    // activity alone would report all ones here and pass every other assertion
    // in this file.
    EXPECT_EQ(block.warps[0].masks[0], 0x0000FFFFu);
}

TEST(Scheduler, BallotCountsOnlyTheLanesItNamed)
{
    // Lanes 16..31 branch past the ballot, so the ballot declares the low half.
    // Their registers hold a yes, and the answer must not include them: the
    // result is over the lanes the program named, not over whatever the file
    // happens to contain.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(5, 1.0f));                           // 0: every lane yes
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 1
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 2: lane >= 16
    p.push_back(make_bra_div(1, 2));                                // 3: those jump on
    p.push_back(make_s_ballot(0, 5, LOW_HALF));                     // 4
    p.push_back(make_ret());                                        // 5

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_EQ(block.warps[0].masks[0], LOW_HALF)
        << "lanes that branched past the ballot voted in it";
}

TEST(Scheduler, AnyAndAllReduceTheBallotToOneAnswerPerWarp)
{
    ThreadBlock block = make_block(1);
    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(voting_program(), block, DeviceSpan{memory.data(), memory.size()});

    // The same answer in every lane, including the ones that voted no: this is
    // a reduction, not a copy of each lane's own vote. Half the warp voting yes
    // is what separates the two — any is true, all is false.
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        EXPECT_FLOAT_EQ(block.warps[0].threads[lane].regs[6], 1.0f)
            << "S_ANY disagreed in lane " << lane;
        EXPECT_FLOAT_EQ(block.warps[0].threads[lane].regs[7], 0.0f)
            << "S_ALL disagreed in lane " << lane;
    }
}

TEST(Scheduler, AllIsTrueWhenEveryParticipatingLaneVoted)
{
    // The other half of S_ALL, and the reason it compares against the active
    // mask rather than against all ones: here every lane votes yes.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(5, 1.0f));
    p.push_back(make_s_ballot(0, 5, ALL_LANES));
    p.push_back(make_s_all(7, 0, ALL_LANES));
    p.push_back(make_ret());

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[7], 1.0f);
}

TEST(Scheduler, AllComparesAgainstWhoArrivedRatherThanAllThirtyTwo)
{
    // Lanes 16..31 branch past, and every lane that does arrive votes yes. So
    // S_ALL is true — of the lanes participating, all of them voted.
    //
    // The scene exists because a full warp cannot tell the two rules apart: with
    // every lane active the participating mask IS all ones, and comparing
    // against the constant gives the same answer. Only under divergence do they
    // part, and then the constant is wrong every time.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(5, 1.0f));                           // 0: every lane yes
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 1
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 2
    p.push_back(make_bra_div(1, 3));                                // 3: 16.. skip to 6
    p.push_back(make_s_ballot(0, 5, LOW_HALF));                     // 4
    p.push_back(make_s_all(7, 0, LOW_HALF));                        // 5
    p.push_back(make_ret());                                        // 6

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_EQ(block.warps[0].masks[0], LOW_HALF);
    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[7], 1.0f)
        << "S_ALL demanded votes from lanes that were never there";
}

TEST(Scheduler, AWarpUniformResultDoesNotReachLanesThatBranchedAway)
{
    // A lane that skipped the reduction is somewhere else in the program, and
    // its registers belong to that path. Writing the answer to all 32 would
    // reach through the divergence and overwrite a value the other branch is
    // still using.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(7, 9.0f));                           // 0: sentinel
    p.push_back(make_v_mov_f32(5, 1.0f));                           // 1
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 2
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 3
    p.push_back(make_bra_div(1, 3));                                // 4: 16.. skip to 7
    p.push_back(make_s_ballot(0, 5, LOW_HALF));                     // 5
    p.push_back(make_s_any(7, 0, LOW_HALF));                        // 6: overwrites r7
    p.push_back(make_ret());                                        // 7

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[7], 1.0f) << "lane 0 took part";
    EXPECT_FLOAT_EQ(block.warps[0].threads[31].regs[7], 9.0f)
        << "lane 31 branched away and had its register written anyway";
}

TEST(Scheduler, WarpPrimitivesRejectAnOperandFromTheWrongFile)
{
    // dst and src0 name different files depending on the opcode — the mask file
    // is four deep where the lane file is 256, so an index that is ordinary in
    // one is out of range in the other. Nothing in the encoding says which.
    const Instruction bad[] = {
        make_s_ballot(WARP_MASK_REGISTERS, 5, ALL_LANES),  // dst is a mask
        make_s_any(6, WARP_MASK_REGISTERS, ALL_LANES),     // src0 is a mask
        make_s_all(7, WARP_MASK_REGISTERS, ALL_LANES),
    };

    for (const Instruction& instr : bad) {
        ThreadBlock block = make_block(1);
        Program p;
        p.push_back(instr);
        p.push_back(make_ret());

        WarpScheduler scheduler;
        std::vector<uint8_t> memory(16, 0);
        EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                     std::runtime_error)
            << opcode_name(instr.op);
    }
}

TEST(Scheduler, ACollectiveIgnoresLanesThatAreHereButNotNamed)
{
    // Nothing has diverged: all 32 lanes are at the instruction. The mask names
    // half of them anyway, and the other half must take no part — they are
    // present, but the program did not ask them.
    //
    // The case that separates a declared set from an inferred one. Wherever a
    // warp has diverged the two coincide, because a lane that branched away is
    // not active either; only a converged warp with a partial declaration tells
    // them apart.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(5, 1.0f));        // every lane votes yes
    p.push_back(make_s_ballot(0, 5, LOW_HALF));  // but only sixteen are asked
    p.push_back(make_s_all(7, 0, LOW_HALF));
    p.push_back(make_s_any(6, 0, LOW_HALF));
    p.push_back(make_ret());

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_EQ(block.warps[0].masks[0], LOW_HALF) << "lanes outside the mask voted";

    // Every named lane voted, so S_ALL is true — against the declared set. Read
    // against the active mask it would be false, all 32 lanes being here.
    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[7], 1.0f);

    // And the answer reaches the participants only.
    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[6], 1.0f);
    EXPECT_FLOAT_EQ(block.warps[0].threads[31].regs[6], 0.0f)
        << "a lane outside the mask was written to";
}

TEST(Scheduler, NamingALaneThatIsNotThereIsRefused)
{
    // The promise the mask makes. Lanes 16..31 have branched away, so declaring
    // all 32 names lanes that cannot take part — under LowestPc because they
    // are elsewhere in the program, and under Independent because they may
    // simply not have been scheduled yet. CUDA leaves this undefined; a
    // simulator can say so instead.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(5, 1.0f));                           // 0
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 1
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 2
    p.push_back(make_bra_div(1, 2));                                // 3: 16.. -> 5
    p.push_back(make_s_ballot(0, 5, ALL_LANES));                    // 4: over-declares
    p.push_back(make_ret());                                        // 5

    seed_lane_ids(block);

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);
}

TEST(Scheduler, SyncwarpGathersLanesThatIndependentSchedulingLetDrift)
{
    // Lanes take different paths and then meet. Under LowestPc they converge on
    // their own, because a lane that ran ahead is made to wait; under
    // Independent nothing makes them, and the sync is the only thing that can.
    //
    // Both policies must produce the same frame, which is what makes the sync
    // worth having rather than merely present.
    Program p;
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 0
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 1
    p.push_back(make_bra_div(1, 3));                                // 2: 16.. -> 5
    p.push_back(make_v_mov_f32(3, 1.0f));                           // 3
    p.push_back(make_bra(2));                                       // 4: -> 6
    p.push_back(make_v_mov_f32(3, 2.0f));                           // 5
    p.push_back(make_s_syncwarp(ALL_LANES));                        // 6
    p.push_back(make_v_mov_f32(9, 7.0f));                           // 7
    p.push_back(make_ret());                                        // 8

    for (const WarpPolicy policy : {WarpPolicy::LowestPc, WarpPolicy::Independent}) {
        ThreadBlock block = make_block(1);
        seed_lane_ids(block);

        WarpScheduler scheduler;
        scheduler.set_policy(policy);
        std::vector<uint8_t> memory(16, 0);
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            EXPECT_FLOAT_EQ(block.warps[0].threads[lane].regs[9], 7.0f)
                << "lane " << lane << " never passed the sync";
        }
    }
}

TEST(Scheduler, SyncwarpRefusesToWaitForALaneAlreadyPastIt)
{
    // A participant that branched beyond the sync is never coming back. Waiting
    // would spin until the step budget ran out and report only that the block
    // did not finish, which says nothing about why.
    ThreadBlock block = make_block(1);

    Program p;
    p.push_back(make_v_mov_f32(2, 16.0f));                          // 0
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));  // 1
    p.push_back(make_bra_div(1, 3));                                // 2: 16.. -> 4
    p.push_back(make_s_syncwarp(ALL_LANES));                        // 3: waits for them
    p.push_back(make_ret());                                        // 4

    seed_lane_ids(block);

    WarpScheduler scheduler;
    scheduler.set_cycle_budget(10000);
    std::vector<uint8_t> memory(16, 0);

    // Checked by message, not merely by type: without the look-ahead this waits
    // until the step budget runs out and throws anyway, so EXPECT_THROW alone
    // would pass against the very implementation this test rules out.
    try {
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
        FAIL() << "the sync completed with a lane that had gone past it";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("S_SYNCWARP"), std::string::npos)
            << "gave up on the step budget rather than naming the lane: " << e.what();
    }
}

// lane i takes lane i+1, wrapping, in place. The rotation is what makes this a
// real exchange rather than a broadcast, and reusing one register for dst and
// src0 is what lets a write reach a source another lane has yet to read.
Program rotate_in_place(uint8_t reg, uint32_t participants)
{
    Program p;
    p.push_back(make_v_mov_f32(1, 1.0f));
    p.push_back(make_v_mov_f32(2, static_cast<float>(WARP_SIZE)));
    p.push_back(make_v_add_f32(6, REG_GLOBAL_ID_X, 1));  // lane + 1
    p.push_back(make_v_cmp_f32(3, 6, 2, CmpOp::LT));     // still in range?
    p.push_back(make_bra_div(3, 2));                     // yes -> skip
    p.push_back(make_v_mov_f32(6, 0.0f));                // no  -> wrap to 0
    p.push_back(make_v_shuffle_f32(reg, reg, 6, participants));
    p.push_back(make_ret());
    return p;
}

TEST(Scheduler, ShuffleReadsEveryLaneBeforeWritingAny)
{
    // The last lane is the one that catches it. Reading forward while iterating
    // forward means every other lane takes a source nothing has touched yet;
    // only the wrap reaches a lane already written, so the first thirty-one
    // agree whether or not the reads were separated from the writes.
    ThreadBlock block = make_block(1);
    seed_lane_ids(block);
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[5] = static_cast<float>(lane);
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(rotate_in_place(5, ALL_LANES), block,
                  DeviceSpan{memory.data(), memory.size()});

    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        EXPECT_FLOAT_EQ(block.warps[0].threads[lane].regs[5],
                        static_cast<float>((lane + 1) % WARP_SIZE))
            << "lane " << lane;
    }
}

TEST(Scheduler, ShuffleRefusesALaneIndexThatIsNotOne)
{
    // The index is a float a kernel computed, so it can be fractional or well
    // outside the warp. Used as a subscript it would read past a 33 KB warp
    // into whatever follows, and quietly — the run below returned zero before
    // this check existed.
    for (const float index : {3.7f, 900.0f, -5.0f}) {
        ThreadBlock block = make_block(1);
        Program p;
        p.push_back(make_v_shuffle_f32(7, 5, 6, ALL_LANES));
        p.push_back(make_ret());
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            block.warps[0].threads[lane].regs[6] = index;
        }

        WarpScheduler scheduler;
        std::vector<uint8_t> memory(16, 0);
        EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                     std::runtime_error)
            << index;
    }
}

TEST(Scheduler, ShuffleIgnoresTheIndexHeldByALaneTakingNoPart)
{
    // A lane outside the mask is on another path, and the register the shuffle
    // reads its index from belongs to that path. Validating it would fail the
    // instruction over a lane that is not in it.
    ThreadBlock block = make_block(1);
    Program p;
    p.push_back(make_v_shuffle_f32(7, 5, 6, LOW_HALF));
    p.push_back(make_ret());
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[5] = 100.0f + static_cast<float>(lane);
        block.warps[0].threads[lane].regs[6] = (lane < 16) ? 0.0f : 12345.6f;
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_FLOAT_EQ(block.warps[0].threads[0].regs[7], 100.0f);
}

TEST(Scheduler, ShuffleRefusesToReadOutsideTheParticipationMask)
{
    // A participant has to end up with a value. Skipping the read instead would
    // leave its dst holding whatever the gather buffer started as — a number
    // nobody computed, in a frame that looks plausible.
    ThreadBlock block = make_block(1);
    Program p;
    p.push_back(make_v_shuffle_f32(7, 5, 6, LOW_HALF));
    p.push_back(make_ret());
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        block.warps[0].threads[lane].regs[5] = 100.0f + static_cast<float>(lane);
        block.warps[0].threads[lane].regs[6] = (lane == 3) ? 20.0f : 0.0f;
    }

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);
}

TEST(Scheduler, AWarpCarriesItsOwnMaskRegisters)
{
    // The storage decision, pinned before anything writes to it: a 32-bit mask
    // cannot live in the float register file, which is exact only to 2^24, so
    // the top eight lanes of a ballot would be lost. Masks belong to the warp,
    // not the lane, and there are several so a kernel holding one ballot can
    // take another.
    const Warp warp;
    EXPECT_EQ(warp.masks.size(), WARP_MASK_REGISTERS);
    for (const uint32_t mask : warp.masks) {
        EXPECT_EQ(mask, 0u) << "a warp must start with no lanes balloted";
    }

    // The reason the file exists at all: this is the value a full ballot has to
    // hold, and a float cannot hold it. Compared as doubles because converting
    // the rounded float back to uint32_t is undefined — it lands one above the
    // range, which is exactly the problem being demonstrated.
    const uint32_t all_lanes = 0xFFFFFFFFu;
    EXPECT_NE(static_cast<double>(static_cast<float>(all_lanes)),
              static_cast<double>(all_lanes));

    // And the boundary it starts at: lane 24 is the first whose bit a float
    // cannot carry alongside the ones below it.
    EXPECT_EQ(static_cast<double>(static_cast<float>((1u << 24) - 1)),
              static_cast<double>((1u << 24) - 1));
    EXPECT_NE(static_cast<double>(static_cast<float>((1u << 25) - 1)),
              static_cast<double>((1u << 25) - 1));
}

// Every lane loads from base + stride*lane, and the stride decides how many
// cache lines a warp touches.
Program strided_load(float stride)
{
    Program p;
    p.push_back(make_v_mov_f32(1, stride));
    p.push_back(make_v_mul_f32(2, REG_GLOBAL_ID_X, 1));  // address
    p.push_back(make_v_ld_global_f32(3, 2, 0.0f));
    p.push_back(make_ret());
    return p;
}

uint64_t weighted_for(float stride, MemoryModel model, size_t bytes)
{
    ThreadBlock block = make_block(1);
    seed_lane_ids(block);

    WarpScheduler scheduler;
    scheduler.set_memory_model(model);
    std::vector<uint8_t> memory(bytes, 0);
    scheduler.run(strided_load(stride), block, DeviceSpan{memory.data(), memory.size()});
    return scheduler.stats().weighted_lane_ops;
}

TEST(Scheduler, ACoalescedLoadCostsALineWhereAScatteredOneCostsThirtyTwo)
{
    // Four bytes a lane puts all 32 in one 128-byte line; 128 bytes a lane puts
    // each in its own. Both scenes are needed: with only one of them, counting
    // lines and counting lanes differ by a constant and either would pass.
    const uint64_t adjacent = weighted_for(4.0f, MemoryModel::Coalesced, 1 << 14);
    const uint64_t scattered = weighted_for(128.0f, MemoryModel::Coalesced, 1 << 14);

    EXPECT_EQ(scattered, adjacent + 31 * instruction_cost(Opcode::V_LD_GLOBAL_F32))
        << "a scattered warp did not pay for 32 lines against one";

    // And the shape of it: the load is the only instruction whose cost moved, so
    // the difference is exactly 31 extra lines.
    EXPECT_EQ(scattered - adjacent, 31u * instruction_cost(Opcode::V_LD_GLOBAL_F32));
}

TEST(Scheduler, TheFlatModelCannotTellTheTwoApart)
{
    // Which is why the tables taken under it say nothing about memory layout —
    // and why V_LD_GLOBAL_VEC3_F32 has a name reserved and no implementation.
    EXPECT_EQ(weighted_for(4.0f, MemoryModel::Flat, 1 << 14),
              weighted_for(128.0f, MemoryModel::Flat, 1 << 14));
}

TEST(Scheduler, EveryLaneReadingOneAddressIsOneLine)
{
    // A broadcast is one line, and so is the adjacent case — but there the 32
    // addresses all differ. Counting distinct addresses instead of distinct
    // lines would pass the scattered test and separate these two, so comparing
    // them is what pins the unit.
    EXPECT_EQ(weighted_for(0.0f, MemoryModel::Coalesced, 1 << 14),
              weighted_for(4.0f, MemoryModel::Coalesced, 1 << 14));
}

TEST(Scheduler, IgnoringLatencyIsWhatMakesOccupancyWorthless)
{
    // The model every figure in benchmarks/ was taken under. instruction_latency
    // has numbers in it now, and Ignored is where they are not read: no warp
    // waits, so nothing stalls and resident warps neither help nor hinder.
    ThreadBlock block = make_block(4);
    Program p;
    p.push_back(make_v_mov_f32(0, 0.0f));
    p.push_back(make_v_ld_global_f32(1, 0, 0.0f));
    p.push_back(make_v_add_f32(2, 1, 1));  // depends on the load
    p.push_back(make_ret());

    WarpScheduler scheduler;
    std::vector<uint8_t> memory(64, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_EQ(scheduler.stats().stall_steps, 0u);
    EXPECT_EQ(scheduler.stats().cycles, scheduler.stats().warp_steps);
}

TEST(Scheduler, AnInstructionWithNoResultMakesNothingWait)
{
    // Which instructions carry a latency is a modelling choice, and these are the
    // ones it says nothing waits on: a store hands off and carries on, and a
    // branch or a barrier produces no value at all.
    for (const Opcode op :
         {Opcode::V_ST_GLOBAL_F32, Opcode::V_ST_SHARED_F32, Opcode::BRA, Opcode::BRA_DIV,
          Opcode::BARRIER, Opcode::S_SYNCWARP, Opcode::RET}) {
        EXPECT_EQ(instruction_latency(op), 0u) << opcode_name(op);
    }

    // And a global load, whose answer depends on where the line was found — the
    // memory model reports it, not this table.
    EXPECT_EQ(instruction_latency(Opcode::V_LD_GLOBAL_F32), 0u);

    // While arithmetic does, or a dependent instruction could never wait.
    EXPECT_GT(instruction_latency(Opcode::V_ADD_F32), 0u);
    EXPECT_GT(instruction_latency(Opcode::V_RCP_F32),
              instruction_latency(Opcode::V_ADD_F32))
        << "a special function unit is no faster than an adder";
    EXPECT_GT(instruction_latency(Opcode::V_LD_SHARED_F32),
              instruction_latency(Opcode::V_RCP_F32))
        << "on-chip memory is no faster than arithmetic";
}

TEST(Scheduler, ModelledSchedulingIssuesTheSameWorkAsIgnored)
{
    // The cycle loop has to reproduce the one every figure in benchmarks/ came
    // from, not merely resemble it. Issued work is the part that carries over: the
    // same instructions reach the same lanes whatever order the warps take their
    // turns in.
    //
    // Cycles and stalls are what the new model adds, and they are checked
    // separately — a warp waiting on a load is the point, not a discrepancy.
    Program p;
    p.push_back(make_v_mov_f32(2, 16.0f));
    p.push_back(make_v_cmp_f32(1, REG_GLOBAL_ID_X, 2, CmpOp::GE));
    p.push_back(make_bra_div(1, 2));  // divergent, so the masks differ per warp
    p.push_back(make_v_mov_f32(3, 1.0f));
    p.push_back(make_v_mov_f32(0, 0.0f));
    p.push_back(make_v_ld_global_f32(4, 0, 0.0f));
    p.push_back(make_barrier());
    p.push_back(make_v_add_f32(5, 4, 3));
    p.push_back(make_ret());

    for (const uint32_t warps : {1u, 2u, 4u}) {
        SchedulerStats readings[2];
        int i = 0;
        for (const LatencyModel model : {LatencyModel::Ignored, LatencyModel::Modelled}) {
            ThreadBlock block = make_block(warps);
            for (uint32_t w = 0; w < warps; ++w) {
                for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                    block.warps[w].threads[lane].regs[REG_GLOBAL_ID_X] =
                        static_cast<float>(lane);
                }
            }
            WarpScheduler scheduler;
            scheduler.set_latency_model(model);
            std::vector<uint8_t> memory(64, 0);
            scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
            readings[i++] = scheduler.stats();
        }

        EXPECT_EQ(readings[0].warp_steps, readings[1].warp_steps) << warps << " warps";
        EXPECT_EQ(readings[0].active_lane_ops, readings[1].active_lane_ops)
            << warps << " warps";
        EXPECT_EQ(readings[0].weighted_lane_ops, readings[1].weighted_lane_ops)
            << warps << " warps";

        // Ignored keeps cycles and issues equal by construction. Modelled cannot
        // fall below — time does not run backwards — but it need not exceed
        // either: enough warps and the waiting disappears, which is what
        // MoreWarpsCoverMoreOfTheWaiting is for.
        EXPECT_EQ(readings[0].cycles, readings[0].warp_steps) << warps << " warps";
        EXPECT_GE(readings[1].cycles, readings[1].warp_steps) << warps << " warps";
    }
}

TEST(Scheduler, MoreWarpsCoverMoreOfTheWaiting)
{
    // The reason warps are batched at all, and the first thing this simulator
    // could not say. One warp waits out every latency alone; enough of them and
    // the waiting disappears into each other's work.
    //
    // Arithmetic only, so the latencies are the tens rather than the hundreds and
    // a block can cover them completely. A chain, so each instruction waits on the
    // one before.
    Program p;
    p.push_back(make_v_mov_f32(0, 2.0f));
    p.push_back(make_v_add_f32(1, 0, 0));
    p.push_back(make_v_rcp_f32(2, 1));
    p.push_back(make_v_add_f32(3, 2, 2));
    p.push_back(make_v_rcp_f32(4, 3));
    p.push_back(make_ret());

    double previous_per_warp = 0.0;
    uint64_t previous_stalls = UINT64_MAX;
    for (const uint32_t warps : {1u, 2u, 4u, 8u, 16u}) {
        ThreadBlock block = make_block(warps);
        WarpScheduler scheduler;
        scheduler.set_latency_model(LatencyModel::Modelled);
        std::vector<uint8_t> memory(64, 0);
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

        const double per_warp = static_cast<double>(scheduler.stats().cycles) / warps;
        if (previous_per_warp > 0.0) {
            EXPECT_LT(per_warp, previous_per_warp)
                << warps << " warps did not cover more than " << (warps / 2);
        }
        EXPECT_LT(scheduler.stats().stall_steps, previous_stalls) << warps << " warps";

        previous_per_warp = per_warp;
        previous_stalls = scheduler.stats().stall_steps;
    }

    // And enough of them cover it entirely, which is what occupancy asks for.
    EXPECT_EQ(previous_stalls, 0u) << "sixteen warps still left the machine idle";
}

TEST(Scheduler, AGlobalLoadIsDeeperThanABlockCanCover)
{
    // The other half of occupancy: whether it is enough depends on what is being
    // hidden. A trip to memory is hundreds of cycles against a block's thirty-two
    // warps of a few each, so filling the block does not make it disappear — it
    // only spreads it.
    Program p;
    p.push_back(make_v_mov_f32(0, 0.0f));
    p.push_back(make_v_ld_global_f32(1, 0, 0.0f));
    p.push_back(make_v_add_f32(2, 1, 1));  // waits on the load
    p.push_back(make_ret());

    ThreadBlock block = make_block(32);
    WarpScheduler scheduler;
    scheduler.set_latency_model(LatencyModel::Modelled);
    std::vector<uint8_t> memory(1 << 14, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    EXPECT_GT(scheduler.stats().stall_steps, 0u)
        << "a memory latency was covered by one block, which it should not be";
}

TEST(Scheduler, ACachedLoadWaitsLessThanAnUncachedOne)
{
    // What connecting the two models buys. Every warp reads the same line, so the
    // first pays a trip to memory and the rest find it in L1 — which is a
    // difference in time as well as in issue capacity, and the flat model has
    // neither.
    Program p;
    p.push_back(make_v_mov_f32(0, 0.0f));
    p.push_back(make_v_ld_global_f32(1, 0, 0.0f));
    p.push_back(make_v_add_f32(2, 1, 1));
    p.push_back(make_ret());

    uint64_t cycles[2];
    int i = 0;
    for (const MemoryModel model : {MemoryModel::Flat, MemoryModel::Cached}) {
        ThreadBlock block = make_block(16);
        WarpScheduler scheduler;
        scheduler.set_latency_model(LatencyModel::Modelled);
        scheduler.set_memory_model(model);
        std::vector<uint8_t> memory(1 << 14, 0);
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
        cycles[i++] = scheduler.stats().cycles;
    }

    EXPECT_LT(cycles[1], cycles[0])
        << "the cache saved nothing in time, only in issue capacity";
}

TEST(Scheduler, LinesFetchedTogetherCostTheWorstOfThemNotTheSumOfThem)
{
    // A warp asking for several lines waits for the slowest, not for all of them in
    // turn — they are fetched together. Cost adds across lines because capacity is
    // spent on each; latency does not.
    //
    // Needs a scattered access to say anything. Every other scene here is a
    // broadcast, one line, where the sum and the worst are the same number.
    Program p;
    p.push_back(make_v_mov_f32(1, static_cast<float>(CACHE_LINE_BYTES)));
    p.push_back(make_v_mul_f32(2, REG_GLOBAL_ID_X, 1));  // lane i -> line i
    p.push_back(make_v_ld_global_f32(3, 2, 0.0f));
    p.push_back(make_v_add_f32(4, 3, 3));  // waits on the load
    p.push_back(make_ret());

    ThreadBlock block = make_block(1);
    seed_lane_ids(block);

    WarpScheduler scheduler;
    scheduler.set_latency_model(LatencyModel::Modelled);
    scheduler.set_memory_model(MemoryModel::Cached);
    std::vector<uint8_t> memory((WARP_SIZE + 1) * CACHE_LINE_BYTES, 0);
    scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});

    // Thirty-two lines, every one a miss, and one warp with nothing to hide behind.
    // Summing their latencies would put the total past thirty-two trips to memory;
    // taking the worst leaves it near one.
    EXPECT_EQ(scheduler.stats().cache_misses, WARP_SIZE);
    EXPECT_LT(scheduler.stats().cycles, 2 * MEMORY_LATENCY)
        << "the warp waited for its lines one after another";
}

TEST(Scheduler, AVectorLoadLandsThreeFloatsInThreeRegisters)
{
    Fixture f;
    f.poke(16, 1.0f);
    f.poke(20, 2.0f);
    f.poke(24, 3.0f);
    broadcast(f.warp(), 0, 16.0f);

    Program p;
    p.push_back(make_v_ld_global_vec3_f32(/*dst=*/10, /*addr_reg=*/0));
    p.push_back(make_ret());
    f.run(p);

    EXPECT_FLOAT_EQ(f.lane(0).regs[10], 1.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[11], 2.0f);
    EXPECT_FLOAT_EQ(f.lane(0).regs[12], 3.0f);

    // The range is checked before anything is written, as it is for the VEC3
    // arithmetic: a load that filled two registers and then threw would leave a
    // lane half updated.
    Fixture over;
    broadcast(over.warp(), 0, 16.0f);
    Program spill;
    spill.push_back(make_v_ld_global_vec3_f32(/*dst=*/254, /*addr_reg=*/0));
    spill.push_back(make_ret());
    EXPECT_THROW(over.run(spill), std::runtime_error);
}

TEST(Scheduler, DependenceIsNotDistinguishedFromIssueOrder)
{
    // The bound on what the cycle count can be asked. A warp that has issued
    // waits out the latency before issuing again, whether or not the next
    // instruction wanted the result, so a dependent chain and independent
    // accesses come out identical. Occupancy is what this models; a scoreboard
    // is what it does not.
    //
    // Pessimistic rather than generous, which is what makes a figure taken from
    // it an upper bound on what dependent loads cost.
    const auto cycles = [](const Program& p) {
        ThreadBlock block = make_block(1);
        std::vector<uint8_t> memory(GLOBAL_BYTES, 0);
        // Each slot points at the next, so the chained loads walk forward.
        for (size_t i = 0; i < 8; ++i) {
            const float next = static_cast<float>((i + 1) * sizeof(float));
            std::memcpy(memory.data() + i * sizeof(float), &next, sizeof(float));
        }
        WarpScheduler scheduler;
        scheduler.set_memory_model(MemoryModel::Cached);
        scheduler.set_latency_model(LatencyModel::Modelled);
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
        return scheduler.stats().cycles;
    };

    Program chained;
    chained.push_back(make_v_mov_f32(0, 0.0f));
    chained.push_back(make_v_ld_global_f32(1, 0));  // each address is the last result
    chained.push_back(make_v_ld_global_f32(2, 1));
    chained.push_back(make_v_ld_global_f32(3, 2));
    chained.push_back(make_ret());

    Program independent;
    independent.push_back(make_v_mov_f32(0, 0.0f));
    independent.push_back(make_v_ld_global_f32(1, 0, 0.0f));
    independent.push_back(make_v_ld_global_f32(2, 0, 4.0f));
    independent.push_back(make_v_ld_global_f32(3, 0, 8.0f));
    independent.push_back(make_ret());

    EXPECT_EQ(cycles(chained), cycles(independent));
}

TEST(Scheduler, ALoadIsPricedFromItsAddressAndNotFromWhatItFetched)
{
    // A load may name its address register as its destination, and the memory
    // model reads that register to decide which line was touched. Pricing after
    // the lanes had run charged the value just fetched as an address — the wrong
    // line where that value happened to be a plausible one, and a refusal where
    // it was negative or fractional, both for a program that is not in error.
    for (const float payload : {-1.0f, 3.5f, 8.0f}) {
        Fixture f;
        f.poke(128, payload);
        broadcast(f.warp(), 0, 128.0f);
        f.sched.set_memory_model(MemoryModel::Coalesced);

        Program p;
        p.push_back(make_v_ld_global_f32(/*dst=*/0, /*addr_reg=*/0));
        p.push_back(make_ret());

        ASSERT_NO_THROW(f.run(p)) << "payload " << payload;
        EXPECT_FLOAT_EQ(f.lane(0).regs[0], payload);

        // One line, whatever arrived in the register: every lane read address 128.
        const uint64_t retire = WARP_SIZE * instruction_cost(Opcode::RET);
        EXPECT_EQ(f.sched.stats().weighted_lane_ops - retire,
                  instruction_cost(Opcode::V_LD_GLOBAL_F32))
            << "payload " << payload;
    }
}

TEST(Scheduler, AVectorLoadIsOneTransactionWhereThreeScalarsAreThree)
{
    // What the opcode exists to show, and why it waited for MemoryModel. Every
    // lane reads the same twelve bytes, which is one line: the wide load asks for
    // it once and three scalar loads ask three times.
    // Measured against a program that only retires, so the comparison is of the
    // loads alone: a RET costs every lane and would otherwise sit in both totals.
    Program bare;
    bare.push_back(make_ret());

    auto measure = [&bare](const Program& p, MemoryModel model) {
        auto weighted = [model](const Program& prog) {
            Fixture f;
            broadcast(f.warp(), 0, 16.0f);
            f.sched.set_memory_model(model);
            f.run(prog);
            return f.sched.stats();
        };
        SchedulerStats stats = weighted(p);
        stats.weighted_lane_ops -= weighted(bare).weighted_lane_ops;
        return stats;
    };

    Program wide;
    wide.push_back(make_v_ld_global_vec3_f32(10, 0));
    wide.push_back(make_ret());

    Program scalars;
    scalars.push_back(make_v_ld_global_f32(10, 0, 0.0f));
    scalars.push_back(make_v_ld_global_f32(11, 0, 4.0f));
    scalars.push_back(make_v_ld_global_f32(12, 0, 8.0f));
    scalars.push_back(make_ret());

    const SchedulerStats wide_coalesced = measure(wide, MemoryModel::Coalesced);
    const SchedulerStats scalar_coalesced = measure(scalars, MemoryModel::Coalesced);
    EXPECT_EQ(wide_coalesced.weighted_lane_ops * 3, scalar_coalesced.weighted_lane_ops);

    // And the reason it could not have been measured before: charged per lane per
    // float, the two are the same number. The flat model cannot tell a
    // transaction from a byte.
    const SchedulerStats wide_flat = measure(wide, MemoryModel::Flat);
    const SchedulerStats scalar_flat = measure(scalars, MemoryModel::Flat);
    EXPECT_EQ(wide_flat.weighted_lane_ops, scalar_flat.weighted_lane_ops);

    // The saving in issues is visible to either model, one instruction against
    // three, and is what the latency test below turns into cycles.
    EXPECT_LT(wide_flat.warp_steps, scalar_flat.warp_steps);
}

TEST(Scheduler, AVectorLoadStraddlingALineIsChargedForBoth)
{
    // The count is of lines touched, not of instructions issued, so a vector that
    // crosses a boundary costs what it costs. Twelve bytes from the last eight of
    // a line reach into the next.
    auto load_cost = [](float address) {
        Fixture f;
        broadcast(f.warp(), 0, address);
        f.sched.set_memory_model(MemoryModel::Coalesced);

        Program p;
        p.push_back(make_v_ld_global_vec3_f32(10, 0));
        p.push_back(make_ret());
        f.run(p);

        Fixture bare;
        Program only_ret;
        only_ret.push_back(make_ret());
        bare.sched.set_memory_model(MemoryModel::Coalesced);
        bare.run(only_ret);

        return f.sched.stats().weighted_lane_ops - bare.sched.stats().weighted_lane_ops;
    };

    EXPECT_EQ(load_cost(static_cast<float>(CACHE_LINE_BYTES - 8)),
              2 * instruction_cost(Opcode::V_LD_GLOBAL_F32));
    EXPECT_EQ(load_cost(static_cast<float>(CACHE_LINE_BYTES)),
              instruction_cost(Opcode::V_LD_GLOBAL_F32));
}

TEST(Scheduler, AVectorLoadWaitsOnceWhereThreeScalarsWaitThreeTimes)
{
    // In-order issue with no scoreboard: a warp that has issued a load cannot
    // issue again until the result lands. Three loads are three waits, and one
    // warp has nobody to hide behind.
    auto cycles = [](const Program& p) {
        ThreadBlock block = make_block(1);
        std::vector<uint8_t> memory(GLOBAL_BYTES, 0);
        WarpScheduler scheduler;
        scheduler.set_memory_model(MemoryModel::Cached);
        scheduler.set_latency_model(LatencyModel::Modelled);
        scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()});
        return scheduler.stats().cycles;
    };

    Program wide;
    wide.push_back(make_v_mov_f32(0, 16.0f));
    wide.push_back(make_v_ld_global_vec3_f32(10, 0));
    wide.push_back(make_v_add_f32(20, 10, 12));  // waits on the load
    wide.push_back(make_ret());

    Program scalars;
    scalars.push_back(make_v_mov_f32(0, 16.0f));
    scalars.push_back(make_v_ld_global_f32(10, 0, 0.0f));
    scalars.push_back(make_v_ld_global_f32(11, 0, 4.0f));
    scalars.push_back(make_v_ld_global_f32(12, 0, 8.0f));
    scalars.push_back(make_v_add_f32(20, 10, 12));
    scalars.push_back(make_ret());

    // The first trip goes to memory either way; what the scalars add is two more
    // waits for lines already in L1.
    EXPECT_LT(cycles(wide), cycles(scalars));
}

TEST(Scheduler, MoreSMsFinishSoonerWithoutDoingLessWork)
{
    // The whole claim of several SMs, and the accounting check that goes with it:
    // the same instructions are issued whatever the machine, and only the clock
    // moves. A cycle count that fell without the work following it would mean
    // blocks were being dropped rather than overlapped.
    Program p;
    p.push_back(make_v_mov_f32(0, 1.0f));
    p.push_back(make_v_add_f32(1, 0, 0));
    p.push_back(make_v_mul_f32(2, 1, 1));
    p.push_back(make_ret());

    const auto measure = [&p](uint32_t sm_count) {
        MyGPURuntime rt(1u << 16, 1024);
        SMConfig config;
        config.sm_count = sm_count;
        rt.myrt_set_sm_config(config);
        rt.myrt_launch(constant_kernel(p), dim3{8, 1, 1}, dim3{32, 1, 1}, nullptr);
        return rt.stats();
    };

    const SchedulerStats one = measure(1);
    const SchedulerStats four = measure(4);

    EXPECT_EQ(one.warp_steps, four.warp_steps);
    EXPECT_EQ(one.active_lane_ops, four.active_lane_ops);
    EXPECT_EQ(one.weighted_lane_ops, four.weighted_lane_ops);
    EXPECT_LT(four.cycles, one.cycles) << "four SMs took as long as one";

    // Eight blocks over four SMs, so the clock should fall by about the factor
    // the machine widened by — not exactly, the last blocks arriving as slots
    // free rather than all at once.
    EXPECT_LE(four.cycles * 2, one.cycles + one.cycles / 4);
}

TEST(Scheduler, OneBlockCannotUseASecondSM)
{
    // The bound on the claim above. Blocks are what an SM takes, so a launch with
    // one of them runs the same however wide the machine is — and the idle SMs
    // show up as wasted issue slots rather than as nothing at all.
    Program p;
    p.push_back(make_v_mov_f32(0, 1.0f));
    p.push_back(make_v_add_f32(1, 0, 0));
    p.push_back(make_ret());

    const auto measure = [&p](uint32_t sm_count) {
        MyGPURuntime rt(1u << 16, 1024);
        SMConfig config;
        config.sm_count = sm_count;
        rt.myrt_set_sm_config(config);
        rt.myrt_launch(constant_kernel(p), dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);
        return rt.stats();
    };

    const SchedulerStats one = measure(1);
    const SchedulerStats four = measure(4);

    EXPECT_EQ(one.cycles, four.cycles);
    EXPECT_EQ(one.weighted_lane_ops, four.weighted_lane_ops);

    // And nothing is recorded as stalling, which is the distinction worth
    // keeping: stall_steps counts an SM that holds work and cannot issue it. An
    // SM with no block is not waiting on anything — it was never given any. What
    // that costs is visible as cycles x SMs against the instructions issued, and
    // needs no counter of its own.
    EXPECT_EQ(four.stall_steps, one.stall_steps);
}

TEST(Scheduler, SharedMemoryAndWarpCountAreWhatLimitResidency)
{
    // The arithmetic an occupancy table is made of, and the reason a kernel that
    // stages through shared memory fits fewer blocks on an SM than one that does
    // not. Which of the three limits binds is rarely obvious, which is why it is
    // one function rather than three call sites.
    GPUSpec spec;
    spec.sms.blocks_per_sm = 8;
    spec.sms.warp_slots_per_sm = 64;
    spec.sms.shared_bytes_per_sm = 16 * 1024;

    EXPECT_EQ(spec.residency(1, 0), 8u) << "the block limit binds";
    EXPECT_EQ(spec.residency(16, 0), 4u) << "warp slots bind";
    EXPECT_EQ(spec.residency(1, 4 * 1024), 4u) << "shared memory binds";
    EXPECT_EQ(spec.residency(16, 2 * 1024), 4u) << "two bind at once";

    // A block too large for any limit still runs, alone. Reporting a machine with
    // no room for it would be an error nobody could act on.
    EXPECT_EQ(spec.residency(128, 0), 1u);
    EXPECT_EQ(spec.residency(1, 1024 * 1024), 1u);
}

TEST(Scheduler, BlocksSharingAnSMShareItsL1)
{
    // L1 belongs to an SM, so two blocks on one SM read each other's lines — the
    // reason it stopped being emptied between blocks. Two blocks reading the same
    // address fetch it once between them.
    Program p;
    p.push_back(make_v_mov_f32(0, 0.0f));
    p.push_back(make_v_ld_global_f32(1, 0, 0.0f));
    p.push_back(make_ret());

    const auto misses = [&p](uint32_t blocks_per_sm) {
        MyGPURuntime rt(1u << 16, 1024);
        SMConfig config;
        config.blocks_per_sm = blocks_per_sm;
        rt.myrt_set_sm_config(config);
        rt.myrt_set_memory_model(MemoryModel::Cached);

        // Two launches, so that L2 cannot answer for the second: it outlives a
        // launch, and what is being asked about here is L1.
        rt.myrt_launch(constant_kernel(p), dim3{2, 1, 1}, dim3{32, 1, 1}, nullptr);
        return rt.stats().l1_hits;
    };

    // One block at a time, the second still finds the line — L1 is no longer
    // emptied between them, which is what a shared cache means.
    EXPECT_GT(misses(1), 0u);
    EXPECT_EQ(misses(1), misses(2))
        << "co-residency changed what a shared cache holds, which it should not "
           "for two blocks reading one address";
}

TEST(Scheduler, TheCycleBudgetCoversTimeSpentWaiting)
{
    // The budget counts cycles rather than issues, so a block that only ever waits
    // still reaches it. Bounding issues alone would leave such a block spinning
    // with the counter untouched — and the deadlock test above depends on the
    // bound being reachable.
    ThreadBlock block = make_block(1);
    const Program p = producer_consumer_program();
    seed_lane_ids(block);

    WarpScheduler scheduler;
    scheduler.set_latency_model(LatencyModel::Modelled);
    scheduler.set_cycle_budget(10000);

    std::vector<uint8_t> memory(16, 0);
    EXPECT_THROW(scheduler.run(p, block, DeviceSpan{memory.data(), memory.size()}),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Asynchronous copy — global to shared without the register file, and without
// the warp waiting. What is new is not where the bytes go but when the warp is
// told they arrived.
// ---------------------------------------------------------------------------

TEST(Scheduler, AnAsynchronousCopyDeliversWhatASynchronousOneWould)
{
    Fixture f;
    f.poke(64, 7.5f);

    // r1 = 64 (global source), r2 = 0 (shared destination)
    f.run(Program{
        make_v_mov_f32(1, 64.0f),
        make_v_mov_f32(2, 0.0f),
        make_v_cp_async_shared_global_f32(2, 1),
        make_s_cp_async_wait(0),
        make_v_ld_shared_f32(3, 2),
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[3], 7.5f);
    EXPECT_FLOAT_EQ(f.block.shared_mem[0], 7.5f);
}

TEST(Scheduler, ReadingACopyBeforeWaitingIsRefused)
{
    // Hardware hands back whatever was in shared memory, so a kernel that forgot
    // the wait passes its tests by luck. This says so instead.
    Fixture f;
    f.poke(64, 1.0f);

    EXPECT_THROW(f.run(Program{
                     make_v_mov_f32(1, 64.0f),
                     make_v_mov_f32(2, 0.0f),
                     make_v_cp_async_shared_global_f32(2, 1),
                     make_v_ld_shared_f32(3, 2),
                     make_ret(),
                 }),
                 std::runtime_error);
}

TEST(Scheduler, ACopyStillInFlightDoesNotBlockTheOtherHalfOfTheBuffer)
{
    // The refusal is by destination range, which is what makes double buffering
    // expressible: the tile being filled is out of bounds, the tile being read
    // is not.
    Fixture f;
    f.poke(64, 3.0f);
    f.block.shared_mem[64] = 9.0f;  // byte 256, filled by an earlier pass

    f.sched.set_latency_model(LatencyModel::Modelled);
    f.run(Program{
        make_v_mov_f32(1, 64.0f),
        make_v_mov_f32(2, 0.0f),
        make_v_cp_async_shared_global_f32(2, 1),  // fills byte 0
        make_v_mov_f32(4, 256.0f),
        make_v_ld_shared_f32(5, 4),  // reads byte 256, which is not in flight
        make_s_cp_async_wait(0),
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[5], 9.0f);
}

TEST(Scheduler, TheWarpDoesNotWaitForACopyUntilItAsksTo)
{
    // The measurement the instruction exists for. Same bytes moved either way;
    // what differs is how many cycles the warp spent not issuing.
    const auto cycles = [](bool asynchronous) {
        Fixture f;
        f.sched.set_latency_model(LatencyModel::Modelled);
        f.sched.set_memory_model(MemoryModel::Coalesced);
        for (uint32_t i = 0; i < 4; ++i) {
            f.poke(64 + i * CACHE_LINE_BYTES, static_cast<float>(i));
        }

        Program prog{make_v_mov_f32(1, 64.0f), make_v_mov_f32(2, 0.0f)};
        for (uint32_t i = 0; i < 4; ++i) {
            const float from = static_cast<float>(i * CACHE_LINE_BYTES);
            const float to = static_cast<float>(i * sizeof(float));
            if (asynchronous) {
                prog.push_back(make_v_mov_f32(3, to));
                prog.push_back(make_v_cp_async_shared_global_f32(3, 1, from));
            } else {
                prog.push_back(make_v_ld_global_f32(4, 1, from));
                prog.push_back(make_v_mov_f32(3, to));
                prog.push_back(make_v_st_shared_f32(3, 4));
            }
        }
        if (asynchronous) {
            prog.push_back(make_s_cp_async_wait(0));
        }
        prog.push_back(make_ret());
        f.run(prog);

        // Whatever the route, the four floats have to be there.
        for (uint32_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(f.block.shared_mem[i], static_cast<float>(i))
                << "float " << i;
        }
        return f.sched.stats().cycles;
    };

    const uint64_t synchronous = cycles(false);
    const uint64_t asynchronous = cycles(true);
    EXPECT_LT(asynchronous, synchronous)
        << "four loads waited on one at a time against four issued and awaited once";
}

TEST(Scheduler, AFullCopyQueueStallsTheWarpUntilTheOldestLands)
{
    // Hardware has a queue here and stalls when it is full. A kernel that issues
    // copies in a loop and never waits is asking for unbounded storage, and this
    // is where the request is answered.
    Fixture f;
    f.sched.set_latency_model(LatencyModel::Modelled);

    Program prog{make_v_mov_f32(1, 64.0f), make_v_mov_f32(2, 0.0f)};
    const uint32_t issued = CP_ASYNC_QUEUE_DEPTH + 4;
    for (uint32_t i = 0; i < issued; ++i) {
        prog.push_back(make_v_mov_f32(3, static_cast<float>(i * sizeof(float))));
        prog.push_back(make_v_cp_async_shared_global_f32(3, 1));
    }
    prog.push_back(make_s_cp_async_wait(0));
    prog.push_back(make_ret());
    f.run(prog);

    // The queue is a bound on what is outstanding, not on what may be issued:
    // everything went out, and the ones past the depth waited their turn.
    EXPECT_GT(f.sched.stats().cycles, issued);
    EXPECT_EQ(f.warp().copies_in_flight, 0u) << "the wait drains the queue";
}

TEST(Scheduler, WaitingForAllButOneLeavesTheMostRecentOutstanding)
{
    // The form double buffering is written in: work on what has landed while the
    // next one is still on its way.
    Fixture f;
    f.sched.set_latency_model(LatencyModel::Modelled);
    f.poke(64, 2.0f);
    f.poke(128, 4.0f);

    f.run(Program{
        make_v_mov_f32(1, 64.0f),
        make_v_mov_f32(2, 0.0f),
        make_v_cp_async_shared_global_f32(2, 1),  // first, into byte 0
        make_v_mov_f32(3, 4.0f),
        make_v_mov_f32(4, 128.0f),
        make_v_cp_async_shared_global_f32(3, 4),  // second, into byte 4
        make_s_cp_async_wait(1),                  // only the first has to have landed
        make_v_ld_shared_f32(5, 2),               // legal: the first landed
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.lane(0).regs[5], 2.0f);
    EXPECT_EQ(f.warp().copies_in_flight, 1u)
        << "a copy leaves the queue when it is waited for, and the second was not";
}

TEST(Scheduler, AnAsynchronousCopyCostsWhatALoadCostsAndNoStore)
{
    // One trip to memory, priced by the lines it touches like any global access,
    // and nothing for the shared store it replaces.
    const auto weighted = [](bool asynchronous) {
        Fixture f;
        f.sched.set_memory_model(MemoryModel::Coalesced);
        Program prog{make_v_mov_f32(1, 64.0f), make_v_mov_f32(2, 0.0f)};
        if (asynchronous) {
            prog.push_back(make_v_cp_async_shared_global_f32(2, 1));
            prog.push_back(make_s_cp_async_wait(0));
        } else {
            prog.push_back(make_v_ld_global_f32(4, 1));
            prog.push_back(make_v_st_shared_f32(2, 4));
        }
        prog.push_back(make_ret());
        f.run(prog);
        return f.sched.stats().weighted_lane_ops;
    };

    // The shared store is 8 a lane over 32 lanes; the wait is 1 a lane.
    EXPECT_EQ(weighted(false) - weighted(true), (8 - 1) * WARP_SIZE);
}

// ---------------------------------------------------------------------------
// Atomics — the one instruction whose answer depends on what the other lanes
// are doing at the same instant
// ---------------------------------------------------------------------------

TEST(Scheduler, EveryLanesAdditionLands)
{
    Fixture f;
    f.poke(0, 0.0f);

    // 32 lanes, each adding one to the same counter.
    f.run(Program{
        make_v_mov_f32(1, 0.0f),
        make_v_mov_f32(2, 1.0f),
        make_v_atom_add_global_f32(3, 1, 2),
        make_ret(),
    });

    EXPECT_FLOAT_EQ(f.peek(0), 32.0f) << "a lost update would leave fewer";
}

TEST(Scheduler, EachLaneIsHandedADifferentSlot)
{
    // What makes an atomic more than a combine, and the whole of how a
    // compaction pass gives every surviving item somewhere to go.
    Fixture f;
    f.poke(0, 0.0f);

    f.run(Program{
        make_v_mov_f32(1, 0.0f),
        make_v_mov_f32(2, 1.0f),
        make_v_atom_add_global_f32(3, 1, 2),
        make_ret(),
    });

    std::vector<bool> seen(WARP_SIZE, false);
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        const float slot = f.lane(lane).regs[3];
        ASSERT_GE(slot, 0.0f);
        ASSERT_LT(slot, static_cast<float>(WARP_SIZE));
        const auto index = static_cast<size_t>(slot);
        EXPECT_FALSE(seen[index]) << "two lanes were handed slot " << index;
        seen[index] = true;
    }
}

TEST(Scheduler, CollidingLanesWaitForEachOtherAndScatteredOnesDoNot)
{
    // The measurement the instruction exists to make: an atomic is performed
    // where the caches meet, and two lanes on one address cannot be served at
    // once. Thirty-two on one address is thirty-two deep.
    const auto cycles = [](bool collide) {
        Fixture f;
        f.sched.set_latency_model(LatencyModel::Modelled);
        f.sched.set_memory_model(MemoryModel::Coalesced);

        // Either every lane at byte 0, or each at its own float.
        Program prog{make_v_mov_f32(1, 4.0f), make_v_mul_f32(2, REG_GLOBAL_ID_X, 1),
                     make_v_mov_f32(3, 0.0f), make_v_mov_f32(4, 1.0f)};
        prog.push_back(make_v_atom_add_global_f32(5, collide ? 3 : 2, 4));
        prog.push_back(make_ret());
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            f.lane(lane).regs[REG_GLOBAL_ID_X] = static_cast<float>(lane);
        }
        f.run(prog);
        return f.sched.stats().cycles;
    };

    EXPECT_GT(cycles(true), cycles(false))
        << "colliding lanes are worked through one at a time";
}

TEST(Scheduler, CoalescingCannotHelpAnAtomic)
{
    // A load of one address by 32 lanes is one transaction. An atomic is 32
    // operations however they are laid out, and the memory model does not get a
    // say — which is the arithmetic behind preferring a reduction.
    const auto weighted = [](MemoryModel model) {
        Fixture f;
        f.sched.set_memory_model(model);
        f.run(Program{
            make_v_mov_f32(1, 0.0f),
            make_v_mov_f32(2, 1.0f),
            make_v_atom_add_global_f32(3, 1, 2),
            make_ret(),
        });
        return f.sched.stats().weighted_lane_ops;
    };

    EXPECT_EQ(weighted(MemoryModel::Flat), weighted(MemoryModel::Coalesced));
    EXPECT_EQ(weighted(MemoryModel::Flat), weighted(MemoryModel::Cached));
}
