#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "ir_builder.hpp"
#include "runtime.hpp"

namespace {

// The builder's whole job is to emit the right instructions, so most of these
// compare its output against what a hand-written kernel would contain.
bool same(const Instruction& a, const Instruction& b)
{
    return a.op == b.op && a.dst == b.dst && a.src0 == b.src0 && a.src1 == b.src1 &&
           a.imm == b.imm;
}

}  // namespace

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

TEST(IRBuilder, AllocatesConsecutiveRegisters)
{
    IRBuilder k;
    const Reg<Scalar> a = k.scalar();
    const Reg<Scalar> b = k.scalar();
    const Reg<Vec3> v = k.vec3();

    EXPECT_EQ(a.first(), 0);
    EXPECT_EQ(b.first(), 1);
    EXPECT_EQ(v.first(), 2) << "a vec3 claims three, starting where the scalars left off";
    EXPECT_EQ(k.registers_used(), 5u);
}

TEST(IRBuilder, AlignsVec4AndMat4)
{
    // V_MATVEC_MAT4_F32 rejects an unaligned operand, so the allocator has to
    // round up rather than leave that to the caller.
    IRBuilder k;
    k.scalar();  // r0, leaving the cursor at 1
    const Reg<Vec4> v = k.vec4();
    EXPECT_EQ(v.first() % 4, 0u);
    EXPECT_EQ(v.first(), 4);

    k.scalar();  // r8
    const Reg<Mat4> m = k.mat4();
    EXPECT_EQ(m.first() % 4, 0u);
    EXPECT_EQ(m.first(), 12);
}

TEST(IRBuilder, RefusesToHandOutThreadCoordinateRegisters)
{
    // r253..r255 carry the thread's identity. Wrapping onto them would have a
    // kernel read its own id as data, so exhaustion has to be an error.
    IRBuilder k;
    EXPECT_THROW(
        {
            for (int i = 0; i < 300; ++i) {
                k.scalar();
            }
        },
        std::runtime_error);
}

TEST(IRBuilder, ComponentViewsIntoAVector)
{
    IRBuilder k;
    const Reg<Vec4> clip = k.vec4();

    // The perspective divide needs clip.w on its own.
    EXPECT_EQ(clip.component(3).first(), clip.first() + 3);
    EXPECT_EQ(clip.component(0).first(), clip.first());
    EXPECT_THROW(clip.component(4), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Constants — the only route a host value takes into a program
// ---------------------------------------------------------------------------

TEST(IRBuilder, ConstantEmitsAMove)
{
    IRBuilder k;
    const Reg<Scalar> eps = k.constant(1e-6f);

    const Program p = k.build();
    ASSERT_GE(p.size(), 1u);
    EXPECT_TRUE(same(p[0], make_v_mov_f32(eps.first(), 1e-6f)));
}

TEST(IRBuilder, Vec3ConstantEmitsThreeMoves)
{
    IRBuilder k;
    const Reg<Vec3> v = k.constant(1.0f, 2.0f, 3.0f);

    const Program p = k.build();
    ASSERT_GE(p.size(), 3u);
    EXPECT_TRUE(same(p[0], make_v_mov_f32(v.first() + 0, 1.0f)));
    EXPECT_TRUE(same(p[1], make_v_mov_f32(v.first() + 1, 2.0f)));
    EXPECT_TRUE(same(p[2], make_v_mov_f32(v.first() + 2, 3.0f)));
}

TEST(IRBuilder, ThreadCoordinatesAreNotAllocated)
{
    IRBuilder k;
    EXPECT_EQ(k.thread_x().first(), REG_GLOBAL_ID_X);
    EXPECT_EQ(k.thread_y().first(), REG_GLOBAL_ID_Y);
    EXPECT_EQ(k.thread_z().first(), REG_GLOBAL_ID_Z);
    EXPECT_EQ(k.registers_used(), 0u) << "reading a coordinate allocates nothing";
}

// ---------------------------------------------------------------------------
// Arithmetic — overloads pick the opcode, so the caller never can
// ---------------------------------------------------------------------------

TEST(IRBuilder, ScalarArithmeticEmitsScalarOpcodes)
{
    IRBuilder k;
    const Reg<Scalar> a = k.scalar();
    const Reg<Scalar> b = k.scalar();
    const Reg<Scalar> sum = k.add(a, b);

    const Program p = k.build();
    ASSERT_GE(p.size(), 1u);
    EXPECT_TRUE(same(p[0], make_v_add_f32(sum.first(), a.first(), b.first())));
}

TEST(IRBuilder, VectorArithmeticEmitsVectorOpcodes)
{
    // The same call, a different shape, a different opcode — the mistake made
    // three times over while writing kernels/ray_triangle.cpp by hand.
    IRBuilder k;
    const Reg<Vec3> a = k.vec3();
    const Reg<Vec3> b = k.vec3();
    const Reg<Vec3> diff = k.sub(a, b);

    const Program p = k.build();
    ASSERT_GE(p.size(), 1u);
    EXPECT_EQ(p[0].op, Opcode::V_SUB_VEC3_F32);
    EXPECT_TRUE(same(p[0], make_v_sub_vec3_f32(diff.first(), a.first(), b.first())));
}

TEST(IRBuilder, DotReturnsAScalarAndCrossAVector)
{
    IRBuilder k;
    const Reg<Vec3> a = k.vec3();
    const Reg<Vec3> b = k.vec3();

    const Reg<Scalar> d = k.dot(a, b);
    const Reg<Vec3> c = k.cross(a, b);

    // A dot result claims one register and a cross result three, which is what
    // stops the next allocation from overlapping either.
    EXPECT_EQ(c.first(), d.first() + 1);
    EXPECT_EQ(k.registers_used(), 10u);
}

TEST(IRBuilder, TransformEmitsTheMatrixOpcode)
{
    IRBuilder k;
    const Reg<Mat4> m = k.mat4();
    const Reg<Vec4> v = k.vec4();
    const Reg<Vec4> out = k.transform(m, v);

    const Program p = k.build();
    ASSERT_GE(p.size(), 1u);
    EXPECT_TRUE(same(p[0], make_v_matvec_mat4_f32(out.first(), m.first(), v.first())));
}

TEST(IRBuilder, FmaAccumulatesWithoutAllocating)
{
    IRBuilder k;
    const Reg<Scalar> acc = k.scalar();
    const Reg<Scalar> a = k.scalar();
    const Reg<Scalar> b = k.scalar();
    const uint32_t before = k.registers_used();

    k.fma(acc, a, b);

    EXPECT_EQ(k.registers_used(), before) << "V_FMA_F32 writes into dst";
    const Program p = k.build();
    ASSERT_GE(p.size(), 1u);
    EXPECT_TRUE(same(p[0], make_v_fma_f32(acc.first(), a.first(), b.first())));
}

// ---------------------------------------------------------------------------
// Memory — argument order cannot be got wrong
// ---------------------------------------------------------------------------

TEST(IRBuilder, LoadAndStoreUseTheAddressAsSrc0)
{
    IRBuilder k;
    const Reg<Scalar> addr = k.scalar();
    const Reg<Scalar> value = k.load(addr, 4.0f);
    k.store(addr, value, 8.0f);

    const Program p = k.build();
    ASSERT_GE(p.size(), 2u);
    EXPECT_TRUE(same(p[0], make_v_ld_global_f32(value.first(), addr.first(), 4.0f)));
    EXPECT_TRUE(same(p[1], make_v_st_global_f32(addr.first(), value.first(), 8.0f)));
}

TEST(IRBuilder, LoadVec3ReadsThreeConsecutiveFloats)
{
    IRBuilder k;
    const Reg<Scalar> addr = k.scalar();
    const Reg<Vec3> v = k.load_vec3(addr);

    const Program p = k.build();
    ASSERT_GE(p.size(), 3u);
    EXPECT_TRUE(same(p[0], make_v_ld_global_f32(v.first() + 0, addr.first(), 0.0f)));
    EXPECT_TRUE(same(p[1], make_v_ld_global_f32(v.first() + 1, addr.first(), 4.0f)));
    EXPECT_TRUE(same(p[2], make_v_ld_global_f32(v.first() + 2, addr.first(), 8.0f)));
}

// ---------------------------------------------------------------------------
// Control flow — the patching the builder exists to hide
// ---------------------------------------------------------------------------

TEST(IRBuilder, ForwardBranchLandsOnItsLabel)
{
    IRBuilder k;
    const Reg<Scalar> cond = k.scalar();
    const Label done = k.label();

    const size_t branch_at = 0;
    k.branch_to(done, cond);
    k.constant(1.0f);  // skipped when the branch is taken
    k.place(done);

    const Program p = k.build();
    ASSERT_GE(p.size(), 2u);
    EXPECT_EQ(p[branch_at].op, Opcode::BRA_DIV);
    EXPECT_EQ(p[branch_at].src0, cond.first());
    // Two instructions on: past the branch itself and past the constant.
    EXPECT_EQ(decode_branch_offset(p[branch_at].imm), 2);
}

TEST(IRBuilder, BackwardBranchIsANegativeOffset)
{
    // How a loop is written, and where an offset cast through an unsigned type
    // stops working.
    IRBuilder k;
    const Label top = k.label();
    k.place(top);
    const Reg<Scalar> cond = k.constant(1.0f);
    k.branch_to(top, cond);

    const Program p = k.build();
    ASSERT_GE(p.size(), 2u);
    EXPECT_EQ(p[1].op, Opcode::BRA_DIV);
    EXPECT_LT(decode_branch_offset(p[1].imm), 0);
}

TEST(IRBuilder, SeveralBranchesShareOneLabel)
{
    // Möller-Trumbore leaves for its miss path from five separate tests, which
    // is why a label exists alongside if_.
    IRBuilder k;
    const Label miss = k.label();
    const Reg<Scalar> a = k.constant(1.0f);
    k.branch_to(miss, a);
    const Reg<Scalar> b = k.constant(2.0f);
    k.branch_to(miss, b);
    k.place(miss);

    const Program p = k.build();
    int branches = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i].op == Opcode::BRA_DIV) {
            ++branches;
            const int64_t target =
                static_cast<int64_t>(i) + decode_branch_offset(p[i].imm);
            EXPECT_EQ(target, 4) << "both land on the same instruction";
        }
    }
    EXPECT_EQ(branches, 2);
}

TEST(IRBuilder, UnplacedLabelIsAnError)
{
    // Branching to a label nobody placed is a bug in the kernel, not something
    // to resolve to zero and let run.
    IRBuilder k;
    const Label nowhere = k.label();
    const Reg<Scalar> cond = k.constant(1.0f);
    k.branch_to(nowhere, cond);

    EXPECT_THROW(k.build(), std::runtime_error);
}

TEST(IRBuilder, IfRunsItsBodyOnlyWhenTheConditionHolds)
{
    // BRA_DIV jumps when a value is non-zero, but skipping a body means jumping
    // when the condition is false, so the builder inverts it.
    IRBuilder k;
    const Reg<Scalar> cond = k.constant(1.0f);
    k.if_(cond, [&] { k.constant(42.0f); });

    const Program p = k.build();
    bool found = false;
    for (const Instruction& i : p) {
        if (i.op == Opcode::BRA_DIV) {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "if_ has to emit a branch";
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

TEST(IRBuilder, BuildAppendsRet)
{
    IRBuilder k;
    k.constant(1.0f);

    const Program p = k.build();
    ASSERT_FALSE(p.empty());
    EXPECT_EQ(p.back().op, Opcode::RET) << "a program that falls off its end throws";
}

TEST(IRBuilder, EmptyProgramIsStillValid)
{
    IRBuilder k;
    const Program p = k.build();
    ASSERT_EQ(p.size(), 1u);
    EXPECT_EQ(p[0].op, Opcode::RET);
}

// ---------------------------------------------------------------------------
// Execution — everything above compares instructions, which cannot tell whether
// a branch offset is the one the scheduler resolves. Only running the program
// answers that, and the loop below is the shape a kernel walking a triangle
// buffer will take.
// ---------------------------------------------------------------------------

TEST(IRBuilder, GeneratedProgramRunsOnTheScheduler)
{
    KernelFunc kernel = [](void**) -> Program {
        IRBuilder k;

        const Reg<Scalar> base = k.constant(0.0f);  // device byte offset 0
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> limit = k.constant(5.0f);
        const Reg<Scalar> sum = k.constant(0.0f);
        const Reg<Scalar> i = k.constant(0.0f);

        // sum = 0 + 1 + 2 + 3 + 4, over a backward branch.
        const Label top = k.label();
        k.place(top);
        k.fma(sum, i, one);  // sum += i * 1
        k.fma(i, one, one);  // i   += 1
        k.branch_to(top, k.lt(i, limit));
        k.store(base, sum, 0.0f);

        const Reg<Scalar> hundred = k.constant(100.0f);
        k.if_(k.lt(sum, hundred), [&] { k.store(base, k.constant(7.0f), 4.0f); });

        // Takes the else path: nothing is greater than 100 here.
        k.if_else(
            k.gt(base, hundred), [&] { k.store(base, hundred, 8.0f); },
            [&] { k.store(base, k.constant(42.0f), 8.0f); });

        return k.build();
    };

    MyGPURuntime rt(1u << 20);
    void* out = rt.myrt_malloc(3 * sizeof(float));
    rt.myrt_launch(kernel, dim3{1, 1, 1}, dim3{32, 1, 1}, nullptr);

    std::vector<float> host(3, -1.0f);
    rt.myrt_memcpy(host.data(), out, 3 * sizeof(float), Direction::DeviceToHost);

    EXPECT_FLOAT_EQ(host[0], 10.0f) << "the backward branch has to close the loop";
    EXPECT_FLOAT_EQ(host[1], 7.0f) << "if_ runs its body when the condition holds";
    EXPECT_FLOAT_EQ(host[2], 42.0f) << "if_else takes the else path";
}
