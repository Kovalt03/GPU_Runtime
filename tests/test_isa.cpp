#include <cstdint>
#include <stdexcept>
#include <string_view>

#include <gtest/gtest.h>

#include "isa.hpp"

namespace {

// The exhaustive-iteration tests rest on Opcode::RET being last.
constexpr int OPCODE_COUNT = static_cast<int>(Opcode::RET) + 1;

// Control flow sits at the end of the enum. The naming-scheme test splits
// the opcodes into these two groups.
bool is_control_flow(Opcode op)
{
    return op == Opcode::BRA || op == Opcode::BRA_DIV || op == Opcode::RET;
}

bool starts_with(std::string_view s, std::string_view prefix)
{
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

bool has_type_suffix(std::string_view name)
{
    // <TYPE> always comes last. F64/F16 are still reserved but belong to the rule.
    return ends_with(name, "_F32") || ends_with(name, "_F64") || ends_with(name, "_F16");
}

bool has_valid_shape_token(std::string_view name)
{
    return name.find("_VEC3_") != std::string_view::npos ||
           name.find("_VEC4_") != std::string_view::npos ||
           name.find("_MAT3_") != std::string_view::npos ||
           name.find("_MAT4_") != std::string_view::npos;
}

bool has_space_token(std::string_view name)
{
    return name.find("_GLOBAL") != std::string_view::npos ||
           name.find("_SHARED") != std::string_view::npos ||
           name.find("_CONST") != std::string_view::npos ||
           name.find("_LOCAL") != std::string_view::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// Structural contracts — if these break, every other layer silently misbehaves.
// ---------------------------------------------------------------------------

TEST(Isa, InstructionSize)
{
    // The 8-byte size is part of the spec. Adding a field or reordering one
    // introduces padding, which trips here first.
    EXPECT_EQ(sizeof(Instruction), 8u);
    EXPECT_EQ(alignof(Instruction), alignof(float));
}

TEST(Isa, OpcodeCount)
{
    // 24 opcodes, 0-indexed → RET == 23
    EXPECT_EQ(static_cast<int>(Opcode::RET), 23);
    EXPECT_EQ(OPCODE_COUNT, 24);

    // Enum values are never serialized, so a change here is not itself a problem.
    // Pinning the category boundaries is a tripwire: it makes it visible when an
    // opcode is inserted mid-list and shifts everything behind it.
    EXPECT_EQ(static_cast<int>(Opcode::V_MUL_F32), 0);
    EXPECT_EQ(static_cast<int>(Opcode::V_ADD_VEC3_F32), 9);
    EXPECT_EQ(static_cast<int>(Opcode::V_MATVEC_MAT4_F32), 15);
    EXPECT_EQ(static_cast<int>(Opcode::V_CMP_F32), 16);
    EXPECT_EQ(static_cast<int>(Opcode::V_LD_GLOBAL_F32), 17);
    EXPECT_EQ(static_cast<int>(Opcode::BRA), 21);
}

// ---------------------------------------------------------------------------
// Naming scheme (DOC/01_virtual_isa.md)
//   ALU     V_<OP>[_<SHAPE>]_<TYPE>
//   Memory  V_<LD|ST>_<SPACE>[_<SHAPE>]_<TYPE>
//   Control <OP>
// A rule that lives only in a document drifts as soon as opcodes are added,
// so it is machine-checked here.
// ---------------------------------------------------------------------------

TEST(Isa, OpcodeNamesFollowScheme)
{
    for (int i = 0; i < OPCODE_COUNT; ++i) {
        const Opcode op = static_cast<Opcode>(i);
        const std::string_view name = opcode_name(op);

        if (is_control_flow(op)) {
            // Control flow writes no register and only mutates warp state
            // → no V_ prefix, no type suffix.
            EXPECT_FALSE(starts_with(name, "V_"))
                << name << ": control flow must not carry the V_ prefix";
            EXPECT_FALSE(has_type_suffix(name))
                << name << ": control flow must not carry a type suffix";
        } else {
            EXPECT_TRUE(starts_with(name, "V_"))
                << name << ": lane-wise instruction must start with V_";
            EXPECT_TRUE(has_type_suffix(name))
                << name << ": type suffix must come last (_F32 / _F64 / _F16)";
        }
    }
}

TEST(Isa, ShapeTokensAreFromTheAllowedSet)
{
    // If VEC/MAT appears at all, it must be one of the defined shape tokens.
    // This keeps ad-hoc spellings such as VADD3 from creeping back in.
    for (int i = 0; i < OPCODE_COUNT; ++i) {
        const std::string_view name = opcode_name(static_cast<Opcode>(i));
        const bool mentions_shape = name.find("VEC") != std::string_view::npos ||
                                    name.find("MAT") != std::string_view::npos;
        if (mentions_shape) {
            EXPECT_TRUE(has_valid_shape_token(name))
                << name << ": shape token must be one of VEC3/VEC4/MAT3/MAT4";
        }
    }
}

TEST(Isa, MemoryOpcodesCarryAddressSpace)
{
    const Opcode mem_ops[] = {Opcode::V_LD_GLOBAL_F32, Opcode::V_ST_GLOBAL_F32,
                              Opcode::V_LD_SHARED_F32, Opcode::V_ST_SHARED_F32};

    for (const Opcode op : mem_ops) {
        const std::string_view name = opcode_name(op);
        EXPECT_TRUE(starts_with(name, "V_LD_") || starts_with(name, "V_ST_"))
            << name << ": memory opcode must start with V_LD_ or V_ST_";
        EXPECT_TRUE(has_space_token(name))
            << name << ": memory opcode must name its address space";
    }
}

TEST(Isa, OpcodeNameCoversAll)
{
    for (int i = 0; i < OPCODE_COUNT; ++i) {
        const std::string_view name = opcode_name(static_cast<Opcode>(i));
        EXPECT_FALSE(name.empty()) << "opcode " << i << " has empty name";
        EXPECT_NE(name, "UNKNOWN") << "opcode " << i << " missing in opcode_name()";
    }
}

TEST(Isa, OpcodeNameRejectsInvalid)
{
    // A value cast in from outside the enum range (e.g. a byte read from a file).
    EXPECT_EQ(opcode_name(static_cast<Opcode>(99)), "UNKNOWN");
}

TEST(Isa, OpcodeNamesAreUnique)
{
    for (int i = 0; i < OPCODE_COUNT; ++i) {
        for (int j = i + 1; j < OPCODE_COUNT; ++j) {
            EXPECT_NE(opcode_name(static_cast<Opcode>(i)),
                      opcode_name(static_cast<Opcode>(j)))
                << "opcode " << i << " and " << j << " share a name";
        }
    }
}

// ---------------------------------------------------------------------------
// Factories — every operand is uint8_t, so the compiler cannot catch a swapped
// argument order.
// ---------------------------------------------------------------------------

TEST(Isa, FactoryFieldsCorrect)
{
    const Instruction inst = make_v_mul_f32(1, 2, 3);
    EXPECT_EQ(inst.op, Opcode::V_MUL_F32);
    EXPECT_EQ(inst.dst, 1);
    EXPECT_EQ(inst.src0, 2);
    EXPECT_EQ(inst.src1, 3);
    EXPECT_EQ(inst.imm, 0.0f);
}

TEST(Isa, MatrixFactoryFields)
{
    const Instruction inst = make_v_matvec_mat4_f32(0, 4, 20);
    EXPECT_EQ(inst.op, Opcode::V_MATVEC_MAT4_F32);
    EXPECT_EQ(inst.dst, 0);
    EXPECT_EQ(inst.src0, 4);
    EXPECT_EQ(inst.src1, 20);
    EXPECT_EQ(inst.imm, 0.0f);
}

TEST(Isa, MatrixCostsMoreThanScalarArithmetic)
{
    // Sixteen products and twelve sums against a single add.
    EXPECT_GT(instruction_cost(Opcode::V_MATVEC_MAT4_F32),
              instruction_cost(Opcode::V_ADD_F32));
    EXPECT_GT(instruction_cost(Opcode::V_MATVEC_MAT4_F32),
              instruction_cost(Opcode::V_CROSS_VEC3_F32));
}

TEST(Isa, UnaryFactoriesLeaveSrc1Unused)
{
    // V_RCP / V_SQRT / V_NORM_VEC3 leave src1 unused. The executor must never
    // read that field, so guarantee it is pinned to 0.
    EXPECT_EQ(make_v_rcp_f32(4, 5).src1, 0);
    EXPECT_EQ(make_v_sqrt_f32(4, 5).src1, 0);
    EXPECT_EQ(make_v_norm_vec3_f32(4, 5).src1, 0);
}

TEST(Isa, StoreFactoriesUseSrc0AsAddress)
{
    // V_ST_* leaves dst unused: address in src0, value in src1. The argument
    // order is reversed relative to V_LD_*, which is the easiest thing to get wrong.
    const Instruction st =
        make_v_st_global_f32(/*addr_reg=*/4, /*src=*/9, /*offset=*/2.0f);
    EXPECT_EQ(st.op, Opcode::V_ST_GLOBAL_F32);
    EXPECT_EQ(st.dst, 0);
    EXPECT_EQ(st.src0, 4);
    EXPECT_EQ(st.src1, 9);
    EXPECT_EQ(st.imm, 2.0f);

    const Instruction ld = make_v_ld_global_f32(/*dst=*/4, /*addr_reg=*/9);
    EXPECT_EQ(ld.dst, 4);
    EXPECT_EQ(ld.src0, 9);
    EXPECT_EQ(ld.imm, 0.0f) << "the default offset must be 0";
}

TEST(Isa, VmovImm)
{
    // No arithmetic is involved — the value is carried through verbatim, so the
    // bit patterns are identical and == suffices.
    EXPECT_EQ(make_v_mov_f32(7, 3.14f).imm, 3.14f);
    EXPECT_EQ(make_v_mov_f32(7, 3.14f).dst, 7);
    EXPECT_EQ(make_v_mov_f32(7, -0.0f).imm, -0.0f);
}

// ---------------------------------------------------------------------------
// imm encoding — if encoder and decoder drift, the kernel executes LT as GE.
// By the time the rendered image looks wrong the cause is hard to trace, so it
// is caught here instead.
// ---------------------------------------------------------------------------

TEST(Isa, VcmpImmEncoding)
{
    const CmpOp all[] = {CmpOp::LT,  CmpOp::GT, CmpOp::EQ,
                         CmpOp::NEQ, CmpOp::LE, CmpOp::GE};

    for (const CmpOp op : all) {
        const Instruction inst = make_v_cmp_f32(38, 24, 35, op);
        EXPECT_EQ(inst.op, Opcode::V_CMP_F32);
        EXPECT_EQ(inst.dst, 38);
        EXPECT_EQ(inst.src0, 24);
        EXPECT_EQ(inst.src1, 35);
        EXPECT_EQ(decode_cmp_op(inst.imm), op)
            << "roundtrip failed for CmpOp " << static_cast<uint32_t>(op);
    }
}

TEST(Isa, BranchOffsetEncoding)
{
    // Branch offsets are a value conversion, not a bit reinterpretation
    // (pc += (int32_t)imm).
    const int32_t offsets[] = {0, 1, -1, 7, -7, 4096, -4096, 1 << 24, -(1 << 24)};

    for (const int32_t off : offsets) {
        EXPECT_EQ(decode_branch_offset(make_bra(off).imm), off)
            << "BRA roundtrip failed for offset " << off;
        EXPECT_EQ(decode_branch_offset(make_bra_div(38, off).imm), off)
            << "BRA_DIV roundtrip failed for offset " << off;
    }

    // BRA_DIV carries its condition register in src0.
    EXPECT_EQ(make_bra_div(38, 12).src0, 38);
    EXPECT_EQ(make_bra(12).src0, 0) << "BRA uses no condition register";
}

TEST(Isa, BranchOffsetOutOfRangeThrows)
{
    // The float significand is 24 bits, so the roundtrip breaks past this range.
    // Fail loudly instead of silently corrupting the PC.
    EXPECT_THROW(make_bra(1 << 25), std::runtime_error);
    EXPECT_THROW(make_bra(-(1 << 25)), std::runtime_error);
    EXPECT_THROW(make_bra_div(0, 1 << 25), std::runtime_error);
    EXPECT_NO_THROW(make_bra(1 << 24));
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

TEST(Isa, ProgramIsVector)
{
    Program prog;
    EXPECT_TRUE(prog.empty());

    prog.push_back(make_v_mov_f32(35, 1e-6f));
    prog.push_back(make_v_sub_vec3_f32(15, 9, 6));
    prog.push_back(make_ret());

    ASSERT_EQ(prog.size(), 3u);  // ASSERT: the lines below index into prog
    EXPECT_EQ(prog[0].op, Opcode::V_MOV_F32);
    EXPECT_EQ(prog[1].op, Opcode::V_SUB_VEC3_F32);
    EXPECT_EQ(prog[2].op, Opcode::RET);
}

TEST(Isa, ProgramIsContiguous)
{
    // Program is a std::vector, so instructions are laid out contiguously.
    // That is why sequential pc access is cache-friendly, and why a size of
    // 8 bytes * N can be assumed.
    Program prog{make_ret(), make_ret(), make_ret()};
    const auto* base = prog.data();
    EXPECT_EQ(&prog[2] - base, 2);
    EXPECT_EQ(
        reinterpret_cast<const char*>(&prog[1]) - reinterpret_cast<const char*>(base), 8);
}
