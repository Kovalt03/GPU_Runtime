#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "isa.hpp"

namespace {

constexpr int32_t BRANCH_OFFSET_LIMIT = 1 << 24;

float encode_branch_offset(int32_t offset)
{
    if (offset > BRANCH_OFFSET_LIMIT || offset < -BRANCH_OFFSET_LIMIT) {
        throw std::runtime_error("branch offset out of encodable range: " +
                                 std::to_string(offset));
    }
    return static_cast<float>(offset);
}

}  // namespace

// The mnemonic table is hand-aligned into columns; clang-format would collapse
// it to a single space and make a missing entry much harder to spot.
// clang-format off
std::string_view opcode_name(Opcode op)
{
    switch (op) {
    // SCALAR
    case Opcode::V_MUL_F32:        return "V_MUL_F32";
    case Opcode::V_ADD_F32:        return "V_ADD_F32";
    case Opcode::V_SUB_F32:        return "V_SUB_F32";
    case Opcode::V_RCP_F32:        return "V_RCP_F32";
    case Opcode::V_SQRT_F32:       return "V_SQRT_F32";
    case Opcode::V_FMA_F32:        return "V_FMA_F32";
    case Opcode::V_MIN_F32:        return "V_MIN_F32";
    case Opcode::V_MAX_F32:        return "V_MAX_F32";
    case Opcode::V_MOV_F32:        return "V_MOV_F32";
    // VECTOR (VEC3)
    case Opcode::V_ADD_VEC3_F32:   return "V_ADD_VEC3_F32";
    case Opcode::V_SUB_VEC3_F32:   return "V_SUB_VEC3_F32";
    case Opcode::V_SCALE_VEC3_F32: return "V_SCALE_VEC3_F32";
    case Opcode::V_DOT_VEC3_F32:   return "V_DOT_VEC3_F32";
    case Opcode::V_CROSS_VEC3_F32: return "V_CROSS_VEC3_F32";
    case Opcode::V_NORM_VEC3_F32:  return "V_NORM_VEC3_F32";
    // MATRIX (MAT4)
    case Opcode::V_MATVEC_MAT4_F32: return "V_MATVEC_MAT4_F32";
    // CMP
    case Opcode::V_CMP_F32:        return "V_CMP_F32";
    // MEM
    case Opcode::V_LD_GLOBAL_F32:  return "V_LD_GLOBAL_F32";
    case Opcode::V_ST_GLOBAL_F32:  return "V_ST_GLOBAL_F32";
    case Opcode::V_LD_SHARED_F32:  return "V_LD_SHARED_F32";
    case Opcode::V_ST_SHARED_F32:  return "V_ST_SHARED_F32";
    // CONTROL FLOW
    case Opcode::BRA:              return "BRA";
    case Opcode::BRA_DIV:          return "BRA_DIV";
    // WARP-LEVEL
    case Opcode::S_BALLOT:         return "S_BALLOT";
    case Opcode::S_ANY:            return "S_ANY";
    case Opcode::S_ALL:            return "S_ALL";
    case Opcode::S_SYNCWARP:       return "S_SYNCWARP";
    case Opcode::V_SHUFFLE_F32:    return "V_SHUFFLE_F32";
    // SYNC
    case Opcode::BARRIER:          return "BARRIER";
    case Opcode::RET:              return "RET";
    }

    return "UNKNOWN";
}
// clang-format on

// ---------------------------------------------------------------------------
// SCALAR
// ---------------------------------------------------------------------------

Instruction make_v_mul_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_MUL_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_add_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_ADD_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_sub_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_SUB_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_rcp_f32(uint8_t dst, uint8_t src0)
{
    return {Opcode::V_RCP_F32, dst, src0, 0, 0.0f};
}

Instruction make_v_sqrt_f32(uint8_t dst, uint8_t src0)
{
    return {Opcode::V_SQRT_F32, dst, src0, 0, 0.0f};
}

// dst operand also serves as the accumulator register: reg[dst] += reg[src0] * reg[src1]
Instruction make_v_fma_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_FMA_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_min_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_MIN_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_max_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_MAX_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_mov_f32(uint8_t dst, float imm)
{
    return {Opcode::V_MOV_F32, dst, 0, 0, imm};
}

// ---------------------------------------------------------------------------
// VECTOR (VEC3) — dst/src is the index of the vector's first register
// ---------------------------------------------------------------------------

Instruction make_v_add_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_ADD_VEC3_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_sub_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_SUB_VEC3_F32, dst, src0, src1, 0.0f};
}

// Only src1_scalar is a single register, not a vector.
Instruction make_v_scale_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1_scalar)
{
    return {Opcode::V_SCALE_VEC3_F32, dst, src0, src1_scalar, 0.0f};
}

// Since the result is a scalar, dst is a single register.
Instruction make_v_dot_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_DOT_VEC3_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_cross_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_CROSS_VEC3_F32, dst, src0, src1, 0.0f};
}

Instruction make_v_norm_vec3_f32(uint8_t dst, uint8_t src0)
{
    return {Opcode::V_NORM_VEC3_F32, dst, src0, 0, 0.0f};
}

// ---------------------------------------------------------------------------
// MATRIX (MAT4)
// ---------------------------------------------------------------------------

// src0 names the first of sixteen registers, src1 and dst the first of four.
Instruction make_v_matvec_mat4_f32(uint8_t dst, uint8_t src0, uint8_t src1)
{
    return {Opcode::V_MATVEC_MAT4_F32, dst, src0, src1, 0.0f};
}

// ---------------------------------------------------------------------------
// CMP
// ---------------------------------------------------------------------------

// CmpOp is encoded as the bit pattern of imm (encode_cmp_op / decode_cmp_op pair).
Instruction make_v_cmp_f32(uint8_t dst, uint8_t src0, uint8_t src1, CmpOp op)
{
    return {Opcode::V_CMP_F32, dst, src0, src1, encode_cmp_op(op)};
}

// ---------------------------------------------------------------------------
// MEM
// ---------------------------------------------------------------------------

Instruction make_v_ld_global_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_GLOBAL_F32, dst, addr_reg, 0, offset};
}

// Store instructions leave dst unused. Address goes in src0, value in src1.
Instruction make_v_st_global_f32(uint8_t addr_reg, uint8_t src, float offset)
{
    return {Opcode::V_ST_GLOBAL_F32, 0, addr_reg, src, offset};
}

Instruction make_v_ld_shared_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_SHARED_F32, dst, addr_reg, 0, offset};
}

Instruction make_v_st_shared_f32(uint8_t addr_reg, uint8_t src, float offset)
{
    return {Opcode::V_ST_SHARED_F32, 0, addr_reg, src, offset};
}

// ---------------------------------------------------------------------------
// CONTROL FLOW
// ---------------------------------------------------------------------------

Instruction make_bra(int32_t offset)
{
    return {Opcode::BRA, 0, 0, 0, encode_branch_offset(offset)};
}

// Only threads whose cond_reg is non-zero branch → activeMask split (divergence).
Instruction make_bra_div(uint8_t cond_reg, int32_t offset)
{
    return {Opcode::BRA_DIV, 0, cond_reg, 0, encode_branch_offset(offset)};
}

// ---------------------------------------------------------------------------
// WARP-LEVEL — participants rides in imm (encode_lane_mask / decode_lane_mask)
// ---------------------------------------------------------------------------

Instruction make_s_ballot(uint8_t dst, uint8_t src0, uint32_t participants)
{
    return {Opcode::S_BALLOT, dst, src0, 0, encode_lane_mask(participants)};
}

Instruction make_s_any(uint8_t dst, uint8_t src0, uint32_t participants)
{
    return {Opcode::S_ANY, dst, src0, 0, encode_lane_mask(participants)};
}

Instruction make_s_all(uint8_t dst, uint8_t src0, uint32_t participants)
{
    return {Opcode::S_ALL, dst, src0, 0, encode_lane_mask(participants)};
}

Instruction make_s_syncwarp(uint32_t participants)
{
    return {Opcode::S_SYNCWARP, 0, 0, 0, encode_lane_mask(participants)};
}

Instruction make_v_shuffle_f32(uint8_t dst, uint8_t src0, uint8_t src1,
                               uint32_t participants)
{
    return {Opcode::V_SHUFFLE_F32, dst, src0, src1, encode_lane_mask(participants)};
}

// ---------------------------------------------------------------------------
// SYNC
// ---------------------------------------------------------------------------

Instruction make_barrier()
{
    return {Opcode::BARRIER, 0, 0, 0, 0.0f};
}

Instruction make_ret()
{
    return {Opcode::RET, 0, 0, 0, 0.0f};
}

// Rough relative latencies, normalised so that a plain FP32 add is 1. The
// spread matters more than the exact figures: the point is that a memory
// round-trip dwarfs arithmetic, which is what makes coalescing worth measuring
// once V_LD_GLOBAL_VEC3_F32 exists.
//
// No default label, so -Wswitch flags an opcode added without a cost.
uint32_t instruction_cost(Opcode op)
{
    switch (op) {
    // Plain FP32 ALU: one unit by definition.
    case Opcode::V_MUL_F32:
    case Opcode::V_ADD_F32:
    case Opcode::V_SUB_F32:
    case Opcode::V_FMA_F32:
    case Opcode::V_MIN_F32:
    case Opcode::V_MAX_F32:
    case Opcode::V_MOV_F32:
    case Opcode::V_CMP_F32: return 1;

    // Special function unit: fewer of them, and several cycles deeper.
    case Opcode::V_RCP_F32:
    case Opcode::V_SQRT_F32: return 4;

    // Three lanes' worth of the scalar operation.
    case Opcode::V_ADD_VEC3_F32:
    case Opcode::V_SUB_VEC3_F32:
    case Opcode::V_SCALE_VEC3_F32:
    case Opcode::V_DOT_VEC3_F32: return 3;

    // Six products and three subtractions.
    case Opcode::V_CROSS_VEC3_F32: return 9;

    // A dot product, a square root, and three divides.
    case Opcode::V_NORM_VEC3_F32: return 12;

    // Sixteen products and twelve sums.
    case Opcode::V_MATVEC_MAT4_F32: return 16;

    // On-chip, so tens of cycles rather than hundreds.
    case Opcode::V_LD_SHARED_F32:
    case Opcode::V_ST_SHARED_F32: return 8;

    // Off-chip. The dominant cost in any real kernel.
    case Opcode::V_LD_GLOBAL_F32:
    case Opcode::V_ST_GLOBAL_F32: return 100;

    // Control flow retires in the scheduler rather than an execution unit.
    case Opcode::BRA:
    case Opcode::BRA_DIV:
    case Opcode::RET: return 1;

    // The instruction itself is nothing; the cost of a barrier is the stall
    // while the slowest warp catches up, and that shows as warps not issuing
    // rather than as weight on this line. S_SYNCWARP is the same argument one
    // level down.
    case Opcode::BARRIER:
    case Opcode::S_SYNCWARP: return 1;

    // Placeholders. A ballot reads all 32 lanes and reduces them, and a shuffle
    // permutes across them, so neither is one lane-op whatever it turns out to
    // cost. The figure comes from measuring a warp reduction against the
    // shared-memory one it replaces; until then a plausible number here would be
    // one nothing measured.
    case Opcode::S_BALLOT:
    case Opcode::S_ANY:
    case Opcode::S_ALL:
    case Opcode::V_SHUFFLE_F32: return 1;
    }

    return 1;
}
