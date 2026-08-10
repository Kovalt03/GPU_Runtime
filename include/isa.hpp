#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Opcode naming scheme
//
//   ALU     V_<OP>[_<SHAPE>]_<TYPE>
//   Memory  V_<LD|ST>_<SPACE>[_<SHAPE>]_<TYPE>
//   Control <OP>                            (no prefix, no type)
//
//   V_      lane-wise: writes the per-thread register file. Control flow only
//           touches warp state (pc / activeMask), so it carries no prefix.
//           A warp-uniform unit would use S_ (reserved).
//   SHAPE   omitted = scalar, VEC3 = 3 consecutive regs.
//           Reserved: VEC4, MAT3 (9 regs), MAT4 (16 regs) — VEC4 and wider
//           must start at a 4-aligned register index.
//   SPACE   GLOBAL | SHARED. Reserved: CONST, LOCAL.
//   TYPE    always last. F32 today; F64 / F16 reserved.
//
// F32 rather than FP32: f32 is the mnemonic standard (PTX add.f32,
// AMD v_add_f32, WGSL/SPIR-V/Rust f32). FP32 belongs to spec sheets.
// ---------------------------------------------------------------------------

// Compare condition for V_CMP_F32. Carried in the imm field as a raw bit
// pattern — see encode_cmp_op / decode_cmp_op.
enum class CmpOp : uint32_t { LT = 0, GT = 1, EQ = 2, NEQ = 3, LE = 4, GE = 5 };

enum class Opcode : uint8_t {
    // Scalar ALU
    V_MUL_F32,   // reg[dst] = reg[src0] * reg[src1]
    V_ADD_F32,   // reg[dst] = reg[src0] + reg[src1]
    V_SUB_F32,   // reg[dst] = reg[src0] - reg[src1]
    V_RCP_F32,   // reg[dst] = 1.0f / reg[src0]              (src1 unused)
    V_SQRT_F32,  // reg[dst] = sqrt(reg[src0])               (src1 unused)
    V_FMA_F32,   // reg[dst] += reg[src0] * reg[src1]        (in-place acc)
    V_MIN_F32,   // reg[dst] = min(reg[src0], reg[src1])
    V_MAX_F32,   // reg[dst] = max(reg[src0], reg[src1])
    V_MOV_F32,   // reg[dst] = imm                           (src0/src1 unused)

    // Vector ALU — VEC3. dst/src hold the FIRST register of the vector.
    V_ADD_VEC3_F32,    // reg[dst..+2] = reg[src0..+2] + reg[src1..+2]
    V_SUB_VEC3_F32,    // reg[dst..+2] = reg[src0..+2] - reg[src1..+2]
    V_SCALE_VEC3_F32,  // reg[dst..+2] = reg[src0..+2] * reg[src1] (src1 scalar)
    V_DOT_VEC3_F32,    // reg[dst]     = dot(reg[src0..+2], reg[src1..+2])
    V_CROSS_VEC3_F32,  // reg[dst..+2] = cross(reg[src0..+2], reg[src1..+2])
    V_NORM_VEC3_F32,   // reg[dst..+2] = normalize(reg[src0..+2]) (src1 unused)

    // Matrix ALU — MAT4 is row-major across 16 consecutive registers. dst and
    // src1 are VEC4. All three start at a register index that is a multiple of
    // 4, the alignment the scheme above reserves for VEC4 and wider.
    V_MATVEC_MAT4_F32,  // reg[dst..+3] = mat4(reg[src0..+15]) * vec4(reg[src1..+3])

    // Compare
    V_CMP_F32,  // reg[dst] = (reg[src0] OP reg[src1]) ? 1.0f : 0.0f

    // Memory — address is always src0, stored value is always src1.
    V_LD_GLOBAL_F32,  // reg[dst] = global[reg[src0] + imm]       (src1 unused)
    V_ST_GLOBAL_F32,  // global[reg[src0] + imm] = reg[src1]      (dst unused)
    V_LD_SHARED_F32,  // reg[dst] = shared[reg[src0] + imm]       (src1 unused)
    V_ST_SHARED_F32,  // shared[reg[src0] + imm] = reg[src1]      (dst unused)

    // Control flow — warp state only, hence no V_ prefix.
    BRA,      // pc += (int32_t)imm                       (unconditional)
    BRA_DIV,  // if (reg[src0] != 0.0f) pc += (int32_t)imm
              //   → splits activeMask (divergence point)

    // CUDA's __syncthreads().
    BARRIER,  // wait for every live warp of the block    (all operands unused)
    RET,      // end thread                               (all operands unused)
};

struct Instruction {
    Opcode op;
    uint8_t dst;   // destination register idx (first register for VEC/MAT)
    uint8_t src0;  // source register0
    uint8_t src1;  // source register1
    float imm;     // immediate value / memory offset / CmpOp
};
static_assert(sizeof(Instruction) == 8, "Instruction must stay 8 bytes");

using Program = std::vector<Instruction>;

// Factory names mirror the mnemonic in lower case: V_ADD_VEC3_F32 →
// make_v_add_vec3_f32. Verbose, but adding an opcode never requires
// inventing a name.

// SCALAR
Instruction make_v_mul_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_add_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_sub_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_rcp_f32(uint8_t dst, uint8_t src0);
Instruction make_v_sqrt_f32(uint8_t dst, uint8_t src0);
Instruction make_v_fma_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_min_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_max_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_mov_f32(uint8_t dst, float imm);

// VECTOR (VEC3)
Instruction make_v_add_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_sub_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_scale_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1_scalar);
Instruction make_v_dot_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_cross_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_norm_vec3_f32(uint8_t dst, uint8_t src0);

// MATRIX (MAT4)
Instruction make_v_matvec_mat4_f32(uint8_t dst, uint8_t src0, uint8_t src1);

// CMP
Instruction make_v_cmp_f32(uint8_t dst, uint8_t src0, uint8_t src1, CmpOp op);

// MEM
Instruction make_v_ld_global_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_global_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);
Instruction make_v_ld_shared_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_shared_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);

// CONTROL FLOW
Instruction make_bra(int32_t offset);
Instruction make_bra_div(uint8_t cond_reg, int32_t offset);
Instruction make_barrier();
Instruction make_ret();

// Bit reinterpretation between 'imm' and 'CmpOp'. Centralized here so that the
// encoder (make_v_cmp_f32) and the decoder (Scheduler) cannot drift apart.
inline float encode_cmp_op(CmpOp op)
{
    const uint32_t bits = static_cast<uint32_t>(op);
    float imm;
    std::memcpy(&imm, &bits, sizeof(float));
    return imm;
}

inline CmpOp decode_cmp_op(float imm)
{
    uint32_t bits;
    std::memcpy(&bits, &imm, sizeof(uint32_t));
    return static_cast<CmpOp>(bits);
}

// The branch offset is a value conversion (not a bit reinterpretation),
// following the `pc += (int32_t)imm` definition. Since the float mantissa is
// 24 bits, a round-trip fails once |offset| exceeds 2^24; the make_bra family
// range-checks on the encoding side.
inline int32_t decode_branch_offset(float imm)
{
    return static_cast<int32_t>(imm);
}

// For disassembly/logging purposes only. Not called during the execution path.
std::string_view opcode_name(Opcode op);

// Relative issue cost, where a plain FP32 add is 1. Real hardware spans roughly
// two orders of magnitude between an add and a global load, so counting every
// opcode as one would make a kernel full of square roots look faster than one
// full of adds. Only throughput readings depend on this; divergence is a ratio
// of issued capacity to capacity used and is unaffected.
uint32_t instruction_cost(Opcode op);
