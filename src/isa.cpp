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
    case Opcode::V_LD_GLOBAL_VEC3_F32: return "V_LD_GLOBAL_VEC3_F32";
    case Opcode::V_ST_GLOBAL_F32:  return "V_ST_GLOBAL_F32";
    case Opcode::V_LD_SHARED_F32:  return "V_LD_SHARED_F32";
    case Opcode::V_ST_SHARED_F32:  return "V_ST_SHARED_F32";
    case Opcode::V_CP_ASYNC_SHARED_GLOBAL_F32: return "V_CP_ASYNC_SHARED_GLOBAL_F32";
    case Opcode::V_ATOM_ADD_GLOBAL_F32: return "V_ATOM_ADD_GLOBAL_F32";
    case Opcode::V_LD_CONST_F32:   return "V_LD_CONST_F32";
    case Opcode::V_LD_CONST_MAT4_F32: return "V_LD_CONST_MAT4_F32";
    case Opcode::V_LD_SHARED_16X16_F32: return "V_LD_SHARED_16X16_F32";
    case Opcode::V_LD_SHARED_16X16_F16: return "V_LD_SHARED_16X16_F16";
    case Opcode::V_LD_CLUSTER_F32: return "V_LD_CLUSTER_F32";
    // CONTROL FLOW
    case Opcode::BRA:              return "BRA";
    case Opcode::BRA_DIV:          return "BRA_DIV";
    // WARP-LEVEL
    case Opcode::S_BALLOT:         return "S_BALLOT";
    case Opcode::S_ANY:            return "S_ANY";
    case Opcode::S_ALL:            return "S_ALL";
    case Opcode::S_SYNCWARP:       return "S_SYNCWARP";
    case Opcode::S_CP_ASYNC_WAIT:  return "S_CP_ASYNC_WAIT";
    case Opcode::V_SHUFFLE_F32:    return "V_SHUFFLE_F32";
    case Opcode::V_MMA_16X16X16_F32: return "V_MMA_16X16X16_F32";
    case Opcode::V_MMA_16X16X16_F16: return "V_MMA_16X16X16_F16";
    // SYNC
    case Opcode::BARRIER:          return "BARRIER";
    case Opcode::REORDER:          return "REORDER";
    case Opcode::BARRIER_CLUSTER:  return "BARRIER_CLUSTER";
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

Instruction make_v_ld_global_vec3_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_GLOBAL_VEC3_F32, dst, addr_reg, 0, offset};
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

Instruction make_v_atom_add_global_f32(uint8_t dst, uint8_t addr_reg, uint8_t src,
                                       float offset)
{
    return {Opcode::V_ATOM_ADD_GLOBAL_F32, dst, addr_reg, src, offset};
}

// Two addresses and no value, so both operand slots are sources and dst is
// unused — the only memory instruction here whose destination is not a register
// or the value in src1.
//
// Arguments read in the order the mnemonic does, destination first, and the
// encoding puts the global address in src0 because that is the one the memory
// model prices. The two orders differ, so neither is left to be inferred.
Instruction make_v_cp_async_shared_global_f32(uint8_t shared_addr_reg,
                                              uint8_t global_addr_reg, float offset)
{
    return {Opcode::V_CP_ASYNC_SHARED_GLOBAL_F32, 0, global_addr_reg, shared_addr_reg,
            offset};
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

// A count rather than a lane mask, so it goes in imm as the whole number it is.
// Every lane of the warp issued the copies together and they land together;
// there is no set of participants to name.
Instruction make_s_cp_async_wait(uint32_t outstanding)
{
    return {Opcode::S_CP_ASYNC_WAIT, 0, 0, 0, static_cast<float>(outstanding)};
}

Instruction make_v_ld_const_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_CONST_F32, dst, addr_reg, 0, offset};
}

Instruction make_v_ld_const_mat4_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_CONST_MAT4_F32, dst, addr_reg, 0, offset};
}

Instruction make_v_ld_shared_16x16_f32(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_SHARED_16X16_F32, dst, addr_reg, 0, offset};
}

Instruction make_v_ld_shared_16x16_f16(uint8_t dst, uint8_t addr_reg, float offset)
{
    return {Opcode::V_LD_SHARED_16X16_F16, dst, addr_reg, 0, offset};
}

Instruction make_v_ld_cluster_f32(uint8_t dst, uint8_t addr_reg, uint8_t rank_reg,
                                  float offset)
{
    return {Opcode::V_LD_CLUSTER_F32, dst, addr_reg, rank_reg, offset};
}

Instruction make_barrier_cluster()
{
    return {Opcode::BARRIER_CLUSTER, 0, 0, 0, 0.0f};
}

Instruction make_reorder(uint8_t key_reg)
{
    return {Opcode::REORDER, 0, key_reg, 0, 0.0f};
}

Instruction make_v_mma_16x16x16_f32(uint8_t dst, uint8_t src0, uint8_t src1,
                                    uint32_t participants)
{
    return {Opcode::V_MMA_16X16X16_F32, dst, src0, src1, encode_lane_mask(participants)};
}

Instruction make_v_mma_16x16x16_f16(uint8_t dst, uint8_t src0, uint8_t src1,
                                    uint32_t participants)
{
    return {Opcode::V_MMA_16X16X16_F16, dst, src0, src1, encode_lane_mask(participants)};
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

    // 4,096 multiply-adds across 32 lanes is 128 a lane, and this is priced at an
    // eighth of that. The number is not the arithmetic: it is the claim that a
    // matrix unit retires the work about eight times faster than the lanes would
    // one FMA at a time, which is the conservative end of what hardware's tensor
    // cores are quoted at.
    //
    // reduction_bench's method rather than a guess left standing —
    // mma_bench asks how expensive this would have to be before the two routes
    // came level, and the answer is 128.
    case Opcode::V_MMA_16X16X16_F32: return 16;

    // Half again, which is the claim a tensor core makes for a narrow input and
    // the reason f16 is on these units. The accumulator is still single
    // precision, so the depth below is unchanged.
    case Opcode::V_MMA_16X16X16_F16: return 8;

    // Answered once for the warp and broadcast, so this is the cost of the whole
    // instruction rather than of a lane — the one place in this table where that
    // is true, and the reason the space exists. Priced at a shared-memory access
    // because that is where a constant cache sits.
    case Opcode::V_LD_CONST_F32: return 8;

    // Sixteen of them in one, and no discount: the window still hands over
    // sixteen floats. What it saves is the sixteen instructions a baked matrix
    // costs, and that shows in the issue count rather than here.
    case Opcode::V_LD_CONST_MAT4_F32: return 16 * 8;

    // On-chip, so tens of cycles rather than hundreds.
    case Opcode::V_LD_SHARED_F32:
    case Opcode::V_ST_SHARED_F32: return 8;

    // Eight floats, so eight times a float. No discount: shared memory has banks
    // rather than lines, and this machine does not model a bank conflict — so
    // there is no transaction for a wide load to save, and claiming one would be
    // inventing a saving. What it does save is on the other line, in latency.
    case Opcode::V_LD_SHARED_16X16_F32: return 8 * 8;

    // Half the registers, half the bytes, half the price. The saving is real
    // rather than claimed: the instruction moves four registers where the other
    // moves eight.
    case Opcode::V_LD_SHARED_16X16_F16: return 4 * 8;

    // Still on-chip, and still not this block's: the request leaves the SM and
    // comes back. Twice a local one, which puts it an order below a cache line
    // and an order above nothing — the position hardware's distributed shared
    // memory holds between shared and L2.
    case Opcode::V_LD_CLUSTER_F32: return 16;

    // Off-chip. The dominant cost in any real kernel.
    case Opcode::V_LD_GLOBAL_F32:
    case Opcode::V_ST_GLOBAL_F32: return 100;

    // The same trip to memory as a global load, and that is the whole of it: no
    // register is written and no shared store follows, so the pair this replaces
    // cost 108 between them. What it saves in issue capacity is the 8; what it
    // saves in time is the wait, and that is not on this line.
    case Opcode::V_CP_ASYNC_SHARED_GLOBAL_F32: return 100;

    // A read and a write that nothing may come between, performed where the
    // caches meet rather than in a lane. Priced above a load because it is one
    // — what it costs beyond that depends on how many lanes want the same
    // address, and that is atomic_access rather than this line.
    case Opcode::V_ATOM_ADD_GLOBAL_F32: return 120;

    // Three floats, so three times a float. What the wide load saves is
    // transactions rather than bytes, and a charge per lane cannot see it —
    // global_transaction_cost is where the two part company.
    case Opcode::V_LD_GLOBAL_VEC3_F32: return 300;

    // Control flow retires in the scheduler rather than an execution unit.
    case Opcode::BRA:
    case Opcode::BRA_DIV:
    case Opcode::RET: return 1;

    // The instruction itself is nothing; the cost of a barrier is the stall
    // while the slowest warp catches up, and that shows as warps not issuing
    // rather than as weight on this line. S_SYNCWARP is the same argument one
    // level down.
    case Opcode::BARRIER:
    case Opcode::BARRIER_CLUSTER:
    case Opcode::S_SYNCWARP: return 1;

    // Not nothing: every thread of the block is sorted and moved. Priced at a
    // shared-memory round trip a thread, which is the cheapest way the hardware
    // could plausibly do it — the registers themselves do not move on a real
    // machine, the thread's identity does.
    case Opcode::REORDER: return 8;

    // Likewise: waiting for a copy costs the wait, which shows as the warp not
    // issuing. The instruction that expresses the wait is free.
    case Opcode::S_CP_ASYNC_WAIT: return 1;

    // A crossbar across the warp rather than a lane's own arithmetic, and priced
    // like the shared-memory traffic it replaces: hardware runs the exchange
    // through the same permute network, and AMD's ds_bpermute literally borrows
    // the LDS wiring for it.
    //
    // 8, the same as a shared load, is the conservative end of what
    // reduction_bench leaves open. Summing a warp costs 68 instructions through
    // shared memory against 33 through the exchange, so a shuffle could cost 22
    // before the two came level — and that comparison already flatters shared
    // memory, since this machine charges nothing for the barrier's stall.
    case Opcode::S_BALLOT:
    case Opcode::S_ANY:
    case Opcode::S_ALL:
    case Opcode::V_SHUFFLE_F32: return 8;
    }

    return 1;
}

uint32_t instruction_latency(Opcode op)
{
    // Chosen ratios, not measurements — there is no hardware here to time, so this
    // has the provenance instruction_cost has: hardware's orders of magnitude,
    // stated as picked.
    switch (op) {
    // A reciprocal or a square root goes through a special function unit, which
    // real hardware runs at a fraction of the rate and several times the depth.
    case Opcode::V_RCP_F32:
    case Opcode::V_SQRT_F32:
    case Opcode::V_NORM_VEC3_F32: return 16;

    // Shared memory is on-chip and an order below a cache hit.
    case Opcode::V_LD_SHARED_F32: return 30;

    // The constant cache sits beside it and answers as quickly.
    case Opcode::V_LD_CONST_F32:
    case Opcode::V_LD_CONST_MAT4_F32: return 30;

    // Deeper than one float and far shallower than eight of them: the address is
    // computed once and the bank sequence runs once. A third again, chosen the way
    // the rest of this table is.
    case Opcode::V_LD_SHARED_16X16_F32:
    case Opcode::V_LD_SHARED_16X16_F16: return 40;

    // Twice that, for the trip out of the SM and back. The point of the
    // instruction is that this is not the 200 an L2 hit costs.
    case Opcode::V_LD_CLUSTER_F32: return 60;

    // Not answered here. What a global load costs in time depends on where the
    // line was found, which is a question for the memory model — the scheduler
    // takes it from global_access_cost instead.
    case Opcode::V_LD_GLOBAL_F32:
    case Opcode::V_LD_GLOBAL_VEC3_F32: return 0;

    // A store is fire-and-forget: the warp hands it to memory and carries on, so
    // nothing downstream waits on it. Modelling a full write buffer would be a
    // different thing again.
    case Opcode::V_ST_GLOBAL_F32:
    case Opcode::V_ST_SHARED_F32: return 0;

    // Not answered here, for the reason the loads are not: what an atomic costs
    // in time is decided by how many lanes collide, which is a property of the
    // addresses rather than of the opcode.
    case Opcode::V_ATOM_ADD_GLOBAL_F32: return 0;

    // Not answered here either, and for a stronger reason than the loads: the
    // copy's latency is never the warp's. It is recorded against the copy and
    // becomes a wait only where S_CP_ASYNC_WAIT asks for it.
    case Opcode::V_CP_ASYNC_SHARED_GLOBAL_F32: return 0;

    // Warp state rather than a result, and the waiting they cause is the
    // scheduler's own — a barrier is not an instruction whose answer arrives late.
    case Opcode::BRA:
    case Opcode::BRA_DIV:
    case Opcode::BARRIER:
    case Opcode::BARRIER_CLUSTER:
    case Opcode::S_SYNCWARP:
    case Opcode::S_CP_ASYNC_WAIT:
    case Opcode::RET: return 0;

    // The block waits for the sort, and the waiting is the scheduler's — but
    // unlike a barrier there is work to do once everyone has arrived, and that
    // work has a depth. A shared-memory round trip again.
    case Opcode::REORDER: return 30;

    // Deeper than the arithmetic pipe, as a unit that retires 4,096 operations in
    // one issue has to be. Sixteen FMAs deep, which is the same ratio to its own
    // cost that an FMA has to its latency.
    case Opcode::V_MMA_16X16X16_F32:
    case Opcode::V_MMA_16X16X16_F16: return 32;

    // A crossbar across the warp, priced like the shared-memory traffic it
    // replaces and no deeper than the arithmetic around it.
    case Opcode::S_BALLOT:
    case Opcode::S_ANY:
    case Opcode::S_ALL:
    case Opcode::V_SHUFFLE_F32: return 8;

    // Ordinary arithmetic, a few cycles deep and fully pipelined.
    case Opcode::V_MUL_F32:
    case Opcode::V_ADD_F32:
    case Opcode::V_SUB_F32:
    case Opcode::V_FMA_F32:
    case Opcode::V_MIN_F32:
    case Opcode::V_MAX_F32:
    case Opcode::V_MOV_F32:
    case Opcode::V_CMP_F32:
    case Opcode::V_ADD_VEC3_F32:
    case Opcode::V_SUB_VEC3_F32:
    case Opcode::V_SCALE_VEC3_F32:
    case Opcode::V_DOT_VEC3_F32:
    case Opcode::V_CROSS_VEC3_F32:
    case Opcode::V_MATVEC_MAT4_F32: return 4;
    }

    return 4;
}
