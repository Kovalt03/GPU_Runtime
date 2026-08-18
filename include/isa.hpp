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
//   SHAPE   omitted = scalar, VEC3 = 3 consecutive regs, MAT4 = 16.
//           Reserved: VEC4, MAT3 (9 regs) — VEC4 and wider must start at a
//           4-aligned register index.
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

    // Three consecutive floats in one instruction, dst naming the first
    // register as every VEC3 does. Not shorthand for three of the above: a warp
    // asking for twelve bytes at one address touches the lines they fall in
    // once, where three instructions pay for their lines three times.
    V_LD_GLOBAL_VEC3_F32,  // reg[dst..+2] = global[reg[src0] + imm ..+8]

    V_ST_GLOBAL_F32,  // global[reg[src0] + imm] = reg[src1]      (dst unused)
    V_LD_SHARED_F32,  // reg[dst] = shared[reg[src0] + imm]       (src1 unused)
    V_ST_SHARED_F32,  // shared[reg[src0] + imm] = reg[src1]      (dst unused)

    // Global to shared without the register file in between, and without the
    // warp waiting for it. Ampere's cp.async, and the scheme's third memory
    // verb: it is neither a load nor a store, so V_LD_ / V_ST_ cannot name it.
    // Spaces read destination first, as PTX's cp.async.shared.global does.
    //
    // The warp issues it and carries on. Nothing may read the destination until
    // S_CP_ASYNC_WAIT says it has landed, and reading it early is refused
    // rather than answered with bytes that would be garbage on hardware.
    //
    // The global address is src0 even though the name reads shared first: the
    // rule that the priced address sits in src0 is what lets the memory model
    // charge every access the same way, and it outranks reading order.
    V_CP_ASYNC_SHARED_GLOBAL_F32,  // shared[reg[src1]] = global[reg[src0] + imm]

    // A whole fragment out of shared memory: eight consecutive floats into the
    // eight consecutive registers V_MMA_16X16X16_F32 reads.
    //
    // Hardware's ldmatrix, and the same argument V_LD_GLOBAL_VEC3_F32 made — a
    // wide load is not several narrow ones. What it buys here is different,
    // though. The wide global load buys transactions, because 32 lanes asking
    // for twelve bytes touch fewer lines than three instructions do. Shared
    // memory has no lines, so this buys **waiting**: one round trip instead of
    // eight, in a machine that issues in order and has nothing to fill them with.
    //
    // The eight elements are consecutive because this ISA's fragment layout says
    // so. Hardware's layout is chosen for the datapath, which is why its ldmatrix
    // has to gather rather than read a run.
    V_LD_SHARED_16X16_F32,  // reg[dst..+7] = shared[reg[src0] + imm ..+28]

    // The same fragment in half precision: 256 halves are 128 registers across the
    // warp, so a lane holds four rather than eight, two to a register.
    //
    // The narrow type is what is in memory. Nothing here converts — a kernel
    // multiplying halves reads them already packed, as it does on hardware, and
    // the packing is the caller's job.
    V_LD_SHARED_16X16_F16,  // reg[dst..+3] = shared[reg[src0] + imm ..+12]

    // Another block's shared memory, from a block in the same cluster.
    //
    // Hopper's distributed shared memory. src1 names which block of the cluster
    // by its rank, and rank 0 with no cluster declared is the block's own — a
    // lone block is a cluster of one, so a kernel written for this does not have
    // to be written twice.
    //
    // The space token is the fourth, beside GLOBAL and SHARED: it is shared
    // memory, but not this block's, and the difference is what it costs.
    V_LD_CLUSTER_F32,  // reg[dst] = shared[reg[src0] + imm] of block reg[src1]

    // Read, add, write back, indivisibly, and hand the lane what was there
    // before. The fourth memory verb, and the only one whose answer depends on
    // what the other lanes are doing at the same instant.
    //
    // dst carries the old value because that is what makes it more than a
    // combine: a lane that gets 7 back from a counter owns slot 7, and nothing
    // else in this ISA can hand 32 lanes 32 different answers from one address.
    // Compaction, and so GPU-driven culling, is that idiom.
    //
    // Lanes naming the same address serialise, which the memory model charges
    // for — see atomic_access. It is the reason a warp reduction exists.
    V_ATOM_ADD_GLOBAL_F32,  // reg[dst] = global[reg[src0]]; global[...] += reg[src1]

    // Control flow — warp state only, hence no V_ prefix.
    BRA,      // pc += (int32_t)imm                       (unconditional)
    BRA_DIV,  // if (reg[src0] != 0.0f) pc += (int32_t)imm
              //   → splits activeMask (divergence point)

    // Warp-level — the lanes talking to each other rather than merely
    // disagreeing. See the two notes below for what the S_ prefix means and
    // what imm carries.
    S_BALLOT,    // mask[dst] = participants where reg[src0] != 0
    S_ANY,       // reg[dst] = mask[src0] != 0,           in every participant
    S_ALL,       // reg[dst] = mask[src0] == participants, in every participant
    S_SYNCWARP,  // wait until every participant has arrived    (dst/src unused)

    // Wait until at most imm copies are still in flight for this warp. Zero
    // waits for all of them; one leaves the most recent outstanding, which is
    // what lets a kernel work on the tile it has while the next one arrives.
    //
    // Warp state rather than a lane result, hence S_ and no type suffix — the
    // copies were issued by the warp and land for the warp.
    S_CP_ASYNC_WAIT,  // wait until in-flight copies <= imm    (dst/src unused)

    // The lane exchange, and the only way a value crosses between lanes without
    // going through memory. src1 holds a lane number rather than a value, and a
    // different one per lane, so the warp gathers in one instruction.
    V_SHUFFLE_F32,  // reg[dst] = reg[src0] of the lane in reg[src1]

    // The same tile with half-precision operands and a single-precision
    // accumulator, which is the arrangement every tensor core makes: the inputs
    // are where the width is saved and the sum is where it would be missed.
    //
    // Two operand registers do the work of four, and it is priced at half — the
    // claim hardware makes for a narrow input, and the reason f16 exists on these
    // units at all.
    V_MMA_16X16X16_F16,  // reg[dst..+7] += A(reg[src0..+3]) * B(reg[src1..+3])

    // One 16x16x16 multiply-accumulate, performed by the warp together:
    // D = A * B + C, with every lane holding an eighth of each matrix.
    //
    // 4,096 multiply-adds in one instruction. It is V_ rather than S_ because
    // every lane ends up with different elements of D, and it is warp-level
    // because no lane holds enough of A or B to compute anything alone — this is
    // the first instruction here whose operands are the warp's registers rather
    // than a lane's.
    //
    // Layout: each matrix is 16x16 row-major across the warp, lane L holding the
    // eight elements at L*8. Rows are 16 wide, so lanes 2r and 2r+1 hold row r
    // between them. Chosen for being explainable — hardware's fragment layouts
    // are chosen for the datapath, and modelling one of those would say nothing
    // this cannot.
    V_MMA_16X16X16_F32,  // reg[dst..+7] += A(reg[src0..+7]) * B(reg[src1..+7])

    // CUDA's __syncthreads().
    BARRIER,  // wait for every live warp of the block    (all operands unused)

    // CUDA's cluster.sync(). One level wider: every live warp of every block in
    // the cluster, which is what makes a block's writes safe for its neighbours
    // to read. A block outside a cluster meets only itself here, so this is a
    // BARRIER for it.
    BARRIER_CLUSTER,  // wait for every live warp of the cluster

    // Regroup the block's threads so that lanes wanting the same thing share a
    // warp. Ada's shader execution reordering, and OptiX's optixReorder.
    //
    // Every thread offers a key in src0 and the block's threads are redistributed
    // across its warps in key order. Nothing about a thread changes except which
    // warp it is in — its registers and its pc travel with it — so the answer is
    // the same either way and only the divergence differs.
    //
    // No prefix and no type, like BARRIER: it writes no register. It is also a
    // rendezvous for the same reason a barrier is, since threads cannot be
    // regrouped while some of them are elsewhere.
    REORDER,  // regroup the block's threads by reg[src0]   (dst/src1 unused)
    RET,      // end thread                               (all operands unused)
};

// --- what S_ means here -----------------------------------------------------
// AMD reserves s_ for instructions writing a scalar register file. This machine
// has none, so the line is drawn on the result: S_ where every lane ends up with
// the same value, V_ where they differ. OpcodeNamesFollowScheme checks it.
//
// So S_ANY and S_ALL are S_ despite writing lane registers — one value
// broadcast is what a scalar register would have held — and a lane exchange is
// V_SHUFFLE_F32, every lane getting something different.

// --- who takes part ---------------------------------------------------------
// The warp-collective instructions name the lanes they operate over in imm,
// rather than taking whichever lanes happen to be at the instruction. Under
// WarpPolicy::Independent the latter is a property of the scheduler and not of
// the program: the same kernel on the same input could ballot over a different
// set from one run to the next.
//
// CUDA made this move when Volta gave each thread its own pc, replacing the
// maskless __ballot and __shfl with the _sync family.
//
// A named lane that has not arrived is refused rather than skipped — a promise
// the program cannot keep says more as a message than as a quietly smaller
// reduction.

// Registers a VEC3 occupies, and floats the wide load moves. Named because the
// scheduler has to size a warp's address list by it and the two must agree.
inline constexpr uint32_t VEC3_COMPONENTS = 3;

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
Instruction make_v_ld_global_vec3_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_global_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);
Instruction make_v_ld_shared_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_shared_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);

// address, then the value to add. dst takes what was there before.
//
// Only add, and only global. min is the other one hardware offers that this
// machine would have a use for — a depth buffer whose pixels are not owned by
// one thread each — and the scheme leaves V_ATOM_MIN_GLOBAL_F32 for it rather
// than being widened now for a scene that does not exist yet.
Instruction make_v_atom_add_global_f32(uint8_t dst, uint8_t addr_reg, uint8_t src,
                                       float offset = 0.0f);

// The copy takes two addresses and no value, which is what it is for: the
// destination is in shared memory, the source in global, and neither passes
// through a register. offset is added to the global address, matching the loads.
Instruction make_v_cp_async_shared_global_f32(uint8_t shared_addr_reg,
                                              uint8_t global_addr_reg,
                                              float offset = 0.0f);

// outstanding is how many copies may still be in flight when the warp goes on.
// Spelled at every call site for the reason make_s_syncwarp's participants is:
// the useful values are 0 and 1, and which one a kernel means is the whole
// difference between waiting for everything and keeping one load ahead.
Instruction make_s_cp_async_wait(uint32_t outstanding);

// CONTROL FLOW
Instruction make_bra(int32_t offset);
Instruction make_bra_div(uint8_t cond_reg, int32_t offset);
// dst indexes Warp::masks for S_BALLOT and src0 does for S_ANY / S_ALL, where
// every other factory here indexes the lane register file. The operand fields
// are uint8_t either way, so nothing but this line says which.
//
// participants is spelled at every call site rather than defaulting to all 32.
// CUDA defaults __syncwarp() to 0xffffffff and it is the commonest way to get
// the set wrong — a kernel that has diverged names lanes that are not there.
Instruction make_s_ballot(uint8_t dst, uint8_t src0, uint32_t participants);
Instruction make_s_any(uint8_t dst, uint8_t src0, uint32_t participants);
Instruction make_s_all(uint8_t dst, uint8_t src0, uint32_t participants);
Instruction make_s_syncwarp(uint32_t participants);

// Four registers rather than eight, and the same rules otherwise.
Instruction make_v_ld_shared_16x16_f16(uint8_t dst, uint8_t addr_reg,
                                       float offset = 0.0f);

// The accumulator is eight registers of single precision; the operands are four
// each of packed halves.
Instruction make_v_mma_16x16x16_f16(uint8_t dst, uint8_t src0, uint8_t src1,
                                    uint32_t participants);

// dst names the first of eight registers, as every wide shape does, and starts
// on a multiple of four.
Instruction make_v_ld_shared_16x16_f32(uint8_t dst, uint8_t addr_reg,
                                       float offset = 0.0f);

// address, then the rank of the block whose shared memory to read.
Instruction make_v_ld_cluster_f32(uint8_t dst, uint8_t addr_reg, uint8_t rank_reg,
                                  float offset = 0.0f);

Instruction make_barrier_cluster();

// The key is a lane register, and threads are grouped by equal keys. Anything
// whole works: what it means is the caller's, and hardware treats it the same
// way — a coherence hint rather than an ordering.
Instruction make_reorder(uint8_t key_reg);

// Every lane takes part, always: the shape is fixed and a fragment is missing
// without one of them. participants is still spelled out, so the call site says
// what the instruction assumes rather than leaving it to be discovered.
//
// dst is the accumulator and is read as well as written — D = A * B + C with C
// and D the same eight registers, which is what "accumulate" means and what lets
// a chain of these sum a long product without touching memory.
Instruction make_v_mma_16x16x16_f32(uint8_t dst, uint8_t src0, uint8_t src1,
                                    uint32_t participants);

// src1 names a lane rather than holding a value, and it does so as a float in a
// lane register: each lane reads from whichever lane its own src1 points at.
Instruction make_v_shuffle_f32(uint8_t dst, uint8_t src0, uint8_t src1,
                               uint32_t participants);

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

// The participation mask rides in imm the same way, and for the same reason: 32
// bits of mask do not survive a float's 24-bit mantissa as a value.
inline float encode_lane_mask(uint32_t participants)
{
    float imm;
    std::memcpy(&imm, &participants, sizeof(float));
    return imm;
}

inline uint32_t decode_lane_mask(float imm)
{
    uint32_t participants;
    std::memcpy(&participants, &imm, sizeof(uint32_t));
    return participants;
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

// How long a result takes to become usable, as against how much issue capacity
// the instruction consumes. instruction_cost is the second; this is the first,
// and the two are independent — a global load occupies the issue slot as briefly
// as an add and takes hundreds of cycles to land.
//
// The distinction is the reason warps are batched at all. A scheduler with other
// warps ready can step over one waiting on a load; a scheduler without them
// cannot, and occupancy is the number that decides which case a kernel is in.
//
// Zero where an instruction produces no result to wait on — a store, a branch, a
// barrier — and zero for a global load too, whose answer depends on where the
// line was found and so comes from the memory model rather than from here.
//
// Consulted only under LatencyModel::Modelled. Every figure in benchmarks/ was
// taken with latency ignored, which is the model where a result is available the
// instant it is issued.
uint32_t instruction_latency(Opcode op);
