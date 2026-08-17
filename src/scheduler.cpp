#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>
#include "isa.hpp"
#include "thread.hpp"

#include "scheduler.hpp"

namespace {

// Global and shared are both addressed as a byte offset from a base, so the
// four load/store opcodes share these. Alignment is checked here rather than in
// decode_address because branch offsets go through that too and have no
// alignment requirement.
//
// std::memcpy, not *reinterpret_cast<float*>(base + addr): the buffer is bytes,
// and reading it as a float would violate strict aliasing.

void require_f32_access(size_t size, size_t addr, const char* what)
{
    if (addr % sizeof(float) != 0) {
        throw std::runtime_error(std::string(what) + ": address " + std::to_string(addr) +
                                 " is not 4-byte aligned");
    }
    // Not addr + sizeof(float) > size: a large addr overflows that sum and slips
    // through. Checking addr first also keeps size - addr from wrapping.
    if (addr > size || size - addr < sizeof(float)) {
        throw std::runtime_error(std::string(what) + ": address " + std::to_string(addr) +
                                 " leaves the " + std::to_string(size) + "-byte span");
    }
}

float load_f32(const uint8_t* base, size_t size, size_t addr, const char* what)
{
    require_f32_access(size, addr, what);
    float value = 0.0f;
    std::memcpy(&value, base + addr, sizeof(float));
    return value;
}

void store_f32(uint8_t* base, size_t size, size_t addr, float value, const char* what)
{
    require_f32_access(size, addr, what);
    std::memcpy(base + addr, &value, sizeof(float));
}

bool compare(CmpOp op, float a, float b, const char* what)
{
    switch (op) {
    case CmpOp::LT: return a < b;
    case CmpOp::GT: return a > b;
    case CmpOp::EQ: return a == b;
    case CmpOp::NEQ: return a != b;
    case CmpOp::LE: return a <= b;
    case CmpOp::GE: return a >= b;
    }
    throw std::runtime_error(std::string(what) + ": unknown compare condition " +
                             std::to_string(static_cast<uint32_t>(op)));
}

// Where a branch at instr_pc lands. decode_branch_offset (isa.hpp) undoes the
// float encoding and yields a signed value, since a backward jump is how a loop
// is written — casting the float straight to uint32_t would be undefined for
// those. The sum is taken in int64_t so that neither end can wrap before the
// range check.
uint32_t branch_target(uint32_t instr_pc, float imm, const char* what)
{
    const int64_t target = static_cast<int64_t>(instr_pc) + decode_branch_offset(imm);
    if (target < 0) {
        throw std::runtime_error(std::string(what) + ": branch from " +
                                 std::to_string(instr_pc) + " lands before the program");
    }
    return static_cast<uint32_t>(target);
}

// Shared memory is declared as floats but addressed in bytes, like global.
uint8_t* shared_bytes(ThreadBlock& block)
{
    return reinterpret_cast<uint8_t*>(block.shared_mem.data());
}

constexpr size_t SHARED_MEM_BYTES = SHARED_MEM_FLOATS * sizeof(float);

}  // namespace

double SchedulerStats::divergence_rate() const
{
    // No steps means no waste rather than a division by zero.
    if (warp_steps == 0) {
        return 0.0;
    }
    // Cast before dividing: both operands are uint64_t, and integer division
    // would floor every rate below 1.0 to zero.
    return static_cast<double>(masked_lane_slots()) / static_cast<double>(lane_slots());
}

size_t decode_address(float value, const char* what)
{
    // NaN first: it compares false against everything, so the tests below would
    // let it through and blame the wrong thing.
    if (std::isnan(value)) {
        throw std::runtime_error(std::string(what) + ": address is NaN");
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(what) + ": address is infinite");
    }
    if (value < 0.0f) {
        throw std::runtime_error(std::string(what) + ": address " +
                                 std::to_string(value) + " is negative");
    }
    // Truncating a fractional address would silently target the wrong bytes.
    if (value != std::floor(value)) {
        throw std::runtime_error(std::string(what) + ": address " +
                                 std::to_string(value) + " is not a whole number");
    }
    return static_cast<size_t>(value);
}

// The mask file is the warp's, not a lane's, and it is four deep where the
// lane file is 256. An operand that would be a perfectly ordinary register
// index is out of range here, so the two cannot share a check.
std::string to_hex(uint32_t v)
{
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08X", v);
    return buf;
}

void require_mask_register(uint32_t mask, const char* what)
{
    if (mask >= WARP_MASK_REGISTERS) {
        throw std::runtime_error(std::string(what) + ": mask register " +
                                 std::to_string(mask) + " leaves the " +
                                 std::to_string(WARP_MASK_REGISTERS) + "-mask file");
    }
}

// Every lane the instruction named has to be here. A mask that names one which
// is not is a promise the program could not keep — under LowestPc because it
// branched elsewhere, under Independent because it has not been scheduled yet,
// and either way the collective result would be over a set nobody asked for.
//
// CUDA calls this undefined. Naming it costs one comparison and turns a wrong
// answer into a message.
void require_participants_present(const Warp& warp, uint32_t participants,
                                  const char* what)
{
    const uint32_t missing = participants & ~warp.active_mask;
    if (missing != 0) {
        throw std::runtime_error(std::string(what) +
                                 ": lanes named in the participation mask have not "
                                 "arrived (declared 0x" +
                                 to_hex(participants) + ", present 0x" +
                                 to_hex(warp.active_mask) + ", missing 0x" +
                                 to_hex(missing) + ")");
    }
}

// Which opcodes step_warp handles in one block, gathering across the lanes and
// then advancing the pc. BARRIER and S_SYNCWARP are not among them: both may
// leave the pc where it is, so each keeps its own block above.
bool is_warp_level(Opcode op)
{
    return op == Opcode::S_BALLOT || op == Opcode::S_ANY || op == Opcode::S_ALL ||
           op == Opcode::V_SHUFFLE_F32;
}

// A lane index arriving as a float, which is how this machine carries every
// integer. It has to be whole and inside the warp before it can index anything.
uint32_t require_lane_index(float value, const char* what)
{
    if (value != std::floor(value) || value < 0.0f ||
        value >= static_cast<float>(WARP_SIZE)) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(value) +
                                 " is not a lane index in [0, " +
                                 std::to_string(WARP_SIZE) + ")");
    }
    return static_cast<uint32_t>(value);
}

void require_register_range(uint32_t reg, uint32_t count, const char* what)
{
    // reg comes from a uint8_t operand and count is at most 16, so this sum
    // cannot overflow a uint32_t — unlike the byte offsets in require_f32_access.
    if (reg + count > REGS_PER_THREAD) {
        throw std::runtime_error(std::string(what) + ": registers " +
                                 std::to_string(reg) + ".." +
                                 std::to_string(reg + count - 1) + " leave the " +
                                 std::to_string(REGS_PER_THREAD) + "-register file");
    }
}

void require_register_alignment(uint32_t reg, uint32_t alignment, const char* what)
{
    if (reg % alignment != 0) {
        throw std::runtime_error(std::string(what) + ": register " + std::to_string(reg) +
                                 " is not a multiple of " + std::to_string(alignment));
    }
}

// One instruction, one lane. thread.pc has already been advanced past instr, so
// a branch assigns rather than adds.
void WarpScheduler::execute(const Instruction& instr, uint32_t instr_pc, Thread& thread,
                            ThreadBlock& block, DeviceSpan global)
{
    // No default label: -Wswitch then reports any opcode added later that is
    // not handled here, the same safety net as opcode_name().

    switch (instr.op) {
    // Scalar ALU. A single register is always in range, the operand fields being
    // uint8_t, so require_register_range is only needed from VEC3 upwards.
    case Opcode::V_MUL_F32:
        thread.regs[instr.dst] = thread.regs[instr.src0] * thread.regs[instr.src1];
        break;

    case Opcode::V_ADD_F32:
        thread.regs[instr.dst] = thread.regs[instr.src0] + thread.regs[instr.src1];
        break;

    case Opcode::V_SUB_F32:
        thread.regs[instr.dst] = thread.regs[instr.src0] - thread.regs[instr.src1];
        break;

    // Division by zero yields inf and a negative square root yields NaN, both
    // left to propagate. Real GPUs do not trap here either, and the kernel
    // guards the domain itself: Möller-Trumbore tests |a| < eps before the
    // reciprocal. Throwing would let one bad lane kill an entire render.
    case Opcode::V_RCP_F32:
        thread.regs[instr.dst] = 1.0f / thread.regs[instr.src0];
        break;

    case Opcode::V_SQRT_F32:
        thread.regs[instr.dst] = std::sqrt(thread.regs[instr.src0]);
        break;

    // dst doubles as the accumulator. An independent three-operand FMA will not
    // fit in an 8-byte Instruction.
    case Opcode::V_FMA_F32:
        thread.regs[instr.dst] += thread.regs[instr.src0] * thread.regs[instr.src1];
        break;

    // std::fmin/fmax rather than std::min/max: given a NaN the latter return
    // whichever argument came first, which would make a BVH slab test depend on
    // operand order.
    case Opcode::V_MIN_F32:
        thread.regs[instr.dst] =
            std::fmin(thread.regs[instr.src0], thread.regs[instr.src1]);
        break;

    case Opcode::V_MAX_F32:
        thread.regs[instr.dst] =
            std::fmax(thread.regs[instr.src0], thread.regs[instr.src1]);
        break;

    // The one instruction that reads no register.
    case Opcode::V_MOV_F32: thread.regs[instr.dst] = instr.imm; break;

    // VEC3. dst/src0/src1 name the FIRST of three consecutive registers, so
    // require_register_range has to run before any of them is touched.
    //
    // Read all inputs before writing any output. The register ranges are free to
    // overlap — Möller-Trumbore does exactly that with
    // `V_ADD_VEC3_F32 r40, r0, r40` — and writing dst[0] first would corrupt an
    // input still needed for dst[1] and dst[2].
    case Opcode::V_ADD_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_ADD_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_ADD_VEC3_F32 src0");
        require_register_range(instr.src1, 3, "V_ADD_VEC3_F32 src1");

        float out[3];
        for (uint32_t i = 0; i < 3; ++i) {
            out[i] = thread.regs[instr.src0 + i] + thread.regs[instr.src1 + i];
        }
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    case Opcode::V_SUB_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_SUB_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_SUB_VEC3_F32 src0");
        require_register_range(instr.src1, 3, "V_SUB_VEC3_F32 src1");

        float out[3];
        for (uint32_t i = 0; i < 3; ++i) {
            out[i] = thread.regs[instr.src0 + i] - thread.regs[instr.src1 + i];
        }
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    // src1 is a single scalar here, not a vector, so it carries no + i.
    case Opcode::V_SCALE_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_SCALE_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_SCALE_VEC3_F32 src0");

        const float scale = thread.regs[instr.src1];
        float out[3];
        for (uint32_t i = 0; i < 3; ++i) {
            out[i] = thread.regs[instr.src0 + i] * scale;
        }
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    // A reduction, not an element-wise operation: three products collapse into
    // the single scalar dst.
    case Opcode::V_DOT_VEC3_F32: {
        require_register_range(instr.dst, 1, "V_DOT_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_DOT_VEC3_F32 src0");
        require_register_range(instr.src1, 3, "V_DOT_VEC3_F32 src1");

        float sum = 0.0f;
        for (uint32_t i = 0; i < 3; ++i) {
            sum += thread.regs[instr.src0 + i] * thread.regs[instr.src1 + i];
        }
        thread.regs[instr.dst] = sum;
        break;
    }

    // Component i draws on the other two components, so this cannot be a loop —
    // and the temporary is mandatory rather than merely tidy, since writing
    // dst[0] would destroy an input that dst[1] and dst[2] still need.
    case Opcode::V_CROSS_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_CROSS_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_CROSS_VEC3_F32 src0");
        require_register_range(instr.src1, 3, "V_CROSS_VEC3_F32 src1");

        const float* a = &thread.regs[instr.src0];
        const float* b = &thread.regs[instr.src1];
        const float out[3] = {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    // src1 unused. A zero-length vector divides by zero and yields NaN in all
    // three components, left to propagate for the same reason as V_RCP_F32.
    case Opcode::V_NORM_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_NORM_VEC3_F32 dst");
        require_register_range(instr.src0, 3, "V_NORM_VEC3_F32 src0");

        // The length depends on all three components, so it has to be complete
        // before any of them is divided.
        float length_sq = 0.0f;
        for (uint32_t i = 0; i < 3; ++i) {
            const float c = thread.regs[instr.src0 + i];
            length_sq += c * c;
        }
        const float length = std::sqrt(length_sq);

        float out[3];
        for (uint32_t i = 0; i < 3; ++i) {
            out[i] = thread.regs[instr.src0 + i] / length;
        }
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    case Opcode::V_LD_GLOBAL_F32: {
        const size_t addr =
            decode_address(thread.regs[instr.src0] + instr.imm, "V_LD_GLOBAL_F32");
        thread.regs[instr.dst] =
            load_f32(global.base, global.size, addr, "V_LD_GLOBAL_F32");
        break;
    }

    // Twelve bytes from one address. No alignment beyond the four bytes every
    // f32 access wants: a VEC3 is three floats wherever they lie, and the
    // triangles these read are packed nine floats to the triangle.
    case Opcode::V_LD_GLOBAL_VEC3_F32: {
        require_register_range(instr.dst, 3, "V_LD_GLOBAL_VEC3_F32 dst");
        const size_t addr =
            decode_address(thread.regs[instr.src0] + instr.imm, "V_LD_GLOBAL_VEC3_F32");
        for (uint32_t i = 0; i < 3; ++i) {
            thread.regs[instr.dst + i] =
                load_f32(global.base, global.size, addr + i * sizeof(float),
                         "V_LD_GLOBAL_VEC3_F32");
        }
        break;
    }

    // Store leaves dst unused: the address is src0 and the value is src1.
    case Opcode::V_ST_GLOBAL_F32: {
        const size_t addr =
            decode_address(thread.regs[instr.src0] + instr.imm, "V_ST_GLOBAL_F32");
        store_f32(global.base, global.size, addr, thread.regs[instr.src1],
                  "V_ST_GLOBAL_F32");
        break;
    }

    case Opcode::V_LD_SHARED_F32: {
        const size_t addr =
            decode_address(thread.regs[instr.src0] + instr.imm, "V_LD_SHARED_F32");
        thread.regs[instr.dst] =
            load_f32(shared_bytes(block), SHARED_MEM_BYTES, addr, "V_LD_SHARED_F32");
        break;
    }

    case Opcode::V_ST_SHARED_F32: {
        const size_t addr =
            decode_address(thread.regs[instr.src0] + instr.imm, "V_ST_SHARED_F32");
        store_f32(shared_bytes(block), SHARED_MEM_BYTES, addr, thread.regs[instr.src1],
                  "V_ST_SHARED_F32");
        break;
    }

    // decode_cmp_op(instr.imm) recovers the condition. The result is written as
    // 1.0f or 0.0f so that BRA_DIV, which only tests against zero, can consume
    // it. A switch over CmpOp with no default keeps -Wswitch watching this one
    // too.
    // MAT4 is row-major: reg[src0 + row * 4 + col]. Sixteen registers for the
    // matrix, four each for the vector and the result, all 4-aligned.
    //
    // Every output component reads every input component, so the temporary is
    // mandatory rather than tidy — writing dst[0] first would destroy a value
    // the other three still need.
    case Opcode::V_MATVEC_MAT4_F32: {
        require_register_range(instr.dst, 4, "V_MATVEC_MAT4_F32 dst");
        require_register_range(instr.src0, 16, "V_MATVEC_MAT4_F32 src0");
        require_register_range(instr.src1, 4, "V_MATVEC_MAT4_F32 src1");
        require_register_alignment(instr.dst, 4, "V_MATVEC_MAT4_F32 dst");
        require_register_alignment(instr.src0, 4, "V_MATVEC_MAT4_F32 src0");
        require_register_alignment(instr.src1, 4, "V_MATVEC_MAT4_F32 src1");

        // Row-major, so row r occupies m[r * 4 .. r * 4 + 3].
        //
        // The temporary is mandatory rather than tidy: every output component
        // reads every input one, so writing dst[0] straight away would destroy a
        // value the other three still need. Möller-Trumbore already relies on
        // operands overlapping, and a camera transform in place would too.
        const float* m = &thread.regs[instr.src0];
        const float* v = &thread.regs[instr.src1];
        float out[4];
        for (uint32_t row = 0; row < 4; ++row) {
            out[row] = 0.0f;
            for (uint32_t col = 0; col < 4; ++col) {
                out[row] += m[row * 4 + col] * v[col];
            }
        }
        for (uint32_t i = 0; i < 4; ++i) {
            thread.regs[instr.dst + i] = out[i];
        }
        break;
    }

    case Opcode::V_CMP_F32: {
        thread.regs[instr.dst] =
            compare(decode_cmp_op(instr.imm), thread.regs[instr.src0],
                    thread.regs[instr.src1], "V_CMP_F32")
                ? 1.0f
                : 0.0f;
        break;
    }

    // Assigns rather than adds: thread.pc was advanced past this instruction
    // before execute() ran, so adding would land one instruction too far.
    case Opcode::BRA: thread.pc = branch_target(instr_pc, instr.imm, "BRA"); break;

    // The same jump, taken only where regs[src0] != 0.0f. Lanes that skip it
    // keep the advanced pc and simply fall through; nothing here touches the
    // mask, since the divergence surfaces on its own once the two groups sit at
    // different pcs.
    case Opcode::BRA_DIV:
        if (thread.regs[instr.src0] != 0.0f) {
            thread.pc = branch_target(instr_pc, instr.imm, "BRA_DIV");
        }
        break;

    // min-PC skips retired threads, so clearing the flag is all that is needed.
    case Opcode::RET: thread.active = false; break;

    case Opcode::BARRIER:
        throw std::runtime_error(
            "BARRIER reached the lane loop; step_warp "
            "should have intercepted it");

    // Warp-level, so none of them belongs in a per-lane loop, and step_warp
    // intercepts them before it: a ballot has to see all 32 lanes at once, and
    // S_SYNCWARP decides whether the warp advances at all.
    case Opcode::S_BALLOT:
    case Opcode::S_ANY:
    case Opcode::S_ALL:
    case Opcode::S_SYNCWARP:
    case Opcode::V_SHUFFLE_F32:
        throw std::runtime_error(
            "warp-level primitive " + std::string(opcode_name(instr.op)) +
            " is not implemented — the opcode exists so the naming and mask "
            "storage can be settled");
    }
}

// Which pc a warp issues when no lane may be starved.
//
// Fair over distinct live pcs, not over lanes: lanes at one pc issue together,
// so a turn each would weight a pc by how many sit at it and starve a lone
// waiter exactly as LowestPc does. The deadlock this breaks is one against
// thirty-one.
//
// Round-robin, taking the smallest live pc above the one issued last and
// wrapping when there is none:
//
//   live {4, 8}, last 0 -> 4    last 4 -> 8    last 8 -> 4
//
// Costs nothing measurable. Every existing kernel issues the same work under
// either policy, warp_steps included — a pc group runs the same instructions
// whatever order the groups are interleaved in.
//
// What it gives away is the reconvergence LowestPc got for free: lanes that run
// ahead are no longer made to wait, so a kernel that needs them together has to
// say so with S_SYNCWARP.
uint32_t WarpScheduler::select_independent_pc(Warp& warp) const
{
    // The caller has already established that some lane is live, so a pc will
    // be found. Both of these staying at their sentinel would mean it did not.
    uint32_t successor = UINT32_MAX;  // smallest live pc above last_issued_pc
    uint32_t smallest = UINT32_MAX;   // smallest live pc at all, for the wrap

    for (const Thread& t : warp.threads) {
        if (!t.active) {
            continue;
        }
        if (smallest > t.pc) {
            smallest = t.pc;
        }
        if (t.pc > warp.last_issued_pc && t.pc < successor) {
            successor = t.pc;
        }
    }
    const uint32_t chosen = (successor != UINT32_MAX) ? successor : smallest;
    warp.last_issued_pc = chosen;
    return chosen;
}

bool LineCache::touch(size_t line)
{
    const auto found = resident_.find(line);
    if (found != resident_.end()) {
        // splice moves it to the most-recently-used end and keeps the iterator
        // valid, so the map needs no update.
        order_.splice(order_.end(), order_, found->second);
        return true;
    }

    if (resident_.size() == capacity_) {
        resident_.erase(order_.front());
        order_.pop_front();
    }
    order_.push_back(line);
    resident_.emplace(line, std::prev(order_.end()));
    return false;
}

void LineCache::clear()
{
    order_.clear();
    resident_.clear();
}

namespace {

// What one transaction costs, whatever instruction asked for it.
//
// A line fetched is a line fetched, so the wide load pays the scalar price per
// line: what it saves is how many lines it asks for. Charging its own cost here
// would hand that straight back, three floats being priced at three hundred once
// per line. Still keyed on the opcode so that pricing a store apart from a load
// stays a change in one place.
uint32_t transaction_cost(Opcode op)
{
    return op == Opcode::V_LD_GLOBAL_VEC3_F32 ? instruction_cost(Opcode::V_LD_GLOBAL_F32)
                                              : instruction_cost(op);
}

// How many floats one lane asks for, which decides how many addresses a warp
// puts to the cache.
uint32_t access_components(Opcode op)
{
    return op == Opcode::V_LD_GLOBAL_VEC3_F32 ? VEC3_COMPONENTS : 1u;
}

}  // namespace

// What one line costs, given where it was found.
//
// Called once per distinct line by global_access_cost, so a warp whose 32 lanes
// share a line looks it up once. Hits here therefore measure reuse *between*
// warps — reuse within one is what coalescing already accounted for.
GlobalAccess WarpScheduler::cache_lookup(size_t line, const Instruction& instr)
{
    if (l1_.touch(line)) {
        ++stats_.l1_hits;
        return {L1_HIT_COST, L1_HIT_LATENCY};
    }

    // touch() installed the line in L1 on the way past, and does the same here,
    // so a line ends up in both levels — inclusive, as NVIDIA's are.
    if (l2_.touch(line)) {
        ++stats_.l2_hits;
        return {L2_HIT_COST, L2_HIT_LATENCY};
    }

    ++stats_.cache_misses;

    return {transaction_cost(instr.op), MEMORY_LATENCY};
}

// What a warp's global load or store costs, given where its lanes are pointing.
GlobalAccess WarpScheduler::global_access(const Warp& warp, const Instruction& instr)
{
    // A store is fire-and-forget, so nothing waits on it whatever it cost.
    const uint32_t store_latency =
        instr.op == Opcode::V_ST_GLOBAL_F32 ? 0u : MEMORY_LATENCY;

    const uint64_t lanes = active_lane_count(warp);
    if (memory_ == MemoryModel::Flat) {
        return {lanes * instruction_cost(instr.op), store_latency};
    }

    // One line per float asked for at worst, so a sorted array counts them
    // without a set. A vector load asks for three and normally lands them in one
    // line, which is the whole of what it is for.
    const uint32_t components = access_components(instr.op);
    std::array<size_t, WARP_SIZE * VEC3_COMPONENTS> lines{};
    uint32_t n = 0;
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        if (!is_active(warp, lane)) {
            continue;
        }
        // The same expression execute() evaluates a lane at a time. Computed
        // twice rather than handed back, which would put the address arithmetic
        // in two places for a saving nothing is waiting on.
        const size_t addr = decode_address(
            warp.threads[lane].regs[instr.src0] + instr.imm, "global access");
        for (uint32_t c = 0; c < components; ++c) {
            lines[n++] = (addr + c * sizeof(float)) / CACHE_LINE_BYTES;
        }
    }
    std::sort(lines.begin(), lines.begin() + n);
    const auto distinct = static_cast<uint64_t>(
        std::unique(lines.begin(), lines.begin() + n) - lines.begin());

    if (memory_ == MemoryModel::Cached) {
        GlobalAccess total;
        for (uint64_t i = 0; i < distinct; ++i) {
            const GlobalAccess line = cache_lookup(lines[i], instr);
            total.cost += line.cost;
            total.latency = std::max(total.latency, line.latency);
        }
        if (instr.op == Opcode::V_ST_GLOBAL_F32) {
            total.latency = 0;
        }
        return total;
    }

    // Per line rather than per lane, which is the whole difference: 32 adjacent
    // floats fit one line and cost what a single lane used to, while 32 scattered
    // ones cost what all 32 used to.
    //
    // A store is charged like a load. Hardware need not read a line a write fills
    // completely and must read one it only partly covers, so the two are not
    // alike — but telling them apart means tracking which lanes cover which
    // bytes, and nothing here asks that yet.
    return {distinct * transaction_cost(instr.op), store_latency};
}

bool WarpScheduler::step_warp(const Program& program, Warp& warp, ThreadBlock& block,
                              DeviceSpan global)
{
    // Thread::pc is the source of truth. The instruction to issue is the lowest
    // pc still live, and no live thread at all means the warp has finished —
    // reported without counting a step, since nothing was issued.
    uint32_t min_pc = UINT32_MAX;
    for (const Thread& t : warp.threads) {
        if (t.active && t.pc < min_pc) {
            min_pc = t.pc;
        }
    }
    if (min_pc == UINT32_MAX) {
        return false;
    }

    // The one place the policy applies. Everything downstream reads the pc that
    // comes out of here and does not care how it was chosen.
    const uint32_t issue_pc =
        (policy_ == WarpPolicy::LowestPc) ? min_pc : select_independent_pc(warp);

    // Built from scratch: the previous step's mask says nothing about which
    // lanes have reached this instruction.
    warp.pc = issue_pc;
    warp.active_mask = 0;
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        if (warp.threads[lane].active && warp.threads[lane].pc == issue_pc) {
            activate(warp, lane);
        }
    }

    // A kernel that forgot its RET walks off the end; fail loudly rather than
    // read past the vector.
    if (warp.pc >= program.size()) {
        throw std::runtime_error("step_warp: pc " + std::to_string(warp.pc) +
                                 " is past the end of a " +
                                 std::to_string(program.size()) + "-instruction program");
    }

    if (program[warp.pc].op == Opcode::BARRIER) {
        // Every live lane has to have arrived. min-PC issue means the warp
        // reaches a barrier with only the lanes that got there; any that
        // branched past sit at a higher pc and will never come back, so the
        // wait would be against threads already gone ahead.
        //
        // CUDA requires the condition around a barrier to evaluate identically
        // across the block and documents that anything else may hang or go
        // quietly wrong. Both are worse than a message.
        uint32_t live = 0;
        for (const Thread& t : warp.threads) {
            if (t.active) {
                ++live;
            }
        }
        if (active_lane_count(warp) != live) {
            throw std::runtime_error(
                "BARRIER at pc " + std::to_string(warp.pc) + " reached by " +
                std::to_string(active_lane_count(warp)) + " of " + std::to_string(live) +
                " live lanes: a barrier inside divergent control flow");
        }

        // Past the barrier before waiting, or a released warp arrives at the
        // same instruction again and never gets anywhere.
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            if (is_active(warp, lane)) {
                ++warp.threads[lane].pc;
            }
        }

        stats_.warp_steps += 1;
        stats_.active_lane_ops += live;
        stats_.weighted_lane_ops += live * instruction_cost(Opcode::BARRIER);

        warp.at_barrier = true;
        return false;
    }

    if (program[warp.pc].op == Opcode::S_SYNCWARP) {
        // The one S_ instruction that waits rather than computing. Its
        // participation mask says which lanes have to meet here, and until they
        // all have, the warp takes its turn without advancing.
        //
        // Not warp-level in the is_warp_level sense, because that path advances
        // the pc unconditionally and waiting is exactly not doing so.
        const uint32_t participants = decode_lane_mask(program[warp.pc].imm);

        // A participant that is live but already past this instruction is never
        // coming back, and waiting for it would spin until the step budget ran
        // out with nothing to say. The other primitives can refuse outright;
        // this one has to look ahead, since a lane still on its way is exactly
        // the case it exists to wait for.
        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            const Thread& t = warp.threads[lane];
            if ((participants & (1u << lane)) != 0 && t.active && t.pc > warp.pc) {
                throw std::runtime_error("S_SYNCWARP at pc " + std::to_string(warp.pc) +
                                         ": lane " + std::to_string(lane) +
                                         " is named in the participation mask but "
                                         "is already at pc " +
                                         std::to_string(t.pc) + " and cannot return");
            }
        }

        if ((participants & ~warp.active_mask) != 0) {
            // Still on their way. Spend the turn so the scheduler moves on to a
            // pc that can make progress, and leave the pc where it is — that is
            // what the waiting consists of.
            //
            // No lane op is counted because no lane did anything, which makes a
            // wait show up as issue capacity spent for nothing. That is what a
            // wait costs, and it is the same thing divergence_rate already
            // measures elsewhere.
            stats_.warp_steps += 1;
            return true;
        }

        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            if (is_active(warp, lane)) {
                ++warp.threads[lane].pc;
            }
        }

        const uint64_t arrived = active_lane_count(warp);
        stats_.warp_steps += 1;
        stats_.active_lane_ops += arrived;
        stats_.weighted_lane_ops += arrived * instruction_cost(Opcode::S_SYNCWARP);

        // Stays in the queue, where BARRIER leaves it: a block barrier waits on
        // the other warps and something else has to release it, while this one
        // waits on its own lanes and has to keep taking turns to see them
        // arrive.
        return true;
    }

    // Warp-level primitives, intercepted for the reason BARRIER is: the lane
    // loop below sees one thread at a time, and a ballot has to see all 32 at
    // once. Everything these need is already in hand — warp.active_mask says
    // who reached this instruction, which is the participating set.
    //
    // The mechanics after the switch are shared and already written: the pc has
    // to be advanced by hand, since skipping the lane loop skips its increment,
    // and the step has to be counted, since skipping the lane loop skips that
    // too. A primitive that forgot either would loop for ever or run free.
    if (is_warp_level(program[warp.pc].op)) {
        const Instruction& instr = program[warp.pc];

        switch (instr.op) {
        case Opcode::S_BALLOT: {
            // masks[dst] = the participants whose reg[src0] is non-zero.
            //
            // Over the declared set rather than over whoever is here, so the
            // answer is the program's and not the scheduler's.
            require_register_range(instr.src0, 1, "S_BALLOT src0");
            require_mask_register(instr.dst, "S_BALLOT dst");
            const uint32_t participants = decode_lane_mask(instr.imm);
            require_participants_present(warp, participants, "S_BALLOT");

            uint32_t voted = 0;
            for (uint32_t lane = 0; lane < WARP_SIZE; lane++) {
                const bool taking_part = (participants & (1u << lane)) != 0;
                if (taking_part && warp.threads[lane].regs[instr.src0] != 0.0f) {
                    voted |= 1u << lane;
                }
            }
            warp.masks[instr.dst] = voted;
            break;
        }

        case Opcode::S_ANY: {
            // reg[dst] = 1.0 in every participant if masks[src0] has any bit
            // set. Written to every participant because the result is
            // warp-uniform and there is nowhere else to put it — the machine has
            // no scalar register file.
            require_mask_register(instr.src0, "S_ANY src0");
            require_register_range(instr.dst, 1, "S_ANY dst");
            const uint32_t participants = decode_lane_mask(instr.imm);
            require_participants_present(warp, participants, "S_ANY");

            const float any = (warp.masks[instr.src0] != 0u) ? 1.0f : 0.0f;
            for (uint32_t lane = 0; lane < WARP_SIZE; lane++) {
                if ((participants & (1u << lane)) != 0) {
                    warp.threads[lane].regs[instr.dst] = any;
                }
            }
            break;
        }

        case Opcode::S_ALL: {
            // As S_ANY, but every participant must have voted.
            //
            // Compared against the declared set, not against all ones and not
            // against whoever turned up: a full mask is unreachable in a warp
            // that has diverged, and the active set is the scheduler's answer
            // rather than the program's.
            require_mask_register(instr.src0, "S_ALL src0");
            require_register_range(instr.dst, 1, "S_ALL dst");
            const uint32_t participants = decode_lane_mask(instr.imm);
            require_participants_present(warp, participants, "S_ALL");

            const float all = (warp.masks[instr.src0] == participants) ? 1.0f : 0.0f;
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                if ((participants & (1u << lane)) != 0) {
                    warp.threads[lane].regs[instr.dst] = all;
                }
            }
            break;
        }

        case Opcode::V_SHUFFLE_F32: {
            // reg[dst] of each participant = reg[src0] of the lane its own
            // reg[src1] names.
            //
            // Gathered in full before anything is written. Writing as it reads
            // would let a lane take the value another has just produced instead
            // of the one it started with — only where dst and src0 are the same
            // register, and only at the wrap, so a rotation is the scene that
            // shows it. No other instruction here leaves its own lane, so none
            // has needed the precaution.
            require_register_range(instr.src0, 1, "V_SHUFFLE_F32 src0");
            require_register_range(instr.src1, 1, "V_SHUFFLE_F32 src1");
            require_register_range(instr.dst, 1, "V_SHUFFLE_F32 dst");
            const uint32_t participants = decode_lane_mask(instr.imm);
            require_participants_present(warp, participants, "V_SHUFFLE_F32");
            std::array<float, WARP_SIZE> gathered{};
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                if ((participants & (1u << lane)) == 0) {
                    continue;
                }

                const uint32_t src_lane = require_lane_index(
                    warp.threads[lane].regs[instr.src1], "V_SHUFFLE_F32 src1");

                // A participant reading outside the mask has nowhere to take a
                // value from, and leaving its dst as gathered was initialised
                // would hand it a number nobody computed.
                if ((participants & (1u << src_lane)) == 0) {
                    throw std::runtime_error(
                        "V_SHUFFLE_F32: lane " + std::to_string(lane) + " reads lane " +
                        std::to_string(src_lane) +
                        ", which the participation mask does not name");
                }
                gathered[lane] = warp.threads[src_lane].regs[instr.src0];
            }
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                if ((participants & (1u << lane)) != 0) {
                    warp.threads[lane].regs[instr.dst] = gathered[lane];
                }
            }
            break;
        }

        default:
            // is_warp_level admitted it, so this cannot be reached. Unlike
            // opcode_name(), which must cover every opcode and therefore has no
            // default, this switch handles a subset on purpose.
            throw std::runtime_error("is_warp_level and step_warp disagree about " +
                                     std::string(opcode_name(instr.op)));
        }

        for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
            if (is_active(warp, lane)) {
                ++warp.threads[lane].pc;
            }
        }

        const uint64_t voting = active_lane_count(warp);
        stats_.warp_steps += 1;
        stats_.active_lane_ops += voting;
        stats_.weighted_lane_ops += voting * instruction_cost(instr.op);

        // true, where BARRIER returns false: a barrier takes the warp out of the
        // queue to wait, and these have nothing to wait for.
        return true;
    }

    // Skipping the masked lanes is what divergence costs: they are paid for by
    // the step and produce nothing. Advancing pc before execute() is what lets a
    // branch simply overwrite it.
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        if (!is_active(warp, lane)) {
            continue;
        }
        ++warp.threads[lane].pc;
        execute(program[warp.pc], warp.pc, warp.threads[lane], block, global);
    }

    // Once per step, outside the lane loop. Counting per lane would make a fully
    // converged step look like 32 separate ones and report zero divergence for
    // every program ever run.
    const uint64_t lanes = active_lane_count(warp);
    stats_.warp_steps += 1;
    stats_.active_lane_ops += lanes;

    // The one place a per-lane cost is not simply lanes x cost: what a global
    // access costs depends on where the lanes point, once the memory model
    // looks. Everything else is charged by opcode alone.
    const Instruction& issued = program[warp.pc];
    if (issued.op == Opcode::V_LD_GLOBAL_F32 ||
        issued.op == Opcode::V_LD_GLOBAL_VEC3_F32 ||
        issued.op == Opcode::V_ST_GLOBAL_F32) {
        const GlobalAccess access = global_access(warp, issued);
        stats_.weighted_lane_ops += access.cost;
        issued_latency_ = access.latency;
    } else {
        stats_.weighted_lane_ops += lanes * instruction_cost(issued.op);
        issued_latency_ = instruction_latency(issued.op);
    }
    return true;
}

void WarpScheduler::release_barrier(ThreadBlock& block)
{
    // Called only once the queue has run dry, and that is the whole arrival
    // test: if nobody can take a turn and somebody is waiting, then every warp
    // with work left has reached the barrier. No arrival counter to keep in
    // step with warp retirement — the queue already knows.
    for (Warp& warp : block.warps) {
        if (warp.at_barrier) {
            warp.at_barrier = false;
            ready_queue_.push(&warp);
        }
    }
}
// Round-robin among the warps that are ready, which is what makes this reproduce
// run() exactly when every latency is zero: nothing waits, so the same warp comes
// up in the same order. That equality is the test — see
// ModelledSchedulingMatchesIgnoredWhenNothingWaits.
//
// When nobody can issue, time moves to the soonest a warp will be ready and the
// gap is charged to stall_steps. Cycles skipped rather than stalls counted, so a
// hundred-cycle wait is not the same reading as a hundred one-cycle waits.
//
// A barrier is looked at only once no live warp is merely waiting. Releasing on
// "nobody ready" alone would open it while warps were still on their way — the
// arrival test has to be that no live warp can issue again without the release,
// and a warp with a pending result can.
void WarpScheduler::run_modelled(const Program& program, ThreadBlock& block,
                                 DeviceSpan global)
{
    const size_t warps = block.warps.size();

    // Retired is not the same as unavailable now, which is the distinction run()
    // does not need: its queue holds exactly the warps that can still run.
    std::vector<bool> live(warps, true);
    size_t remaining = warps;

    uint64_t now = 0;
    size_t cursor = 0;

    while (remaining > 0) {
        if (now > cycle_budget_) {
            throw std::runtime_error(
                "WarpScheduler::run: block did not finish within " +
                std::to_string(cycle_budget_) +
                " cycles — a lane is waiting on one this scheduler will never "
                "issue, or the kernel does not terminate");
        }

        // Ready: live, not waiting at a barrier, and its result has landed.
        size_t chosen = warps;
        for (size_t n = 0; n < warps; ++n) {
            const size_t i = (cursor + n) % warps;
            const Warp& w = block.warps[i];
            if (live[i] && !w.at_barrier && w.ready_at <= now) {
                chosen = i;
                break;
            }
        }

        if (chosen != warps) {
            Warp& warp = block.warps[chosen];
            cursor = (chosen + 1) % warps;

            // Whether an instruction actually went out, which is not the same as
            // whether the warp takes another turn: a warp discovered to have
            // retired issued nothing and costs no time, while one arriving at a
            // barrier issued the BARRIER and does.
            const uint64_t before = stats_.warp_steps;
            if (!step_warp(program, warp, block, global)) {
                // Retired, or now waiting at a barrier. at_barrier tells them
                // apart, and only the first is done for good.
                if (!warp.at_barrier) {
                    live[chosen] = false;
                    --remaining;
                }
            } else {
                // step_warp reports the latency of what it issued, which for a
                // global access comes from the memory model rather than from the
                // opcode alone.
                warp.ready_at = now + issued_latency_;
            }
            if (stats_.warp_steps != before) {
                ++now;
            }
            continue;
        }

        // Nobody ready. If any live warp is merely waiting, move time to it.
        uint64_t soonest = UINT64_MAX;
        for (size_t i = 0; i < warps; ++i) {
            const Warp& w = block.warps[i];
            if (live[i] && !w.at_barrier) {
                soonest = std::min(soonest, w.ready_at);
            }
        }
        if (soonest != UINT64_MAX) {
            stats_.stall_steps += soonest - now;
            now = soonest;
            continue;
        }

        // Every live warp is at a barrier, so all of them have arrived: nothing
        // can issue again without the release.
        bool released = false;
        for (Warp& warp : block.warps) {
            if (warp.at_barrier) {
                warp.at_barrier = false;
                released = true;
            }
        }
        if (!released) {
            break;
        }
    }

    stats_.cycles = now;
}

void WarpScheduler::run(const Program& program, ThreadBlock& block, DeviceSpan global)
{
    // A previous run() may have thrown partway and left pointers here that now
    // belong to a destroyed block.
    ready_queue_ = {};

    // One block's private cache. L2 is the device's and stays.
    l1_.clear();

    for (Warp& warp : block.warps) {
        warp.at_barrier = false;
        ready_queue_.push(&warp);
    }

    for (Warp& warp : block.warps) {
        warp.ready_at = 0;
    }

    if (latency_ == LatencyModel::Modelled) {
        run_modelled(program, block, global);
        return;
    }

    // A turn is a cycle here, nothing ever waiting. Counted per turn rather than
    // per warp, so the limit is on the block's total work and not on how the queue
    // happened to interleave.
    uint64_t steps = 0;
    while (!ready_queue_.empty()) {
        if (++steps > cycle_budget_) {
            throw std::runtime_error(
                "WarpScheduler::run: block did not finish within " +
                std::to_string(cycle_budget_) +
                " cycles — a lane is waiting on one this scheduler will never "
                "issue, or the kernel does not terminate");
        }

        Warp* warp = ready_queue_.front();
        ready_queue_.pop();
        if (step_warp(program, *warp, block, global)) {
            ready_queue_.push(warp);
        }

        // Equal to warp_steps while nothing waits, and its own counter so the two
        // can part without a rate quietly changing meaning.
        stats_.cycles = stats_.warp_steps;

        // A warp that retires before reaching the barrier stops being waited
        // for. Waiting for it instead would hang, and a simulator that hangs is
        // harder to debug than one that carries on.
        if (ready_queue_.empty()) {
            release_barrier(block);
        }
    }
}

void WarpScheduler::reset_stats()
{
    // Assigning a fresh value rather than zeroing each member, so that a counter
    // added later cannot be forgotten here.
    //
    // The caches are deliberately untouched. This runs between blocks, and
    // emptying L2 here would leave it holding nothing at any point a kernel could
    // observe — the level exists to outlive a block.
    stats_ = SchedulerStats{};
}
