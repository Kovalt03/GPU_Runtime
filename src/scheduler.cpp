#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

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
    }
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

    // Built from scratch: the previous step's mask says nothing about which
    // lanes have reached this instruction.
    warp.pc = min_pc;
    warp.active_mask = 0;
    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        if (warp.threads[lane].active && warp.threads[lane].pc == min_pc) {
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
    stats_.weighted_lane_ops += lanes * instruction_cost(program[warp.pc].op);
    return true;
}

void WarpScheduler::run(const Program& program, ThreadBlock& block, DeviceSpan global)
{
    // A previous run() may have thrown partway and left pointers here that now
    // belong to a destroyed block.
    ready_queue_ = {};

    for (Warp& warp : block.warps) {
        ready_queue_.push(&warp);
    }

    // One step per turn, not run-to-completion: that is what makes this
    // round-robin, and it is also how a real scheduler hides the latency of a
    // long-running instruction behind the other warps.
    while (!ready_queue_.empty()) {
        Warp* warp = ready_queue_.front();
        ready_queue_.pop();
        if (step_warp(program, *warp, block, global)) {
            ready_queue_.push(warp);
        }
    }
}

void WarpScheduler::reset_stats()
{
    // Assigning a fresh value rather than zeroing each member, so that a counter
    // added later cannot be forgotten here.
    stats_ = SchedulerStats{};
}
