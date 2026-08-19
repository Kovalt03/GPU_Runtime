#include <limits>
#include <stdexcept>
#include <string>

#include "ir_builder.hpp"
#include "runtime.hpp"  // REG_GLOBAL_ID_X/_Y/_Z, which the builder reserves

namespace {

// The launch writes the thread's and block's coordinates from here upward, so
// the allocator stops short rather than handing one out and having a kernel
// quietly overwrite its own identity.
constexpr uint32_t FIRST_RESERVED_REGISTER = REG_BLOCK_ID_X;

// A label that has been handed out but not yet placed. Distinct from address 0,
// which is a perfectly ordinary target for a backward branch.
constexpr uint32_t LABEL_UNPLACED = std::numeric_limits<uint32_t>::max();

// Bytes per float in device memory, which is what a vector load has to step by
// between components.
constexpr float COMPONENT_STRIDE = 4.0f;

}  // namespace

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

template <typename Shape>
Reg<Shape> IRBuilder::alloc()
{
    // Running out throws rather than wrapping: clamping would hand back a
    // register that overlaps the thread coordinates, and a kernel reading its
    // own id as a triangle vertex is not a failure anyone traces.
    const uint32_t alignment = Shape::ALIGNMENT;
    const uint32_t first = (next_free_ + alignment - 1) / alignment * alignment;

    if (first + Shape::REGISTERS > FIRST_RESERVED_REGISTER) {
        throw std::runtime_error("IRBuilder: out of registers — asked for " +
                                 std::to_string(Shape::REGISTERS) + " at r" +
                                 std::to_string(first) + ", but r" +
                                 std::to_string(FIRST_RESERVED_REGISTER) +
                                 " upward carries the thread coordinates");
    }

    next_free_ = first + Shape::REGISTERS;
    return Reg<Shape>(static_cast<uint8_t>(first));
}

// Only these instantiations exist, so the definition can stay in the .cpp.
template Reg<Scalar> IRBuilder::alloc<Scalar>();
template Reg<Vec3> IRBuilder::alloc<Vec3>();
template Reg<Vec4> IRBuilder::alloc<Vec4>();
template Reg<Mat4> IRBuilder::alloc<Mat4>();
template Reg<Frag> IRBuilder::alloc<Frag>();
template Reg<HalfFrag> IRBuilder::alloc<HalfFrag>();

template <typename Shape>
Reg<Scalar> Reg<Shape>::component(uint32_t index) const
{
    // What lets a VEC3 operation read the leading three registers of a VEC4:
    // the perspective divide needs clip.xyz scaled by 1/clip.w, and both views
    // are just offsets into the same range.
    if (index >= Shape::REGISTERS) {
        throw std::runtime_error("Reg::component: index " + std::to_string(index) +
                                 " is past the " + std::to_string(Shape::REGISTERS) +
                                 " registers this value occupies");
    }
    return Reg<Scalar>(static_cast<uint8_t>(first_ + index));
}

template Reg<Scalar> Reg<Scalar>::component(uint32_t) const;
template Reg<Scalar> Reg<Vec3>::component(uint32_t) const;
template Reg<Scalar> Reg<Vec4>::component(uint32_t) const;
template Reg<Scalar> Reg<Mat4>::component(uint32_t) const;
template Reg<Scalar> Reg<Frag>::component(uint32_t) const;
template Reg<Scalar> Reg<HalfFrag>::component(uint32_t) const;

void IRBuilder::emit(Instruction instr)
{
    program_.push_back(instr);
}

// ---------------------------------------------------------------------------
// Arithmetic
//
// Each allocates its result and emits one instruction. The overloads are what
// remove the choice between V_SUB_F32 and V_SUB_VEC3_F32 from the caller.
// ---------------------------------------------------------------------------

Reg<Scalar> IRBuilder::add(Reg<Scalar> a, Reg<Scalar> b)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_add_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Scalar> IRBuilder::sub(Reg<Scalar> a, Reg<Scalar> b)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_sub_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Scalar> IRBuilder::mul(Reg<Scalar> a, Reg<Scalar> b)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_mul_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Scalar> IRBuilder::rcp(Reg<Scalar> a)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_rcp_f32(dst.first(), a.first()));
    return dst;
}

Reg<Scalar> IRBuilder::sqrt(Reg<Scalar> a)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_sqrt_f32(dst.first(), a.first()));
    return dst;
}

Reg<Scalar> IRBuilder::min(Reg<Scalar> a, Reg<Scalar> b)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_min_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Scalar> IRBuilder::max(Reg<Scalar> a, Reg<Scalar> b)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_max_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Vec3> IRBuilder::add(Reg<Vec3> a, Reg<Vec3> b)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_add_vec3_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Vec3> IRBuilder::sub(Reg<Vec3> a, Reg<Vec3> b)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_sub_vec3_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Vec3> IRBuilder::scale(Reg<Vec3> v, Reg<Scalar> s)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_scale_vec3_f32(dst.first(), v.first(), s.first()));
    return dst;
}

Reg<Scalar> IRBuilder::dot(Reg<Vec3> a, Reg<Vec3> b)
{
    // A scalar destination for a vector operation, which is exactly the pairing
    // the shape types exist to keep straight.
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_dot_vec3_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Vec3> IRBuilder::cross(Reg<Vec3> a, Reg<Vec3> b)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_cross_vec3_f32(dst.first(), a.first(), b.first()));
    return dst;
}

Reg<Vec3> IRBuilder::normalize(Reg<Vec3> v)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_norm_vec3_f32(dst.first(), v.first()));
    return dst;
}

Reg<Vec4> IRBuilder::transform(Reg<Mat4> m, Reg<Vec4> v)
{
    // Every operand of V_MATVEC_MAT4_F32 has to be 4-aligned, which alloc<Vec4>
    // and alloc<Mat4> already guarantee — the reason alignment lives in the
    // shape rather than in each caller.
    const Reg<Vec4> dst = alloc<Vec4>();
    emit(make_v_matvec_mat4_f32(dst.first(), m.first(), v.first()));
    return dst;
}

Reg<Mat4> IRBuilder::compose(Reg<Mat4> a, Reg<Mat4> b)
{
    const Reg<Mat4> dst = alloc<Mat4>();
    emit(make_v_matmul_mat4_f32(dst.first(), a.first(), b.first()));
    return dst;
}

void IRBuilder::fma(Reg<Scalar> acc, Reg<Scalar> a, Reg<Scalar> b)
{
    // Accumulates into acc rather than allocating, matching V_FMA_F32.
    emit(make_v_fma_f32(acc.first(), a.first(), b.first()));
}

void IRBuilder::set(Reg<Scalar> dst, float value)
{
    emit(make_v_mov_f32(dst.first(), value));
}

void IRBuilder::copy_into(Reg<Scalar> dst, Reg<Scalar> src)
{
    // V_MOV_F32 only carries an immediate, so a register-to-register move has
    // to go through arithmetic. max(x, x) is x for every value it can hold, and
    // unlike adding a zero it needs no second register to have been set up.
    emit(make_v_max_f32(dst.first(), src.first(), src.first()));
}

Reg<Scalar> IRBuilder::copy(Reg<Scalar> a)
{
    // V_MOV_F32 takes an immediate, so adding a zero register is the idiom.
    //
    // The zero is emitted here rather than shared across calls: a cached one
    // would be written wherever it happened to be created, and a caller that
    // first copies inside an if_ body would leave every later copy reading a
    // register the branch skipped.
    const Reg<Scalar> zero = constant(0.0f);
    return add(a, zero);
}

Reg<Vec3> IRBuilder::copy(Reg<Vec3> a)
{
    const Reg<Vec3> zero = constant(0.0f, 0.0f, 0.0f);
    return add(a, zero);
}

// ---------------------------------------------------------------------------
// Constants and thread coordinates
// ---------------------------------------------------------------------------

Reg<Scalar> IRBuilder::constant(float value)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_mov_f32(dst.first(), value));
    return dst;
}

Reg<Vec3> IRBuilder::constant(float x, float y, float z)
{
    // Three moves rather than one: V_MOV_F32 carries a single immediate, and an
    // instruction is 8 bytes with room for exactly one.
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_mov_f32(static_cast<uint8_t>(dst.first() + 0), x));
    emit(make_v_mov_f32(static_cast<uint8_t>(dst.first() + 1), y));
    emit(make_v_mov_f32(static_cast<uint8_t>(dst.first() + 2), z));
    return dst;
}

Reg<Scalar> IRBuilder::thread_x() const
{
    return Reg<Scalar>(REG_GLOBAL_ID_X);
}

Reg<Scalar> IRBuilder::thread_y() const
{
    return Reg<Scalar>(REG_GLOBAL_ID_Y);
}

Reg<Scalar> IRBuilder::thread_z() const
{
    return Reg<Scalar>(REG_GLOBAL_ID_Z);
}

Reg<Scalar> IRBuilder::cluster_rank() const
{
    return Reg<Scalar>(REG_CLUSTER_RANK);
}

Reg<Scalar> IRBuilder::const_base() const
{
    return Reg<Scalar>(REG_CONST_BASE);
}

Reg<Scalar> IRBuilder::block_x() const
{
    return Reg<Scalar>(REG_BLOCK_ID_X);
}

Reg<Scalar> IRBuilder::block_y() const
{
    return Reg<Scalar>(REG_BLOCK_ID_Y);
}

Reg<Scalar> IRBuilder::block_z() const
{
    return Reg<Scalar>(REG_BLOCK_ID_Z);
}

// ---------------------------------------------------------------------------
// Comparison and memory
// ---------------------------------------------------------------------------

Reg<Scalar> IRBuilder::compare(Reg<Scalar> a, Reg<Scalar> b, CmpOp op)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_cmp_f32(dst.first(), a.first(), b.first(), op));
    return dst;
}

Reg<Scalar> IRBuilder::lt(Reg<Scalar> a, Reg<Scalar> b)
{
    return compare(a, b, CmpOp::LT);
}

Reg<Scalar> IRBuilder::gt(Reg<Scalar> a, Reg<Scalar> b)
{
    return compare(a, b, CmpOp::GT);
}

Reg<Scalar> IRBuilder::ge(Reg<Scalar> a, Reg<Scalar> b)
{
    return compare(a, b, CmpOp::GE);
}

Reg<Scalar> IRBuilder::le(Reg<Scalar> a, Reg<Scalar> b)
{
    return compare(a, b, CmpOp::LE);
}

Reg<Scalar> IRBuilder::any(Reg<Scalar> a, Reg<Scalar> b)
{
    // Both operands are 1.0 or 0.0 and BRA_DIV only asks whether a value is
    // non-zero, so a sum answers "either" without a second branch.
    return add(a, b);
}

Reg<Scalar> IRBuilder::load(Reg<Scalar> address, float offset)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_ld_global_f32(dst.first(), address.first(), offset));
    return dst;
}

void IRBuilder::load_into(Reg<Scalar> dst, Reg<Scalar> address, float offset)
{
    emit(make_v_ld_global_f32(dst.first(), address.first(), offset));
}

void IRBuilder::load_shared_into(Reg<Scalar> dst, Reg<Scalar> address, float offset)
{
    emit(make_v_ld_shared_f32(dst.first(), address.first(), offset));
}

Reg<Frag> IRBuilder::load_shared_fragment(Reg<Scalar> address, float offset)
{
    const Reg<Frag> out = alloc<Frag>();
    emit(make_v_ld_shared_16x16_f32(out.first(), address.first(), offset));
    return out;
}

Reg<Scalar> IRBuilder::load_const(Reg<Scalar> address, float offset)
{
    const Reg<Scalar> out = alloc<Scalar>();
    emit(make_v_ld_const_f32(out.first(), address.first(), offset));
    return out;
}

Reg<Mat4> IRBuilder::load_const_mat4(Reg<Scalar> address, float offset)
{
    const Reg<Mat4> out = alloc<Mat4>();
    emit(make_v_ld_const_mat4_f32(out.first(), address.first(), offset));
    return out;
}

Reg<HalfFrag> IRBuilder::load_shared_half_fragment(Reg<Scalar> address, float offset)
{
    const Reg<HalfFrag> out = alloc<HalfFrag>();
    emit(make_v_ld_shared_16x16_f16(out.first(), address.first(), offset));
    return out;
}

Reg<Vec3> IRBuilder::load_vec3(Reg<Scalar> address, float offset)
{
    const Reg<Vec3> dst = alloc<Vec3>();
    emit(make_v_ld_global_vec3_f32(dst.first(), address.first(), offset));
    return dst;
}

void IRBuilder::load_vec3_into(Reg<Vec3> dst, Reg<Scalar> address, float offset)
{
    emit(make_v_ld_global_vec3_f32(dst.first(), address.first(), offset));
}

Reg<Scalar> IRBuilder::load_shared(Reg<Scalar> address, float offset)
{
    const Reg<Scalar> dst = alloc<Scalar>();
    emit(make_v_ld_shared_f32(dst.first(), address.first(), offset));
    return dst;
}

void IRBuilder::store_shared(Reg<Scalar> address, Reg<Scalar> value, float offset)
{
    emit(make_v_st_shared_f32(address.first(), value.first(), offset));
}

void IRBuilder::store(Reg<Scalar> address, Reg<Scalar> value, float offset)
{
    emit(make_v_st_global_f32(address.first(), value.first(), offset));
}

Reg<Scalar> IRBuilder::atomic_add(Reg<Scalar> address, Reg<Scalar> value, float offset)
{
    const Reg<Scalar> old = alloc<Scalar>();
    emit(make_v_atom_add_global_f32(old.first(), address.first(), value.first(), offset));
    return old;
}

void IRBuilder::cp_async(Reg<Scalar> shared_address, Reg<Scalar> global_address,
                         float offset)
{
    emit(make_v_cp_async_shared_global_f32(shared_address.first(), global_address.first(),
                                           offset));
}

void IRBuilder::cp_async_wait(uint32_t outstanding)
{
    emit(make_s_cp_async_wait(outstanding));
}

void IRBuilder::store_vec3(Reg<Scalar> address, Reg<Vec3> value, float offset)
{
    for (uint32_t i = 0; i < Vec3::REGISTERS; ++i) {
        emit(make_v_st_global_f32(address.first(),
                                  static_cast<uint8_t>(value.first() + i),
                                  offset + static_cast<float>(i) * COMPONENT_STRIDE));
    }
}

// ---------------------------------------------------------------------------
// Control flow
//
// A branch is emitted with a placeholder offset and recorded; place() fixes the
// target, and build() patches every branch that was waiting on it. Forward and
// backward jumps go through the same path, a backward one simply being placed
// before the branch that reaches it.
// ---------------------------------------------------------------------------

void IRBuilder::barrier()
{
    emit(make_barrier());
}

void IRBuilder::reorder(Reg<Scalar> key)
{
    emit(make_reorder(key.first()));
}

void IRBuilder::barrier_cluster()
{
    emit(make_barrier_cluster());
}

void IRBuilder::mma(Reg<Frag> accumulator, Reg<Frag> a, Reg<Frag> b)
{
    // Every lane of the warp, always: the shape needs all of them, and the
    // scheduler refuses a mask that names fewer.
    emit(make_v_mma_16x16x16_f32(accumulator.first(), a.first(), b.first(), 0xFFFFFFFFu));
}

void IRBuilder::mma(Reg<Frag> accumulator, Reg<HalfFrag> a, Reg<HalfFrag> b)
{
    emit(make_v_mma_16x16x16_f16(accumulator.first(), a.first(), b.first(), 0xFFFFFFFFu));
}

Reg<Scalar> IRBuilder::load_cluster(Reg<Scalar> address, Reg<Scalar> rank, float offset)
{
    const Reg<Scalar> out = alloc<Scalar>();
    emit(make_v_ld_cluster_f32(out.first(), address.first(), rank.first(), offset));
    return out;
}

Label IRBuilder::label()
{
    const uint32_t id = static_cast<uint32_t>(label_targets_.size());
    label_targets_.push_back(LABEL_UNPLACED);
    return Label(id);
}

void IRBuilder::place(Label target)
{
    // Placing a label twice is reported rather than silently keeping the last
    // one: the branches already waiting on it would land somewhere the kernel
    // never named.
    if (target.id() >= label_targets_.size()) {
        throw std::runtime_error("IRBuilder::place: label " +
                                 std::to_string(target.id()) +
                                 " came from another builder");
    }
    if (label_targets_[target.id()] != LABEL_UNPLACED) {
        throw std::runtime_error("IRBuilder::place: label " +
                                 std::to_string(target.id()) + " is already placed at " +
                                 std::to_string(label_targets_[target.id()]));
    }
    label_targets_[target.id()] = static_cast<uint32_t>(program_.size());
}

void IRBuilder::branch_to(Label target)
{
    pending_branches_.emplace_back(target.id(), program_.size());
    emit(make_bra(0));
}

void IRBuilder::branch_to(Label target, Reg<Scalar> when_set)
{
    pending_branches_.emplace_back(target.id(), program_.size());
    emit(make_bra_div(when_set.first(), 0));
}

// ---------------------------------------------------------------------------
// Structured control flow
//
// Built on the labels above, and needing the condition inverted: BRA_DIV jumps
// when a value is non-zero, but skipping a body means jumping when it is false.
// ---------------------------------------------------------------------------

void IRBuilder::if_(Reg<Scalar> cond, const std::function<void()>& body)
{
    const Reg<Scalar> zero = constant(0.0f);
    const Reg<Scalar> skip = compare(cond, zero, CmpOp::EQ);

    const Label end = label();
    branch_to(end, skip);
    body();
    place(end);
}

void IRBuilder::if_else(Reg<Scalar> cond, const std::function<void()>& then_body,
                        const std::function<void()>& else_body)
{
    // As if_, plus an unconditional branch over the else body — the shape
    // kernels/ray_triangle.cpp lays out by hand.
    const Reg<Scalar> zero = constant(0.0f);
    const Reg<Scalar> skip = compare(cond, zero, CmpOp::EQ);

    const Label otherwise = label();
    const Label end = label();

    branch_to(otherwise, skip);
    then_body();
    branch_to(end);

    place(otherwise);
    else_body();
    place(end);
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

Program IRBuilder::build()
{
    // A label that was branched to but never placed is a bug in the kernel, not
    // something to paper over with a jump to zero.
    emit(make_ret());

    for (const std::pair<uint32_t, size_t>& pending : pending_branches_) {
        const uint32_t label_id = pending.first;
        const size_t branch_at = pending.second;

        if (label_targets_[label_id] == LABEL_UNPLACED) {
            throw std::runtime_error("IRBuilder::build: label " +
                                     std::to_string(label_id) +
                                     " is branched to from instruction " +
                                     std::to_string(branch_at) + " but never placed");
        }

        // The scheduler computes its target as instr_pc + offset, taking the
        // branch's own address as the origin.
        const int64_t offset = static_cast<int64_t>(label_targets_[label_id]) -
                               static_cast<int64_t>(branch_at);

        // Rebuilt rather than patched in place so that the range check in
        // make_bra / make_bra_div covers a generated offset too.
        Instruction& branch = program_[branch_at];
        if (branch.op == Opcode::BRA) {
            branch = make_bra(static_cast<int32_t>(offset));
        } else {
            branch = make_bra_div(branch.src0, static_cast<int32_t>(offset));
        }
    }

    pending_branches_.clear();
    return program_;
}
