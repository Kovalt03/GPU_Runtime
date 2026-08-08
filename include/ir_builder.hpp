#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "isa.hpp"

// An IR builder over the instruction set: the layer a real stack puts between a
// language and its machine code. Without it a kernel is written the way
// kernels/ray_triangle.cpp is — registers numbered by hand, branch offsets
// patched by hand, and nothing to stop a scalar being passed where a vector
// belongs.

// --- shapes -----------------------------------------------------------------
// A shape says how many consecutive registers a value occupies and where it may
// start. Making them distinct types is the point: V_DOT_VEC3_F32 then cannot be
// handed a scalar, and a float cannot be passed where a register index belongs.

struct Scalar {
    static constexpr uint32_t REGISTERS = 1;
    static constexpr uint32_t ALIGNMENT = 1;
};

struct Vec3 {
    static constexpr uint32_t REGISTERS = 3;
    static constexpr uint32_t ALIGNMENT = 1;
};

// VEC4 and wider are 4-aligned, matching what V_MATVEC_MAT4_F32 enforces.
struct Vec4 {
    static constexpr uint32_t REGISTERS = 4;
    static constexpr uint32_t ALIGNMENT = 4;
};

struct Mat4 {
    static constexpr uint32_t REGISTERS = 16;
    static constexpr uint32_t ALIGNMENT = 4;
};

// A typed handle to a register range. Deliberately not convertible to uint8_t:
// the whole point is that Reg<Vec3> and Reg<Scalar> cannot be interchanged, and
// that no raw number reaches an operand slot by accident.
template <typename Shape>
class Reg {
public:
    Reg() = default;

    explicit Reg(uint8_t first) : first_(first) {}

    uint8_t first() const
    {
        return first_;
    }

    // Component of a vector, as a scalar. The perspective divide needs clip.w
    // on its own.
    Reg<Scalar> component(uint32_t index) const;

    // The leading three registers, seen as a VEC3 — the other half of that
    // divide, which scales clip.xyz by 1/clip.w. Defined here rather than in
    // the .cpp so that asking a Scalar for one fails at the call site instead
    // of at link time; a builder whose point is good diagnostics should not
    // answer misuse with an undefined symbol.
    Reg<Vec3> xyz() const
    {
        static_assert(Shape::REGISTERS >= Vec3::REGISTERS,
                      "xyz() needs a value at least three registers wide");
        return Reg<Vec3>(first_);
    }

private:
    uint8_t first_ = 0;
};

// A branch target whose address is not known when the branch is emitted. Five
// separate tests in Möller-Trumbore jump to one miss path, so nesting if_ would
// mean nesting five deep; a label suits that shape.
class Label {
public:
    explicit Label(uint32_t id) : id_(id) {}

    uint32_t id() const
    {
        return id_;
    }

private:
    uint32_t id_ = 0;
};

class IRBuilder {
public:
    // --- allocation ---------------------------------------------------------
    // Bump allocation, with no release. The builder runs once per kernel: a
    // loop in the finished program is a backward branch, not a loop here, so
    // nothing is allocated twice. A real compiler would do liveness analysis
    // and spill; this does not, and throws when it runs out rather than
    // wrapping onto the thread coordinates.
    template <typename Shape>
    Reg<Shape> alloc();

    Reg<Scalar> scalar()
    {
        return alloc<Scalar>();
    }
    Reg<Vec3> vec3()
    {
        return alloc<Vec3>();
    }
    Reg<Vec4> vec4()
    {
        return alloc<Vec4>();
    }
    Reg<Mat4> mat4()
    {
        return alloc<Mat4>();
    }

    uint32_t registers_used() const
    {
        return next_free_;
    }

    // --- constants ----------------------------------------------------------
    // The only door a host value gets through, which is what makes passing one
    // into an operand slot impossible rather than merely discouraged.
    Reg<Scalar> constant(float value);
    Reg<Vec3> constant(float x, float y, float z);

    // --- thread coordinates -------------------------------------------------
    // Reserved by the launch, so the allocator never hands them out.
    Reg<Scalar> thread_x() const;
    Reg<Scalar> thread_y() const;
    Reg<Scalar> thread_z() const;

    // --- arithmetic ---------------------------------------------------------
    // Overloaded on shape, so V_SUB_F32 and V_SUB_VEC3_F32 are never chosen by
    // hand. Each returns a freshly allocated result.
    Reg<Scalar> add(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> sub(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> mul(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> rcp(Reg<Scalar> a);
    Reg<Scalar> sqrt(Reg<Scalar> a);
    Reg<Scalar> min(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> max(Reg<Scalar> a, Reg<Scalar> b);

    Reg<Vec3> add(Reg<Vec3> a, Reg<Vec3> b);
    Reg<Vec3> sub(Reg<Vec3> a, Reg<Vec3> b);
    Reg<Vec3> scale(Reg<Vec3> v, Reg<Scalar> s);
    Reg<Scalar> dot(Reg<Vec3> a, Reg<Vec3> b);
    Reg<Vec3> cross(Reg<Vec3> a, Reg<Vec3> b);
    Reg<Vec3> normalize(Reg<Vec3> v);

    Reg<Vec4> transform(Reg<Mat4> m, Reg<Vec4> v);

    // Accumulates in place, matching V_FMA_F32: acc += a * b.
    void fma(Reg<Scalar> acc, Reg<Scalar> a, Reg<Scalar> b);

    // Writes a value into a register that already exists, rather than
    // allocating a fresh one. Assembling a VEC4 needs exactly this: the w
    // component has to land in the fourth register of a range whose first three
    // came from somewhere else, and no allocating call can put a value at a
    // chosen index.
    void set(Reg<Scalar> dst, float value);

    // Moves one register into another that already exists. The third of the
    // family, alongside set (an immediate) and load_into (memory), and the one
    // a branch needs: both arms of an if_else have to leave their result in the
    // same place, and every allocating call hands back somewhere new.
    void copy_into(Reg<Scalar> dst, Reg<Scalar> src);

    // Copies a register. V_MOV_F32 takes an immediate, so this goes through
    // arithmetic — adding zero is the idiom.
    Reg<Scalar> copy(Reg<Scalar> a);
    Reg<Vec3> copy(Reg<Vec3> a);

    // --- comparison ---------------------------------------------------------
    // Results are 1.0 or 0.0, which is what BRA_DIV consumes.
    Reg<Scalar> compare(Reg<Scalar> a, Reg<Scalar> b, CmpOp op);
    Reg<Scalar> lt(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> gt(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> ge(Reg<Scalar> a, Reg<Scalar> b);
    Reg<Scalar> le(Reg<Scalar> a, Reg<Scalar> b);

    // Logical OR of two flags. Adding them works because each is 1.0 or 0.0 and
    // BRA_DIV only asks whether the result is non-zero — one branch instead of
    // two, and so one divergence point instead of two.
    Reg<Scalar> any(Reg<Scalar> a, Reg<Scalar> b);

    // --- memory -------------------------------------------------------------
    // Load takes (address), store takes (address, value). Getting them the
    // wrong way round is a mistake the raw factories allow and this does not,
    // the address being an argument and the value a return.
    Reg<Scalar> load(Reg<Scalar> address, float offset = 0.0f);
    Reg<Vec3> load_vec3(Reg<Scalar> address, float offset = 0.0f);

    // As load, into a register the caller already holds — the counterpart to
    // set(), and what fills the first three slots of a VEC4 in place.
    void load_into(Reg<Scalar> dst, Reg<Scalar> address, float offset = 0.0f);
    void store(Reg<Scalar> address, Reg<Scalar> value, float offset = 0.0f);
    void store_vec3(Reg<Scalar> address, Reg<Vec3> value, float offset = 0.0f);

    // --- control flow -------------------------------------------------------
    Label label();
    void place(Label target);
    void branch_to(Label target);                        // unconditional
    void branch_to(Label target, Reg<Scalar> when_set);  // when the flag is not 0

    // Structured forms, for the cases a label would only clutter. BRA_DIV can
    // only jump when a value is non-zero, so skipping a body costs one extra
    // instruction to invert the condition; that is hidden here.
    void if_(Reg<Scalar> cond, const std::function<void()>& body);
    void if_else(Reg<Scalar> cond, const std::function<void()>& then_body,
                 const std::function<void()>& else_body);

    // --- output -------------------------------------------------------------
    // Appends RET and resolves every label. Throws if a label was branched to
    // but never placed.
    Program build();

private:
    Program program_;
    uint32_t next_free_ = 0;

    // Where each label ended up, and every branch still waiting for it.
    std::vector<uint32_t> label_targets_;
    std::vector<std::pair<uint32_t, size_t>> pending_branches_;

    void emit(Instruction instr);
};
