#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "isa.hpp"
#include "thread.hpp"

// An IR builder over the instruction set: the layer a real stack puts between a
// language and its machine code. Without it a kernel is written the way
// apps/ray_triangle.cpp is — registers numbered by hand, branch offsets
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

// A lane's eighth of a 16x16 tile, which is what V_MMA_16X16X16_F32 reads and
// writes. Eight registers because a warp is 32 lanes and a tile is 256 elements —
// the shape is the instruction's, not a choice.
struct Frag {
    static constexpr uint32_t REGISTERS = MMA_FRAGMENT_REGISTERS;
    static constexpr uint32_t ALIGNMENT = 4;
};

// The same fragment with two halves to a register, so half as many of them.
struct HalfFrag {
    static constexpr uint32_t REGISTERS = HALF_FRAGMENT_REGISTERS;
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

// A branch target whose address is not known when the branch is emitted. Four
// separate tests in Möller-Trumbore leave for one miss path, so nesting if_
// would mean nesting four deep; a label suits that shape.
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
    Reg<Frag> fragment()
    {
        return alloc<Frag>();
    }
    Reg<HalfFrag> half_fragment()
    {
        return alloc<HalfFrag>();
    }

    uint32_t registers_used() const
    {
        return next_free_;
    }

    // Hands back everything allocated inside it.
    //
    // The allocator is a bump pointer with no liveness analysis, so every
    // expression a kernel writes costs a register for the whole program. That is
    // fine until a kernel is long: two levels of tree traversal ran out at 250,
    // and most of what it had allocated was slab-test arithmetic that had been
    // dead for a hundred instructions.
    //
    // **A value made inside a scope must not be read after it.** The next
    // allocation gets those registers, and nothing checks. So this is for a
    // computation that ends by writing into a register declared outside it —
    // which is the shape of a slab test, and why the two here are wrapped and
    // nothing else is.
    class Scratch {
    public:
        explicit Scratch(IRBuilder& builder)
            : builder_(builder), mark_(builder.next_free_)
        {
        }

        ~Scratch()
        {
            builder_.next_free_ = mark_;
        }

        Scratch(const Scratch&) = delete;
        Scratch& operator=(const Scratch&) = delete;

    private:
        IRBuilder& builder_;
        uint32_t mark_;
    };

    // --- constants ----------------------------------------------------------
    // The only door a host value gets through, which is what makes passing one
    // into an operand slot impossible rather than merely discouraged.
    Reg<Scalar> constant(float value);
    Reg<Vec3> constant(float x, float y, float z);

    // --- thread and block coordinates ---------------------------------------
    // Reserved by the launch, so the allocator never hands them out.
    Reg<Scalar> thread_x() const;
    Reg<Scalar> thread_y() const;
    Reg<Scalar> thread_z() const;

    // Which block this thread runs in. Needed by anything a block owns
    // collectively, such as the tile of the screen it covers — a global
    // coordinate cannot be divided back down to it.
    Reg<Scalar> cluster_rank() const;

    // Where this launch's uniforms start. Every thread holds the same number, so
    // an address built from it is warp-uniform — which is what the constant
    // window's pricing rests on.
    Reg<Scalar> const_base() const;
    Reg<Scalar> block_x() const;
    Reg<Scalar> block_y() const;
    Reg<Scalar> block_z() const;

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

    // a * b, both MAT4. What makes composing a transform once cheaper than
    // applying two at every vertex — and what decides where the crossing is,
    // since this costs four of the transform above.
    Reg<Mat4> compose(Reg<Mat4> a, Reg<Mat4> b);

    // accumulator += a * b, over a 16x16x16 tile, by the whole warp. In place,
    // which is what lets a K loop carry its answer in registers.
    void mma(Reg<Frag> accumulator, Reg<Frag> a, Reg<Frag> b);

    // Half-precision operands, single-precision accumulator. The types make the
    // pairing impossible to get wrong: a Frag cannot be passed where a HalfFrag
    // belongs, and the accumulator is a Frag in both.
    void mma(Reg<Frag> accumulator, Reg<HalfFrag> a, Reg<HalfFrag> b);

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

    // One instruction, not three: a warp asking for twelve bytes at one address
    // pays for the lines they fall in once. Prefer it wherever three floats are
    // wanted together, which is every vertex these kernels read.
    Reg<Vec3> load_vec3(Reg<Scalar> address, float offset = 0.0f);

    // As load, into a register the caller already holds — the counterpart to
    // set(), and what fills the first three slots of a VEC4 in place.
    void load_into(Reg<Scalar> dst, Reg<Scalar> address, float offset = 0.0f);

    // The same for shared memory. A fragment is assembled a register at a time,
    // and each has to land in a chosen one — an allocating load would hand back
    // eight registers that are not consecutive.
    void load_shared_into(Reg<Scalar> dst, Reg<Scalar> address, float offset = 0.0f);

    // A whole fragment in one instruction, where the eight above are eight round
    // trips. The registers come back consecutive, which is what the matrix
    // instruction needs and what eight allocating loads could not promise.
    Reg<Frag> load_shared_fragment(Reg<Scalar> address, float offset = 0.0f);

    // A uniform, and a matrix of them. Answered once for the warp and broadcast.
    Reg<Scalar> load_const(Reg<Scalar> address, float offset = 0.0f);
    Reg<Mat4> load_const_mat4(Reg<Scalar> address, float offset = 0.0f);

    // A matrix from global memory, at an address that may differ by lane. What
    // the constant window cannot do: its charge rests on every lane wanting the
    // same one, and a bone palette or an instance list indexed per thread does
    // not. Sixteen loads' worth under a flat charge, one transaction's worth
    // under a coalesced one.
    Reg<Mat4> load_mat4(Reg<Scalar> address, float offset = 0.0f);

    // The same in half precision: four registers of packed halves, which nothing
    // but the F16 instructions may read.
    Reg<HalfFrag> load_shared_half_fragment(Reg<Scalar> address, float offset = 0.0f);
    void load_vec3_into(Reg<Vec3> dst, Reg<Scalar> address, float offset = 0.0f);

    // The block's own scratchpad. Addresses are byte offsets, as the global
    // pair uses, so the two read alike at a call site.
    //
    // Nothing above says "global" because nothing else existed when they were
    // written. The asymmetry is a wart; renaming them would touch every kernel
    // and is not worth doing in the change that adds these.
    Reg<Scalar> load_shared(Reg<Scalar> address, float offset = 0.0f);
    void store_shared(Reg<Scalar> address, Reg<Scalar> value, float offset = 0.0f);
    void store(Reg<Scalar> address, Reg<Scalar> value, float offset = 0.0f);
    void store_vec3(Reg<Scalar> address, Reg<Vec3> value, float offset = 0.0f);

    // Adds to one address indivisibly and hands back what was there before.
    //
    // The return is the point as much as the addition: 32 lanes adding one to a
    // counter get 32 different numbers back, and that is how a compaction pass
    // hands each surviving item a slot of its own. Lanes naming the same address
    // are served one at a time, which is what a warp reduction exists to avoid.
    Reg<Scalar> atomic_add(Reg<Scalar> address, Reg<Scalar> value, float offset = 0.0f);

    // Global to shared without a register in between, and without waiting.
    //
    // Arguments read as the assignment does — destination, then source — and
    // the offset belongs to the source, as it does for load(). What it replaces
    // is store_shared(dst, load(src)): two instructions, a register, and a warp
    // that waits out the load before it can issue the store.
    //
    // Nothing may read the destination until cp_async_wait has been passed, and
    // the scheduler refuses a read that comes early rather than answering it.
    void cp_async(Reg<Scalar> shared_address, Reg<Scalar> global_address,
                  float offset = 0.0f);

    // Wait until at most `outstanding` copies are still in flight. Zero waits
    // for all of them; one leaves the most recent, which is how a kernel keeps
    // working on what it has while the next arrives.
    void cp_async_wait(uint32_t outstanding);

    // --- control flow -------------------------------------------------------
    Label label();
    void place(Label target);
    void branch_to(Label target);                        // unconditional
    void branch_to(Label target, Reg<Scalar> when_set);  // when the flag is not 0

    // Must sit outside any if_: every thread of the block has to reach it, and
    // the scheduler throws when one does not.
    void barrier();

    // One level wider: every warp of every block of the cluster. What makes a
    // block's writes safe for its neighbours to read.
    void barrier_cluster();

    // A float from another block of the cluster, by its rank. Rank 0 with no
    // cluster is this block's own shared memory.
    Reg<Scalar> load_cluster(Reg<Scalar> address, Reg<Scalar> rank, float offset = 0.0f);

    // Regroups the block's threads so that equal keys share a warp. A rendezvous
    // like barrier(), and under the same rule about divergent control flow.
    //
    // Nothing a thread holds changes, so a kernel with one of these computes what
    // it computed without it — what changes is which lanes disagree, and so what
    // the divergence costs.
    void reorder(Reg<Scalar> key);

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
