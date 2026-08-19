#pragma once

#include <cstdint>
#include <vector>

#include "bvh.hpp"
#include "isa.hpp"
#include "math3d.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"

// --- ray tracing ------------------------------------------------------------
// The same scene by the other route, and the comparison the project was built
// for: two renderers over one image, diverging for different reasons.
//
// One launch, not two. Rays are cast in world space and the triangles are
// already there, so nothing has to be projected first — the vertex stage has no
// counterpart here. That is the structural difference between the two paths,
// and the reason the ray tracer needs no matrix at all.

// Three vertices in world space. A separate type from ScreenTriangle for the
// same reason Float3 is separate from Reg<Vec3>: the two hold identical numbers
// and mixing them is the mistake worth making impossible.
struct WorldTriangle {
    Float3 v0;
    Float3 v1;
    Float3 v2;
};

// The camera as the kernel wants it. right and up arrive already scaled by the
// field of view and the aspect, so a ray is
//
//   direction = right * sx + up * sy + forward
//
// for sx and sy in [-1, 1] — five instructions and no matrix. A matrix would
// cost about the same to apply and sixteen moves to set up, and would make the
// one path that needs no transform carry one anyway.
struct RayBasis {
    Float3 origin;
    Float3 right;
    Float3 up;
    Float3 forward;
};

RayBasis ray_basis(const Camera& camera, float aspect);

// The tolerance both intersection tests use, the host one and the kernel. One
// symbol rather than a literal in each, because the two are compared pixel for
// pixel and an edge lands on a different side the moment they drift apart.
//
// Not an argument: it is a property of the test, not of a scene, and passing it
// in would just move the chance of two callers disagreeing one level up.
//
// It does two jobs. Against the determinant it rejects a ray parallel to the
// triangle — and, being a signed comparison rather than an absolute one, a back
// face along with it. Against t it keeps a hit behind the origin from counting.
inline constexpr float INTERSECT_EPSILON = 1e-6f;

// How the kernel finds the triangles a ray might meet.
//
// Same launch, same thread meaning, same intersection test — what differs is
// only which triangles reach it. That is what makes this a flag rather than a
// second kernel, the way `predicated` and `indexed` are flags.
enum class Traversal {
    // Every triangle for every pixel. Cost linear in the scene, and the shape
    // this route had before there was anything better.
    Linear,

    // A bounding volume hierarchy, walked with a stack in shared memory.
    Bvh,

    // Two levels: a tree over instances in world space, and at each of its
    // leaves the ray moves into that instance's space and walks the tree over
    // the geometry there.
    //
    // The ray is transformed rather than the geometry, which is what makes one
    // tree serve every copy. The parameter t survives the move — a hit at t in
    // object space is the hit at t in world space, the direction being carried
    // as a vector and left unnormalised — so the running best still compares
    // across instances without being brought back.
    Tlas,
};

// Which child of an interior node is entered first.
//
// Only meaningful under Traversal::Bvh, and a flag because it is a trade rather
// than an improvement: entering the nearer child first lets the running best cut
// the far subtree off, and finding out which is nearer costs two slab tests at
// every interior node whether or not the cut ever happens.
enum class TraversalOrder {
    // Left then right, as the build laid them out. The nearest hit is still
    // correct — every node is tested — and the pruning is whatever the order
    // happens to give.
    Unordered,

    // The nearer child first, chosen branchlessly: a comparison yields 1.0 or
    // 0.0, and adding it to the left index names one child or the other without
    // splitting the warp to do it.
    Nearest,
};

// When the winning hit is coloured.
//
// A ray tracer that shades inside its loop shades every candidate and blends the
// loser away, which costs nothing to write and is what this route did from the
// start. It also makes reordering impossible: REORDER is a rendezvous, so it can
// only run where every thread of the block has finished traversing, and by then
// the shading has already happened.
//
// Deferring keeps four scalars — the two barycentrics, the parameter and the
// material — and colours once at the end. Which is what the hardware asks for
// too: trace, then optixReorder(material), then shade.
enum class ShadeWhen {
    // At every candidate that beats the running best.
    Inline,

    // Once, after the whole traversal, from what the loop kept.
    Deferred,

    // The same, with the block's threads regrouped by material first, so lanes
    // about to run the same shader share a warp.
    //
    // Held apart from Deferred rather than folded into it because the two
    // changes have to be separable: one moves the shading and the other sorts
    // the threads, and a single flag would report their sum as though it were
    // the reorder's doing.
    DeferredReordered,
};

struct RaytraceStageArgs {
    RayBasis basis;
    Shading shading;
    Traversal traversal = Traversal::Linear;
    TraversalOrder order = TraversalOrder::Unordered;

    // Byte offsets from the base of device memory. Three floats a vertex, in
    // world space and never rewritten — no 1/w, there being no projection on
    // this path to produce one.
    size_t triangles_offset = 0;
    size_t framebuffer_offset = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t triangle_count = 0;

    // The four early exits are folded into one flag rather than branched on, as
    // RasterStageArgs::predicated does for coverage. Read on the host when the
    // kernel is built, so the flag costs no lane anything.
    //
    // The route where the trade looked most likely to pay — the raster branch
    // guards a shade and little else, while this one guards most of
    // Möller-Trumbore — and it loses by more: about 4.4% against the raster
    // blend's 2%, with divergence going from 12.5% to zero. Predicating an
    // early exit means every lane finishes an intersection it would have
    // abandoned, and on a scene of small triangles most rays leave at the first
    // test of four.
    //
    // Folding the exits away also removes what they guarded, which coverage
    // never had to worry about. Two of the three differences in the kernel
    // follow from that rather than from predication itself.
    bool predicated = false;

    // Where the tree is, and how deep a stack it needs. Both come from the build
    // — Bvh::max_depth is reported rather than guessed because a stack one short
    // overruns into the next lane's slice of shared memory.
    //
    // The stack is in shared memory because an instruction names its registers in
    // immediate fields: there is no way to index the register file with a running
    // pointer. Real hardware puts a traversal stack in the same place, for the
    // same reason.
    size_t bvh_offset = 0;
    uint32_t stack_depth = 0;

    // The upper level: its nodes, its instances, and how deep it goes. The stack
    // holds both levels at once — the upper one is still on it while the lower is
    // walked — so a lane's slice is tlas_stack_depth + stack_depth entries.
    size_t tlas_offset = 0;
    size_t instances_offset = 0;
    uint32_t tlas_stack_depth = 0;

    // Where the colour is worked out. Deferring needs a shader that asks only
    // for what the loop can keep, which Barycentric and Custom do and a point
    // light does not — it wants the winning triangle's edges, and keeping those
    // is six more registers carried through the whole traversal.
    ShadeWhen shade_when = ShadeWhen::Inline;

    // How many rows of pixels a block covers.
    //
    // One is what every figure in benchmarks/ was taken on, and it means a block
    // is a single warp. Which makes REORDER a no-op: it moves threads between
    // the warps of a block, and there is only the one. A route that wants to
    // reorder has to ask for the room first.
    //
    // Threads are flattened row-major and cut into warps of 32, so a block N
    // rows tall holds the same warps grouped differently — the same lanes on the
    // same pixels, with somewhere for them to go.
    uint32_t block_rows = 1;
};

// Builds the ray tracer. args[0] must point at a RaytraceStageArgs.
//
// Möller-Trumbore leaves for the next triangle from four separate tests, which
// is what Label exists for — nesting if_ four deep would pay for the inverted
// condition each time and read as a staircase.
Program build_raytrace_program(void** args);

// Runs it, one thread per pixel. The launch geometry matches the untiled
// rasteriser's exactly, since the divergence figures are meant to be compared
// and a different warp shape would compare two things at once.
void run_raytrace_stage(MyGPURuntime& rt, const RaytraceStageArgs& args);
