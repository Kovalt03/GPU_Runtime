#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"

// A bounding volume hierarchy over world-space triangles.
//
// Host code with no ISA in it, beside mesh.hpp for the same reason: an
// acceleration structure is something the geometry arrives with, not a stage of
// the pipeline. What the device does with it lives in pipeline/raytrace.hpp.
//
// This exists because the ray tracer walks every triangle for every pixel. That
// is what makes its cost linear in the scene, and it is the last algorithm in
// this repository still written that way.

// How a node's triangles are split between its children.
//
// A flag rather than a decision: both build the same number of nodes over the
// same triangles, and what differs is how many a ray has to visit. That is a
// measurement, so both stay.
enum class BvhSplit {
    // The midpoint of the widest axis, by triangle centroid. Cheap, and blind to
    // how much empty space a split leaves behind.
    Median,

    // Surface area heuristic: the split whose children are cheapest to trace,
    // estimated as area x count. The standard, and what a builder is judged
    // against.
    SAH,
};

// Eight floats a node — a quarter of a cache line, and two wide loads.
//
//   [0..2]  bounds min        [3..5]  bounds max
//   [6]     the left child, or the first triangle when this is a leaf
//   [7]     how many triangles — zero marks an interior node
//
// Children are adjacent, so an interior node keeps one index rather than two.
// Indices are floats because the device has no integer registers; 2^24 is the
// exact range, which is 16 million nodes and far past what fits in memory here.
inline constexpr uint32_t BVH_NODE_FLOATS = 8;
inline constexpr uint32_t BVH_NODE_BYTES = BVH_NODE_FLOATS * sizeof(float);

// How many triangles a leaf may hold before it has to split.
//
// A cap rather than a target, and the heuristic picks only where to split. The
// trade a benchmark argument exists to explore: one triangle a leaf makes the
// deepest tree and the tightest bounds, eight makes a shallow tree that tests
// more triangles once a ray arrives. Four is the usual answer and the default.
inline constexpr uint32_t BVH_DEFAULT_LEAF = 4;

// An axis-aligned box, which is what a tree is built over whatever the leaves
// turn out to hold. Triangles reduce to one and so do whole objects.
struct Box {
    Float3 lo;
    Float3 hi;
};

struct Bvh {
    // BVH_NODE_FLOATS each, root first. Uploaded as it stands.
    std::vector<float> nodes;

    // Where each leaf's items came from: order[i] is the index, in the caller's
    // list, of the item now sitting at position i. A leaf names a range of these
    // positions, so a caller has to permute its own list to match.
    //
    // build_bvh does that permutation for triangles below; a tree over instances
    // has to do it with this, its items not being something this file knows.
    std::vector<uint32_t> order;

    // The triangles, permuted so that a leaf is a contiguous range rather than a
    // list of indices. Costs one reordering on the host and saves the device a
    // dependent load per triangle — the same trade the tiled routes make when
    // bin_triangles de-indexes into a tile.
    std::vector<Float3> triangles;

    // The deepest path from the root, in nodes. What a traversal stack has to
    // hold, and why a build reports it rather than a caller guessing.
    uint32_t max_depth = 0;

    uint32_t node_count() const;
    uint32_t triangle_count() const;

    // The root's bounds, which is the whole scene. Used by tests and by anything
    // that wants to place a camera without knowing the geometry.
    Float3 bounds_min() const;
    Float3 bounds_max() const;
};

// Builds one over a flattened vertex list — three vertices a triangle, the form
// the ray tracer already takes.
//
// Throws std::runtime_error if the list is not a multiple of three or is empty:
// a tree over nothing has no root, and every traversal here starts by reading
// one.
Bvh build_bvh(const std::vector<Float3>& world, BvhSplit split = BvhSplit::SAH,
              uint32_t leaf_size = BVH_DEFAULT_LEAF);

// The same, over boxes a caller has worked out itself. What the triangle build
// runs on underneath, and what a tree over objects needs: an instance's box is
// its geometry's transformed, which this file has no way to compute.
//
// Leaves the permutation in `order` and `triangles` empty.
Bvh build_bvh_over(const std::vector<Box>& boxes, BvhSplit split = BvhSplit::SAH,
                   uint32_t leaf_size = BVH_DEFAULT_LEAF);

// --- the second level --------------------------------------------------------
// A tree over objects rather than over triangles.
//
// The lower level (a BLAS) is built once over geometry in its own space; the
// upper one (a TLAS) is built over where the copies of it ended up. A ray walks
// the upper tree in world space, and at a leaf it moves into the instance's
// space and walks the lower one there.
//
// Moving the ray rather than the geometry is the whole trick. Transforming a
// scene's triangles for every instance would cost a tree apiece; transforming
// one ray costs two MATVECs, and the parameter t survives it — a hit at t in
// object space is the hit at t in world space, provided the direction is carried
// as a vector and left unnormalised.

// Nineteen floats an instance: the world-to-object matrix, then where the
// instance's own lower-level tree and triangles begin, then what it is made of.
//
// The two offsets are what make this a scene rather than a field of copies. With
// them baked into the kernel every instance walked the same tree at the same
// address, so a hundred cubes were expressible and a cube beside a sphere was
// not. They cost two loads at a leaf where sixteen were already being spent.
//
//   [0..15]  world to object, row-major
//   [16]     byte offset of this instance's lower-level nodes
//   [17]     byte offset of its triangles
//   [18]     which material — read by a shader, and nothing else looks at it
inline constexpr uint32_t TLAS_INSTANCE_FLOATS = 19;
inline constexpr uint32_t TLAS_INSTANCE_BYTES = TLAS_INSTANCE_FLOATS * sizeof(float);

// Which geometry, placed where, made of what.
//
// The material is a number this file never interprets. It reaches a fragment
// shader and stops there, which is the point — what makes lanes of a warp take
// different paths has to come from the scene rather than from a key invented to
// produce it.
struct TlasInstance {
    // Index into the trees passed to build_tlas.
    uint32_t blas = 0;

    // Object to world. The inverse is what a ray needs, and the build works it
    // out; this one is here because the instance's box comes from it.
    Float4x4 model;

    uint32_t material = 0;
};

struct Tlas {
    // Over the instances' world-space boxes. A leaf names a range of the two
    // vectors below, the build having already permuted them into its order.
    Bvh tree;

    // What the caller passed, reordered. A caller that keeps its own per-instance
    // data has to permute it the same way, which is what `tree.order` is for.
    std::vector<TlasInstance> placed;

    // World to object, one per placed instance and in the same order. Kept apart
    // from the struct above because it is derived rather than given, and mixing
    // the two would invite a caller to fill in the wrong one.
    std::vector<Float4x4> inverses;

    uint32_t instance_count() const;
};

// Builds one over the boxes each instance's geometry lands in.
//
// The trees supply the object-space bounds, so a caller passes the ones it
// already built rather than the geometry again. An instance's box is the AABB of
// the eight transformed corners of its own tree's root — loose for a rotation,
// and the standard answer, since keeping it tight would mean re-deriving bounds
// from geometry the upper level does not read.
//
// Throws std::runtime_error if there are no instances, an instance names a tree
// that is not there, or a matrix cannot be inverted.
Tlas build_tlas(const std::vector<Bvh>& trees, const std::vector<TlasInstance>& instances,
                BvhSplit split = BvhSplit::SAH, uint32_t leaf_size = 1);

// How many nodes a ray from `origin` through `direction` would visit, counted on
// the host.
//
// The device reports cycles and instructions, which mix traversal with the
// intersection tests it leads to. This counts nodes and nothing else, which is
// what separates "the tree is better" from "the kernel is better" — and it is
// how Median and SAH are compared without a launch in the way.
uint32_t nodes_visited(const Bvh& bvh, Float3 origin, Float3 direction);
