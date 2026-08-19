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

struct Bvh {
    // BVH_NODE_FLOATS each, root first. Uploaded as it stands.
    std::vector<float> nodes;

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

// How many nodes a ray from `origin` through `direction` would visit, counted on
// the host.
//
// The device reports cycles and instructions, which mix traversal with the
// intersection tests it leads to. This counts nodes and nothing else, which is
// what separates "the tree is better" from "the kernel is better" — and it is
// how Median and SAH are compared without a launch in the way.
uint32_t nodes_visited(const Bvh& bvh, Float3 origin, Float3 direction);
