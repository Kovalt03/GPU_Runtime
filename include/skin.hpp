#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "math3d.hpp"
#include "pipeline/types.hpp"
#include "skeleton.hpp"

// Binding a mesh to a skeleton, and moving it.
//
// One bone a vertex — rigid attachment rather than the weighted blend of linear
// blend skinning. That is what the data this was built against actually holds:
// a rig exported as one mesh per bone shares a vertex pool and partitions the
// triangles, so a triangle names exactly one joint and there are no weights to
// read. Weights would have to be invented, and a measurement on invented weights
// says less than one on a real partition.
//
// The plan reserved V_SCALE_MAT4_F32 and V_ADD_MAT4_F32 for the weighted blend.
// Neither is built, and the arithmetic says why even if weights arrived: a
// vertex is one position, so blending four matrices (four scales and three adds
// over sixteen elements, then one transform) costs 128 where transforming by
// four matrices and blending the results costs 76. Composing only pays when the
// same matrix serves more than four vectors, and a vertex is one.

struct Skin {
    // Which joint each vertex follows.
    std::vector<uint32_t> bone;

    // World to bone at rest, one a joint. What takes a vertex out of the pose it
    // was modelled in and into the space its bone moves.
    std::vector<Float4x4> inverse_rest;

    uint32_t vertex_count() const;
};

// The transform that puts a mesh inside a skeleton's rest pose: a uniform scale
// matching their heights, then a translation matching their centres.
//
// The first thing an auto-rigger does, and leaving it out is what makes binding
// look like it works while it does not — a mesh scaled but not moved sits
// wherever it was modelled, every vertex ends up nearest the same two or three
// joints, and the frames still animate. Bone coverage is the thing to check, not
// whether anything moved.
Float4x4 fit_to_rest(const std::vector<Float3>& vertices, const Motion& motion);

// Binds every vertex to the nearest bone segment of the rest pose.
//
// What an auto-rigger does, and what the meshes in assets/ need because they
// arrive with no rig at all. The nearest *segment* rather than the nearest
// joint: a point beside the middle of a forearm is nearer to neither end than to
// the line between them, and binding it to a joint would swap it across the
// elbow as the arm turned.
//
// The vertices are taken as they are: fit_to_rest above is applied by the caller,
// which also has to render them, so binding a mesh to one place and drawing it in
// another cannot happen by forgetting an argument.
Skin bind_nearest(const std::vector<Float3>& vertices, const Motion& motion);

// The same, with the binding given rather than derived: `bone[i]` is the joint
// vertex i follows. For a rig that already says so.
Skin bind_given(const std::vector<uint32_t>& bone, const Motion& motion);

// One matrix a joint for a frame: where the joint is now, composed with where it
// was at rest inverted. A vertex multiplied by its own is where it has moved to.
std::vector<Float4x4> skin_palette(const Motion& motion, const Skin& skin,
                                   uint32_t frame);

// Emits the skinning. One lookup and one transform a vertex.
//
// The bone index is read from a buffer rather than carried as a varying: a
// shader may address anything it knows the offset of, and a varying would spend
// a screen-vertex slot on a number pass 2 never looks at.
//
// The palette read is V_LD_GLOBAL_MAT4_F32 and has to be. Neighbouring vertices
// follow different joints, so the address differs by lane — which is exactly
// what the constant window cannot price and what the wide global load is for.
VertexFn skinning_shader(size_t bone_offset, size_t palette_offset);
