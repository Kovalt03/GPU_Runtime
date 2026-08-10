#pragma once

#include <cstdint>

#include "math3d.hpp"

// The graphics layer. Everything below is a thin wrapper over myrt_launch: the
// runtime stays general-purpose the way CUDA is, and the knowledge that these
// numbers are triangles lives only here.
//
// Rasterisation runs in two passes, because the two halves are indexed
// differently. The MVP transform is per vertex and coverage is per pixel, and a
// single kernel would have to pick one and waste threads on the other. Real
// hardware splits them for the same reason, into a vertex stage and a fragment
// stage, and this runtime can already launch twice.
//
//   pass 1  vertex_program   1 thread = 1 vertex   world -> screen
//   pass 2  raster_program   1 thread = 1 pixel    coverage, nearest wins
//
// Each stage is a header of its own. This one holds only what more than one of
// them needs: the strides they address device memory by, and the triangle the
// vertex stage hands the raster stages.

// --- device buffer layout ---------------------------------------------------
// Kernels address device memory by byte offset, so the strides are named once
// here rather than spelled out at every load.

// x, y, z in world space.
inline constexpr uint32_t WORLD_VERTEX_FLOATS = 3;
inline constexpr uint32_t WORLD_VERTEX_BYTES = WORLD_VERTEX_FLOATS * sizeof(float);

// x, y in pixels, the NDC depth pass 2 compares, and 1/w.
//
// w is kept because the divide is exactly what makes screen-space interpolation
// wrong for anything but depth. Barycentric weights taken from projected
// vertices are affine; an attribute carried across a perspective-projected
// triangle needs them weighted by 1/w and renormalised. Depth does not, being
// linear in screen space by construction — which is the whole reason a depth
// buffer stores NDC z.
//
// Costs a float per vertex and a reciprocal per pixel, and without it the
// rasteriser and the ray tracer only agree on a scene whose triangles all sit
// at one depth.
inline constexpr uint32_t SCREEN_VERTEX_FLOATS = 4;
inline constexpr uint32_t SCREEN_VERTEX_BYTES = SCREEN_VERTEX_FLOATS * sizeof(float);

// A triangle as pass 1 leaves it: x and y in pixels, z the NDC depth.
struct ScreenTriangle {
    Float3 v0;
    Float3 v1;
    Float3 v2;

    // One per vertex, in that order. Named rather than packed into a Float3,
    // which would read as a vector and is three unrelated scalars.
    float inv_w0 = 1.0f;
    float inv_w1 = 1.0f;
    float inv_w2 = 1.0f;
};

// RGB per pixel, the layout kernels/ray_triangle.cpp already writes, so the two
// renderers can be compared image against image rather than by description.
inline constexpr uint32_t PIXEL_FLOATS = 3;
inline constexpr uint32_t PIXEL_BYTES = PIXEL_FLOATS * sizeof(float);
