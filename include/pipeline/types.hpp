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

// --- shading -----------------------------------------------------------------
// How a hit is coloured, shared by both renderers so that a frame from one can
// be held against a frame from the other.
//
// Barycentric is the debug colouring, and was for a long time the only one the
// rasteriser could produce: pass 1 keeps screen x, y, depth and 1/w, so the
// world position and normal a light needs are gone by the time pass 2 runs. What
// changed is that the geometry now stays on the device — the world vertices pass
// 1 read are still there, and a face normal a triangle is a small buffer beside
// them.
enum class ShadingMode {
    Barycentric,
    Diffuse,
};

struct Shading {
    ShadingMode mode = ShadingMode::Barycentric;

    // World space. Only read in Diffuse.
    Float3 light_position{2.0f, 3.0f, 1.0f};
    Float3 base_colour{1.0f, 1.0f, 1.0f};
};

// One unit normal a triangle, which is what the raster routes read rather than
// deriving it per pixel. The ray tracer takes the cross product it already has
// the edges for; the rasteriser would have to load three world vertices to do
// the same, and does load them — for the point a point light needs.
inline constexpr uint32_t FACE_NORMAL_FLOATS = 3;
inline constexpr uint32_t FACE_NORMAL_BYTES = FACE_NORMAL_FLOATS * sizeof(float);

// --- depth -------------------------------------------------------------------
// What a raster launch does with the depth buffer.
//
// A depth prepass is two launches over the same geometry: the first keeps the
// nearest depth a pixel and colours nothing, the second colours only the
// triangle that depth names. What it buys is that a pixel is shaded once instead
// of once per covering triangle; what it costs is a second walk of every
// triangle, which is not cheap here — coverage is most of the loop.
enum class DepthUse {
    // No depth buffer. The running best in a register is all a single-pass walk
    // needs, one thread owning one pixel outright.
    None,

    // Write the nearest depth and shade nothing.
    Prepass,

    // Read it, and shade only the triangle that owns the pixel.
    EarlyZ,
};

// One float a pixel. Separate from the colour buffer rather than a fourth
// channel of it, because the prepass writes only this and the frame is read back
// as RGB.
inline constexpr uint32_t DEPTH_BYTES = sizeof(float);
