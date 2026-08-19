#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/types.hpp"

// Host implementations of what the kernels compute, and the oracle each kernel
// is checked against. Nothing here runs on the simulated GPU, and no
// application or benchmark calls any of it — the test target is the only
// consumer, which is why it is built there rather than into the library.
//
// The rule that makes them worth having: a reference may not be smarter than
// the kernel it stands for. project_vertex does not guard w because the kernel
// divides with V_RCP_F32 unconditionally; a reference that clamped would stop
// being a reference at exactly the inputs worth checking. shade_diffuse names
// the one place the two knowingly differ.
//
// Drift is caught, not trusted to review: every kernel has a test comparing it
// against the function here, so a reference that grows a smarter branch turns
// one of them red.
//
//   project_vertex        VertexStageMatchesTheHostProjection
//   shade_pixel           RasterStageMatchesTheHostShading
//   shade_pixel_nearest   TiledRasterDrawsTheSameFrameAsTheWalk
//   trace_pixel           RaytraceStageMatchesTheHostTrace
//   shade_diffuse         LitKernelMatchesTheLitHost

// Möller-Trumbore on the host. t is measured in units of the direction given,
// which is not normalised: the length cancels when t values from one pixel are
// compared against each other, and a normalise costs 12.
struct Hit {
    bool hit = false;
    float t = 0.0f;
    float u = 0.0f;  // barycentric weight of v1
    float v = 0.0f;  // barycentric weight of v2
};

// The same projection on the host. Pass 2 wants it too, to decide what a whole
// frame should have looked like.
//
//   clip   = view_projection * (world, 1)
//   ndc    = clip.xyz / clip.w
//   screen = ((ndc.x + 1) / 2 * width, (1 - ndc.y) / 2 * height, ndc.z)
//
// y is flipped because NDC counts upward from the bottom and an image counts
// downward from the top row.
Float3 project_vertex(const Float4x4& view_projection, Float3 world, uint32_t width,
                      uint32_t height);

// A whole triangle through pass 1, on the host: the positions project_vertex
// gives plus the reciprocals of w that the kernel keeps alongside them.
//
// The natural unit, since a ScreenTriangle is what the raster path consumes and
// a vertex on its own cannot carry the reciprocal.
ScreenTriangle project_triangle(const Float4x4& view_projection, Float3 v0, Float3 v1,
                                Float3 v2, uint32_t width, uint32_t height);

// Twice the signed area of the triangle (a, b, p): the z component of a 2D
// cross product. Its sign says which side of the line ab the point falls on,
// and the three of them together decide coverage.
float edge_function(Float3 a, Float3 b, float px, float py);

// Barycentric weights of (px, py), in vertex order.
//
// Normalised by the signed area, which is what lets coverage be "all three are
// >= 0" whatever way the triangle winds. That is not a nicety here: the y flip
// in pass 1 reverses the winding of every triangle, so a test written against
// one order would draw nothing at all.
Float3 barycentric(Float3 v0, Float3 v1, Float3 v2, float px, float py);

// The colour one triangle gives a pixel, black where it does not cover it.
//
// Samples the pixel centre, matching the ray tracer: an edge landing on a
// corner otherwise leaves neighbouring pixels disagreeing and frays it.
//
// Ordered (w1, w2, w0), which is what makes each vertex the same primary
// apps/ray_triangle.cpp produces from (u, v, 1 - u - v). The two renderers
// are meant to be compared as images, and a rotation in hue is the kind of
// wrong that still looks deliberate.
//
// The reciprocals default to 1, which is the affine case — every vertex at one
// depth. Tests that build a triangle by hand rather than projecting one want
// exactly that.
Float3 shade_pixel(Float3 v0, Float3 v1, Float3 v2, uint32_t px, uint32_t py,
                   float inv_w0 = 1.0f, float inv_w1 = 1.0f, float inv_w2 = 1.0f);

// Turns affine barycentric weights into the ones an attribute needs.
//
//   denominator = w0/w0_clip + w1/w1_clip + w2/w2_clip
//   corrected_i = (w_i / wi_clip) / denominator
//
// Leaves the weights alone when every vertex shares a depth, which is why the
// two renderers agreed before this existed and why a scene like that proves
// nothing about interpolation.
Float3 perspective_correct(Float3 affine, float inv_w0, float inv_w1, float inv_w2);

// Depth of the triangle at a pixel.
//
// A plain weighted sum, which would be wrong for anything else carried across a
// projected triangle: pass 1 already divided z by w, so it varies linearly in
// screen space. That is why a depth buffer stores NDC z and not distance from
// the camera.
float interpolate_depth(Float3 v0, Float3 v1, Float3 v2, Float3 weights);

// The whole of pass 2 for one pixel, and the reference a rendered frame is
// checked against: the colour of the nearest triangle covering it, black if
// none does.
//
// Nearest means smallest depth, NDC running -1 at the near plane to +1 at the
// far one.
Float3 shade_pixel_nearest(const std::vector<ScreenTriangle>& triangles, uint32_t px,
                           uint32_t py);

Hit intersect(const WorldTriangle& triangle, Float3 origin, Float3 direction);

// Lambert, for one hit. normal and to_light are expected normalised; the
// clamp at zero is what stops a surface facing away from the light from
// being lit from behind.
//
//   diffuse = max(0, dot(normal, to_light))
//   colour  = base * diffuse
Float3 shade_diffuse(Float3 normal, Float3 hit, const Shading& shading);

// The colour a pixel takes from the nearest triangle its ray meets, black on a
// miss — the whole of the kernel for one pixel, and its reference.
//
// Coloured (u, v, 1 - u - v), which is what shade_pixel_nearest produces from
// the barycentric weights it computes a different way. The two renderers have
// to agree pixel for pixel, and that only works if they agree here first.
//
// Defaulted so the calls that predate lighting keep meaning what they did.
Float3 trace_pixel(const std::vector<WorldTriangle>& triangles, const RayBasis& basis,
                   uint32_t px, uint32_t py, uint32_t width, uint32_t height,
                   const Shading& shading = Shading{});
