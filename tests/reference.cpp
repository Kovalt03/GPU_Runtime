#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "math3d.hpp"
#include "pipeline.hpp"

namespace {

// clip space to pixels, the half of pass 1 that does not depend on how clip was
// reached. Shared so project_vertex and project_triangle cannot drift.
Float3 to_viewport(Float4 clip, uint32_t width, uint32_t height)
{
    // The perspective divide. Nothing guards w, deliberately: this is the
    // reference the kernel is measured against, and the kernel divides with
    // V_RCP_F32 unconditionally. A host that threw, or clamped, would stop being
    // a reference at exactly the inputs worth checking.
    const float inv_w = 1.0f / clip.w;
    const Float3 ndc{clip.x * inv_w, clip.y * inv_w, clip.z * inv_w};

    // NDC counts upward from the bottom of the frame; image rows count downward
    // from the top. Hence y is flipped and x is not.
    return Float3{
        (ndc.x + 1.0f) * 0.5f * static_cast<float>(width),
        (1.0f - ndc.y) * 0.5f * static_cast<float>(height),
        ndc.z,
    };
}

}  // namespace

Float3 project_vertex(const Float4x4& view_projection, Float3 world, uint32_t width,
                      uint32_t height)
{
    // w = 1 because a vertex is a position: the last column of the matrix is
    // the translation, and it has to apply. A normal or a ray direction would
    // pass 0 here.
    return to_viewport(transform(view_projection, world, 1.0f), width, height);
}

ScreenTriangle project_triangle(const Float4x4& view_projection, Float3 v0, Float3 v1,
                                Float3 v2, uint32_t width, uint32_t height)
{
    const Float3 corners[3] = {v0, v1, v2};
    ScreenTriangle out;
    Float3* positions[3] = {&out.v0, &out.v1, &out.v2};
    float* reciprocals[3] = {&out.inv_w0, &out.inv_w1, &out.inv_w2};

    for (uint32_t c = 0; c < 3; ++c) {
        // One transform, both answers. Computing them separately would leave two
        // places that have to agree about which w the divide used.
        const Float4 clip = transform(view_projection, corners[c], 1.0f);
        *positions[c] = to_viewport(clip, width, height);
        *reciprocals[c] = 1.0f / clip.w;
    }
    return out;
}

float edge_function(Float3 a, Float3 b, float px, float py)
{
    return (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
}

Float3 barycentric(Float3 v0, Float3 v1, Float3 v2, float px, float py)
{
    float w[3];
    w[0] = edge_function(v1, v2, px, py);
    w[1] = edge_function(v2, v0, px, py);
    w[2] = edge_function(v0, v1, px, py);
    const float inv_area = 1.0f / (w[0] + w[1] + w[2]);

    return Float3{w[0] * inv_area, w[1] * inv_area, w[2] * inv_area};
}

Float3 shade_pixel(Float3 v0, Float3 v1, Float3 v2, uint32_t px, uint32_t py,
                   float inv_w0, float inv_w1, float inv_w2)
{
    const float cx = static_cast<float>(px) + 0.5f;
    const float cy = static_cast<float>(py) + 0.5f;

    const Float3 w = barycentric(v0, v1, v2, cx, cy);

    // Tested the positive way round on purpose: a degenerate triangle makes
    // every weight NaN, and NaN fails >= as well as <, so it falls through to
    // the background instead of writing NaN into the frame.
    if (w.x >= 0.0f && w.y >= 0.0f && w.z >= 0.0f) {
        // Coverage is decided on the affine weights and colour on the corrected
        // ones. Only the second is an attribute carried across the triangle;
        // the first is a question about which side of three lines the pixel is.
        const Float3 c = perspective_correct(w, inv_w0, inv_w1, inv_w2);

        // (w1, w2, w0). The ray tracer colours a hit (u, v, 1 - u - v), where u
        // weights v1 and v weights v2 — so this ordering is what makes the two
        // renderers produce the same picture rather than one rotated in hue.
        return Float3{c.y, c.z, c.x};
    }
    return Float3{0.0f, 0.0f, 0.0f};
}

Float3 perspective_correct(Float3 affine, float inv_w0, float inv_w1, float inv_w2)
{
    const float weighted_x = affine.x * inv_w0;
    const float weighted_y = affine.y * inv_w1;
    const float weighted_z = affine.z * inv_w2;

    const float total = weighted_x + weighted_y + weighted_z;
    if (total == 0.0f) {
        return affine;
    }

    const float inv_total = 1.0f / total;
    return Float3{weighted_x * inv_total, weighted_y * inv_total, weighted_z * inv_total};
}

float interpolate_depth(Float3 v0, Float3 v1, Float3 v2, Float3 weights)
{
    return weights.x * v0.z + weights.y * v1.z + weights.z * v2.z;
}

Float3 shade_pixel_nearest(const std::vector<ScreenTriangle>& triangles, uint32_t px,
                           uint32_t py)
{
    // No depth buffer and no atomics: one thread owns one pixel outright, so
    // the running best is two local values. A thread per triangle would have
    // needed both, and neither exists in this ISA.
    //
    // The cost of that choice is this loop. Every pixel visits every triangle,
    // which is O(pixels x triangles) against the O(fragments) real hardware
    // pays — it bins triangles into tiles first, so a pixel only ever sees the
    // few that reach it. Fixing that is a change to how work is assigned, not
    // to where depth is kept, and it is measured against this version.
    const float cx = static_cast<float>(px) + 0.5f;
    const float cy = static_cast<float>(py) + 0.5f;

    // Beyond the far plane, NDC z running -1 near to +1 far, so the first
    // covering triangle always takes it.
    float best_z = 2.0f;
    Float3 best{0.0f, 0.0f, 0.0f};

    for (const ScreenTriangle& t : triangles) {
        const Float3 w = barycentric(t.v0, t.v1, t.v2, cx, cy);
        if (!(w.x >= 0.0f && w.y >= 0.0f && w.z >= 0.0f)) {
            continue;
        }

        // Depth from the affine weights — NDC z is linear in screen space, and
        // correcting it would be wrong rather than merely wasteful.
        //
        // Strict <, so coplanar triangles resolve to the first in the buffer
        // rather than flickering on a rounding difference.
        const float z = interpolate_depth(t.v0, t.v1, t.v2, w);
        if (z < best_z) {
            best_z = z;
            const Float3 c = perspective_correct(w, t.inv_w0, t.inv_w1, t.inv_w2);
            best = Float3{c.y, c.z, c.x};
        }
    }
    return best;
}

Hit intersect(const WorldTriangle& triangle, Float3 origin, Float3 direction)
{
    const Float3 e1 = triangle.v1 - triangle.v0;
    const Float3 e2 = triangle.v2 - triangle.v0;
    const Float3 h = cross(direction, e2);
    const float a = dot(e1, h);

    // Returning rather than noting a miss and carrying on: a is the divisor
    // below, and a parallel ray leaves it at zero.
    if (a < INTERSECT_EPSILON) {
        return Hit{};
    }

    const float f = 1.0f / a;
    const Float3 s = origin - triangle.v0;
    const float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return Hit{};
    }

    const Float3 q = cross(s, e1);
    const float v = f * dot(direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return Hit{};
    }

    const float t = f * dot(e2, q);
    if (t < INTERSECT_EPSILON) {
        return Hit{};
    }

    return Hit{true, t, u, v};
}

Float3 shade_diffuse(Float3 normal, Float3 hit, const Shading& shading)
{
    // The one place this reference is stricter than the kernel it stands for:
    // normalize throws on a light sitting exactly on the surface, where
    // V_NORM_VEC3_F32 divides by zero and carries on.
    const Float3 to_light = normalize(shading.light_position - hit);
    const float diffuse = std::max(0.0f, dot(normal, to_light));
    const Float3 colour = shading.base_colour * diffuse;
    return colour;
}

Float3 trace_pixel(const std::vector<WorldTriangle>& triangles, const RayBasis& basis,
                   uint32_t px, uint32_t py, uint32_t width, uint32_t height,
                   const Shading& shading)
{
    const float sx =
        (static_cast<float>(px) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f;
    const float sy =
        1.0f - (static_cast<float>(py) + 0.5f) / static_cast<float>(height) * 2.0f;
    const Float3 direction = basis.right * sx + basis.up * sy + basis.forward;

    // Black to start, so a ray that meets nothing leaves the background.
    float best_t = std::numeric_limits<float>::infinity();
    Float3 colour;

    for (const WorldTriangle& triangle : triangles) {
        const Hit hit = intersect(triangle, basis.origin, direction);

        // The miss has to be tested, not just the distance. A miss returns t at
        // zero, which beats every real hit and would paint the frame the colour
        // of nothing.
        if (!hit.hit || hit.t >= best_t) {
            continue;
        }
        best_t = hit.t;

        if (shading.mode == ShadingMode::Diffuse) {
            // No flipping: intersect culls back faces, so every hit that gets
            // here is wound the way cross expects.
            const Float3 normal =
                normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));

            const Float3 point = basis.origin + direction * hit.t;
            colour = shade_diffuse(normal, point, shading);
        } else {
            colour = Float3{hit.u, hit.v, 1.0f - hit.u - hit.v};
        }
    }
    return colour;
}
