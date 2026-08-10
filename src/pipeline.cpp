#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ir_builder.hpp"
#include "math3d.hpp"
#include "pipeline.hpp"
#include "thread.hpp"  // WARP_SIZE, the launch width

namespace {

// Rows and columns of the view-projection matrix, so the walk that bakes it
// into registers reads as the row-major rule rather than as a bare 4.
constexpr uint32_t MATRIX_DIMENSION = 4;

// The device counterpart of edge_function: the same expression, emitted as
// instructions rather than evaluated. The two have to stay in step, since the
// host one is what the kernel's output is checked against.
//
//   (px - ax) * (by - ay) - (py - ay) * (bx - ax)
Reg<Scalar> emit_edge(IRBuilder& k, Reg<Scalar> ax, Reg<Scalar> ay, Reg<Scalar> bx,
                      Reg<Scalar> by, Reg<Scalar> px, Reg<Scalar> py)
{
    return k.sub(k.mul(k.sub(px, ax), k.sub(by, ay)),
                 k.mul(k.sub(py, ay), k.sub(bx, ax)));
}

// Writes the corrected weights into dst as (w1, w2, w0), the order the
// framebuffer takes. The device counterpart of perspective_correct, minus its
// guard against a zero denominator: inside a covered pixel the affine weights
// sum to one and every 1/w is positive, so the sum cannot reach zero.
void emit_shade(IRBuilder& k, Reg<Vec3> dst, Reg<Scalar> w0, Reg<Scalar> w1,
                Reg<Scalar> w2, Reg<Scalar> iw0, Reg<Scalar> iw1, Reg<Scalar> iw2)
{
    const Reg<Scalar> a0 = k.mul(w0, iw0);
    const Reg<Scalar> a1 = k.mul(w1, iw1);
    const Reg<Scalar> a2 = k.mul(w2, iw2);
    const Reg<Scalar> inv_total = k.rcp(k.add(k.add(a0, a1), a2));

    k.copy_into(dst.component(0), k.mul(a1, inv_total));
    k.copy_into(dst.component(1), k.mul(a2, inv_total));
    k.copy_into(dst.component(2), k.mul(a0, inv_total));
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass 1 — projection
//
// The host version first. It is arithmetic with no ISA in the way, and once it
// passes its own tests the kernel has something to be wrong against, which is
// the only practical way to debug one here.
// ---------------------------------------------------------------------------

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

// The same projection as a program: the first kernel written entirely with
// IRBuilder, and the first use of V_MATVEC_MAT4_F32 outside its own test.

Program build_vertex_program(void** args)
{
    const VertexStageArgs& a = *static_cast<const VertexStageArgs*>(args[0]);
    IRBuilder k;

    // The uniform, baked in as sixteen moves. Row-major on both sides, so no
    // transpose belongs here — if a render comes out transposed, this loop is
    // the first suspect.
    const Reg<Mat4> mvp = k.mat4();
    for (uint32_t row = 0; row < MATRIX_DIMENSION; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIMENSION; ++col) {
            k.set(mvp.component(row * MATRIX_DIMENSION + col),
                  a.view_projection.at(row, col));
        }
    }

    // A launch rounds up to whole warps, so unless vertex_count is a multiple
    // of 32 the last warp runs lanes with no vertex to read. Everything below
    // sits inside the guard; without it those lanes address past both buffers.
    const Reg<Scalar> id = k.thread_x();
    const Reg<Scalar> live = k.lt(id, k.constant(static_cast<float>(a.vertex_count)));

    k.if_(live, [&] {
        // The address is computed on the device, since it depends on the lane.
        // The buffer's own offset rides in the load's immediate instead of an
        // instruction of its own — V_LD_GLOBAL_F32 reads global[reg + imm].
        const Reg<Scalar> stride = k.constant(static_cast<float>(WORLD_VERTEX_BYTES));
        const Reg<Scalar> addr = k.mul(id, stride);

        // load_into and set rather than load_vec3: the position has to occupy
        // the leading three registers of a VEC4 with 1 in the fourth, and only
        // these two can put a value at a chosen index.
        const Reg<Vec4> position = k.vec4();
        const float world_base = static_cast<float>(a.world_offset);
        k.load_into(position.component(0), addr, world_base + 0.0f);
        k.load_into(position.component(1), addr, world_base + 4.0f);
        k.load_into(position.component(2), addr, world_base + 8.0f);
        k.set(position.component(3), 1.0f);

        const Reg<Vec4> clip = k.transform(mvp, position);

        // The perspective divide, and why Reg carries both views: w on its own
        // and xyz as a vector, over one register range.
        const Reg<Scalar> inv_w = k.rcp(clip.component(3));
        const Reg<Vec3> ndc = k.scale(clip.xyz(), inv_w);

        // Scalar work, because the y flip is not the same expression as x: NDC
        // counts upward from the bottom and image rows count down from the top.
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> half_w = k.constant(static_cast<float>(a.width) * 0.5f);
        const Reg<Scalar> half_h = k.constant(static_cast<float>(a.height) * 0.5f);

        const Reg<Scalar> sx = k.mul(k.add(ndc.component(0), one), half_w);
        const Reg<Scalar> sy = k.mul(k.sub(one, ndc.component(1)), half_h);
        const Reg<Scalar> sz = ndc.component(2);

        // A second address register, the two buffers no longer sharing a
        // stride: the screen vertex carries 1/w and the world one does not.
        const Reg<Scalar> out_addr =
            k.mul(id, k.constant(static_cast<float>(SCREEN_VERTEX_BYTES)));

        const float screen_base = static_cast<float>(a.screen_offset);
        k.store(out_addr, sx, screen_base + 0.0f);
        k.store(out_addr, sy, screen_base + 4.0f);
        k.store(out_addr, sz, screen_base + 8.0f);

        // inv_w falls out of the divide above at no extra cost, and pass 2
        // cannot recover it once w is gone.
        k.store(out_addr, inv_w, screen_base + 12.0f);
    });

    return k.build();
}

void run_vertex_stage(MyGPURuntime& rt, const VertexStageArgs& args)
{
    // Caught here rather than left to myrt_launch, which would reject the empty
    // grid with a message about launch geometry and say nothing about vertices.
    if (args.vertex_count == 0) {
        throw std::runtime_error("run_vertex_stage: a mesh with no vertices");
    }

    // 1D, one warp wide: a vertex has no second coordinate to spend, unlike a
    // pixel. The last block is partly out of range whenever vertex_count is not
    // a multiple of the warp, which is what the guard in the kernel is for.
    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.vertex_count + WARP_SIZE - 1) / WARP_SIZE, 1, 1};

    // myrt_launch takes void** after CUDA's convention, so the const has to
    // come off. The kernel only reads it, and args outlives the call.
    void* raw[] = {const_cast<VertexStageArgs*>(&args)};
    rt.myrt_launch(build_vertex_program, grid, block, raw);
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

// ---------------------------------------------------------------------------
// Pass 2 — coverage
//
// Three sign conventions stack up here — the triangle's winding, the y flip
// pass 1 applied, and where inside a pixel the sample sits — and each of them
// wrong still renders a picture that looks deliberate. Hence a host version
// again, and tests written as geometric facts rather than as expected numbers.
// ---------------------------------------------------------------------------

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

// Coverage as a program. The first kernel whose lanes take different paths for
// a reason that is not a bounds check: a warp covers 32 horizontally adjacent
// pixels, so a triangle edge crossing it splits the warp. The divergence
// benchmark already measured what that costs — one diverged lane is as
// expensive as sixteen.

Program build_raster_program(void** args)
{
    const RasterStageArgs& a = *static_cast<const RasterStageArgs*>(args[0]);
    IRBuilder k;

    // Both coordinates need checking, not just one: the launch rounds up along
    // x and y both. min() of two flags is their AND, each being 1.0 or 0.0 —
    // one divergence point instead of the two that nesting if_ would cost.
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        // Loop invariants, hoisted: emitting them inside the body would run a
        // move per triangle for values that never change.
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> half = k.constant(0.5f);

        // The pixel centre, matching what the ray tracer samples.
        const Reg<Scalar> cx = k.add(px, half);
        const Reg<Scalar> cy = k.add(py, half);

        // The running best is all a depth buffer would have held: one thread
        // owns one pixel outright, so nothing is shared and no atomic is needed.
        // A thread per triangle would have required both.
        //
        // Started beyond the far plane so the first covering triangle takes it.
        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // The cursor walks the buffer, the counter ends the loop. Both advance
        // through fma, the only opcode that accumulates in place.
        const Reg<Scalar> tri_addr = k.constant(static_cast<float>(a.screen_offset));
        const Reg<Scalar> stride =
            k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
        const Reg<Scalar> i = k.constant(0.0f);
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));

        // The loop splits no warp: every lane runs the same number of
        // iterations and reaches the backward branch together. Only the
        // coverage test inside it diverges.
        //
        // Every lane also loads the same twelve floats, the triangle not
        // depending on the pixel. Real hardware broadcasts these from a scalar
        // unit; here all 32 lanes issue the same load — the redundancy the
        // sixteen matrix moves already cost pass 1, and the reason the ISA
        // reserves an S_ prefix.
        const Label top = k.label();
        k.place(top);

        // Position then 1/w, three times — the layout pass 1 writes and
        // bin_triangles copies.
        const Reg<Scalar> x0 = k.load(tri_addr, 0.0f);
        const Reg<Scalar> y0 = k.load(tri_addr, 4.0f);
        const Reg<Scalar> z0 = k.load(tri_addr, 8.0f);
        const Reg<Scalar> iw0 = k.load(tri_addr, 12.0f);
        const Reg<Scalar> x1 = k.load(tri_addr, 16.0f);
        const Reg<Scalar> y1 = k.load(tri_addr, 20.0f);
        const Reg<Scalar> z1 = k.load(tri_addr, 24.0f);
        const Reg<Scalar> iw1 = k.load(tri_addr, 28.0f);
        const Reg<Scalar> x2 = k.load(tri_addr, 32.0f);
        const Reg<Scalar> y2 = k.load(tri_addr, 36.0f);
        const Reg<Scalar> z2 = k.load(tri_addr, 40.0f);
        const Reg<Scalar> iw2 = k.load(tri_addr, 44.0f);

        // Each weight is the edge opposite its vertex, as in barycentric().
        const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
        const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
        const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

        // The sum is twice the signed area and does not vary with the pixel, so
        // one reciprocal serves all three weights.
        const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
        const Reg<Scalar> inv_area = k.rcp(area);
        const Reg<Scalar> w0 = k.mul(e0, inv_area);
        const Reg<Scalar> w1 = k.mul(e1, inv_area);
        const Reg<Scalar> w2 = k.mul(e2, inv_area);

        // Positive comparisons, as in shade_pixel: NaN fails them, so a
        // degenerate triangle leaves the background.
        const Reg<Scalar> inside =
            k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

        const Reg<Scalar> depth = k.mul(w0, z0);
        k.fma(depth, w1, z1);
        k.fma(depth, w2, z2);

        // Both conditions folded into one flag, and so one divergence point:
        // covered, and nearer than anything kept so far.
        const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));
        // Depth from the affine weights, colour from the corrected ones:
        // NDC z is linear in screen space and an attribute is not.
        k.if_(take, [&] {
            k.copy_into(best_z, depth);
            emit_shade(k, best, w0, w1, w2, iw0, iw1, iw2);
        });

        k.fma(tri_addr, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

        // One store after the whole walk, rather than one per triangle. Every
        // in-image lane runs it together: the divergence above is the cost of
        // deciding a colour, not of writing one.
        // One pixel, once, after the whole scene has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });

    // No matrix, which is the other half of why the pipeline is split: pass 1
    // spends sixteen registers on a uniform and this one never does.
    return k.build();
}

void run_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_raster_stage: an image with no pixels");
    }
    // The kernel's loop tests its counter at the bottom, so a count of zero
    // would still walk one triangle's worth of whatever the buffer holds.
    if (args.triangle_count == 0) {
        throw std::runtime_error("run_raster_stage: nothing to draw");
    }

    // 32 along x so that one warp lands on 32 horizontally adjacent pixels of a
    // single row. That is what makes a triangle edge split a warp rather than
    // fall between two, which is the whole thing being measured.
    //
    // 2D rather than a flat index because the ISA has no integer division to
    // recover a column from one.
    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE, args.height, 1};

    void* raw[] = {const_cast<RasterStageArgs*>(&args)};
    rt.myrt_launch(build_raster_program, grid, block, raw);
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

// ---------------------------------------------------------------------------
// Tiling
//
// The same picture as the walk above, reached by reading a shorter list. What
// changes is only which triangles a pixel ever sees, so the two are directly
// comparable — and have to produce identical frames.
// ---------------------------------------------------------------------------

TileBinning bin_triangles(const std::vector<ScreenTriangle>& triangles, uint32_t width,
                          uint32_t height)
{
    (void)triangles;
    (void)width;
    (void)height;
    // tiles_x = ceil(width / TILE_WIDTH), tiles_y likewise. Rounding up, so the
    // last tile is partly off screen and its threads fall out on the bounds
    // check the kernel already has.
    //
    TileBinning binning;
    binning.tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    binning.tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    // For each tile, walk every triangle and keep the ones whose screen
    // bounding box overlaps it:
    //
    //   min/max of the three x and the three y
    //   overlap when  min.x < (tx + 1) * TILE_WIDTH  and  max.x >= tx * TILE_WIDTH
    //   and the same along y
    //
    struct Box {
        float min_x, min_y, max_x, max_y;
    };
    std::vector<Box> boxes;

    for (const ScreenTriangle& t : triangles) {
        boxes.push_back(
            Box{std::min({t.v0.x, t.v1.x, t.v2.x}), std::min({t.v0.y, t.v1.y, t.v2.y}),
                std::max({t.v0.x, t.v1.x, t.v2.x}), std::max({t.v0.y, t.v1.y, t.v2.y})});
    }

    for (uint32_t ty = 0; ty < binning.tiles_y; ++ty) {
        for (uint32_t tx = 0; tx < binning.tiles_x; ++tx) {
            const float left = static_cast<float>(tx * TILE_WIDTH);
            const float top = static_cast<float>(ty * TILE_HEIGHT);

            binning.table.push_back(
                static_cast<float>(binning.vertices.size() / TILE_TRIANGLE_FLOATS));
            uint32_t count = 0;

            for (size_t i = 0; i < triangles.size(); ++i) {
                const Box& b = boxes[i];
                // Written as "no overlap", which is four independent rejections
                // rather than four conditions that all have to line up.
                if (b.max_x < left || b.min_x >= left + TILE_WIDTH) {
                    continue;
                }
                if (b.max_y < top || b.min_y >= top + TILE_HEIGHT) {
                    continue;
                }

                // Interleaved as pass 1 writes it — position then 1/w, three
                // times — so a kernel reads a binned triangle and a screen one
                // with the same offsets.
                const ScreenTriangle& t = triangles[i];
                const float reciprocals[3] = {t.inv_w0, t.inv_w1, t.inv_w2};
                const Float3 corners[3] = {t.v0, t.v1, t.v2};
                for (uint32_t c = 0; c < 3; ++c) {
                    binning.vertices.push_back(corners[c].x);
                    binning.vertices.push_back(corners[c].y);
                    binning.vertices.push_back(corners[c].z);
                    binning.vertices.push_back(reciprocals[c]);
                }
                ++count;
            }
            binning.table.push_back(static_cast<float>(count));
        }
    }
    return binning;
}

Program build_tiled_raster_program(void** args)
{
    const TiledRasterStageArgs& a = *static_cast<const TiledRasterStageArgs*>(args[0]);
    IRBuilder k;

    // The last tile in each direction hangs off the screen, so the bounds check
    // is the same one the untiled kernel needs.
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> half = k.constant(0.5f);

        const Reg<Scalar> cx = k.add(px, half);
        const Reg<Scalar> cy = k.add(py, half);

        // Which tile this block covers — the whole reason blockIdx exists. A
        // global coordinate cannot be divided back down to it.
        //
        // Uniform across the block, so all 32 lanes of a warp load the same two
        // floats. A scalar unit is what real hardware would use for this, and
        // the S_ prefix the ISA reserves is for exactly that.
        const Reg<Scalar> tile = k.add(
            k.mul(k.block_y(), k.constant(static_cast<float>(a.tiles_x))), k.block_x());
        const Reg<Scalar> table_addr = k.mul(tile, k.constant(2.0f * sizeof(float)));

        const float table_base = static_cast<float>(a.tile_table_offset);
        const Reg<Scalar> first = k.load(table_addr, table_base + 0.0f);
        const Reg<Scalar> count = k.load(table_addr, table_base + 4.0f);

        // The running best, as in the untiled kernel: one thread owns one pixel,
        // so nothing is shared and no atomic is needed. Started beyond the far
        // plane so the first covering triangle takes it.
        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // An empty tile has to skip the walk outright. The loop tests its
        // counter at the bottom, so without this guard it would read one
        // triangle's worth of whatever follows the tile's run — and empty tiles
        // are most of the screen, which is both where the saving comes from and
        // where the corruption would be worst.
        k.if_(k.gt(count, zero), [&] {
            const Reg<Scalar> stride =
                k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
            const Reg<Scalar> tri_addr =
                k.add(k.mul(first, stride),
                      k.constant(static_cast<float>(a.tile_vertices_offset)));
            const Reg<Scalar> i = k.constant(0.0f);

            const Label top = k.label();
            k.place(top);

            // Position then 1/w, three times — the layout pass 1 writes and
            // bin_triangles copies.
            const Reg<Scalar> x0 = k.load(tri_addr, 0.0f);
            const Reg<Scalar> y0 = k.load(tri_addr, 4.0f);
            const Reg<Scalar> z0 = k.load(tri_addr, 8.0f);
            const Reg<Scalar> iw0 = k.load(tri_addr, 12.0f);
            const Reg<Scalar> x1 = k.load(tri_addr, 16.0f);
            const Reg<Scalar> y1 = k.load(tri_addr, 20.0f);
            const Reg<Scalar> z1 = k.load(tri_addr, 24.0f);
            const Reg<Scalar> iw1 = k.load(tri_addr, 28.0f);
            const Reg<Scalar> x2 = k.load(tri_addr, 32.0f);
            const Reg<Scalar> y2 = k.load(tri_addr, 36.0f);
            const Reg<Scalar> z2 = k.load(tri_addr, 40.0f);
            const Reg<Scalar> iw2 = k.load(tri_addr, 44.0f);

            const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
            const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
            const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

            const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
            const Reg<Scalar> inv_area = k.rcp(area);
            const Reg<Scalar> w0 = k.mul(e0, inv_area);
            const Reg<Scalar> w1 = k.mul(e1, inv_area);
            const Reg<Scalar> w2 = k.mul(e2, inv_area);

            const Reg<Scalar> inside =
                k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

            const Reg<Scalar> depth = k.mul(w0, z0);
            k.fma(depth, w1, z1);
            k.fma(depth, w2, z2);

            const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));
            // Depth from the affine weights, colour from the corrected ones:
            // NDC z is linear in screen space and an attribute is not.
            k.if_(take, [&] {
                k.copy_into(best_z, depth);
                emit_shade(k, best, w0, w1, w2, iw0, iw1, iw2);
            });

            k.fma(tri_addr, stride, one);
            k.fma(i, one, one);
            k.branch_to(top, k.lt(i, count));
        });

        // One pixel, once, after the whole scene has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });

    return k.build();
}

void run_tiled_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_tiled_raster_stage: an image with no pixels");
    }

    // One block per tile, which is what makes blockIdx the tile index. A block
    // is 256 threads, so eight warps, each covering one row of the tile.
    // Nothing is shared between them — the ISA has no barrier — so they are
    // independent beyond reading the same triangle list.
    const dim3 block{TILE_WIDTH, TILE_HEIGHT, 1};
    const dim3 grid{args.tiles_x, (args.height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1};

    void* raw[] = {const_cast<TiledRasterStageArgs*>(&args)};
    rt.myrt_launch(build_tiled_raster_program, grid, block, raw);
}

// ---------------------------------------------------------------------------
// Tiling, through shared memory
//
// Same frame, same walk. The tile's triangles are staged once per block rather
// than read from global by every pixel — the first kernel here that has two
// warps depend on each other, and so the first that needs BARRIER.
// ---------------------------------------------------------------------------

Program build_shared_raster_program(void** args)
{
    (void)args;

    const TiledRasterStageArgs& a = *static_cast<const TiledRasterStageArgs*>(args[0]);
    IRBuilder k;
    //
    // [a] Which tile, and where its run starts — unchanged from
    //     build_tiled_raster_program:
    //
    const Reg<Scalar> tile =
        k.add(k.mul(k.block_y(), k.constant(static_cast<float>(a.tiles_x))), k.block_x());
    const Reg<Scalar> table_addr = k.mul(tile, k.constant(2.0f * sizeof(float)));

    const float table_base = static_cast<float>(a.tile_table_offset);
    const Reg<Scalar> first = k.load(table_addr, table_base + 0.0f);
    const Reg<Scalar> count = k.load(table_addr, table_base + 4.0f);

    const Reg<Scalar> stride = k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
    const Reg<Scalar> tri_addr = k.add(
        k.mul(first, stride), k.constant(static_cast<float>(a.tile_vertices_offset)));
    // thread_x and thread_y are global coordinates, so the block's own origin
    // has to come off before they can index anything the block owns. Both are
    // kept: the pixel work below still wants them as they are.
    const Reg<Scalar> tile_w = k.constant(static_cast<float>(TILE_WIDTH));
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> tx = k.sub(px, k.mul(k.block_x(), tile_w));
    const Reg<Scalar> ty =
        k.sub(py, k.mul(k.block_y(), k.constant(static_cast<float>(TILE_HEIGHT))));

    // 0 .. 255, and the stride the cooperative fill steps by.
    const Reg<Scalar> lane = k.add(k.mul(ty, tile_w), tx);

    // The fill. A screen triangle a triangle — position and 1/w three times over
    // — shared out across the block's 256 threads: each takes the float at its
    // own lane index, then every 256th one after that.
    const Reg<Scalar> one = k.constant(1.0f);
    const Reg<Scalar> four = k.constant(4.0f);
    const Reg<Scalar> block_threads =
        k.constant(static_cast<float>(TILE_WIDTH * TILE_HEIGHT));
    const Reg<Scalar> staged =
        k.mul(count, k.constant(static_cast<float>(TILE_TRIANGLE_FLOATS)));

    // A copy, because fma advances the cursor in place and a loop counter that
    // doubles as the thread's identity reads badly.
    const Reg<Scalar> cursor = k.copy(lane);

    // Rotated rather than wrapped in if_: a lane with nothing to stage leaves
    // before the body instead of entering it against a guard, and the rest drop
    // out one at a time as the cursor passes the end. min-PC keeps issuing the
    // body for whoever is left, and they all meet again at fill_done.
    const Label fill_done = k.label();
    const Label fill_top = k.label();

    k.branch_to(fill_done, k.ge(cursor, staged));
    k.place(fill_top);

    const Reg<Scalar> byte = k.mul(cursor, four);
    k.store_shared(byte, k.load(k.add(tri_addr, byte), 0.0f), 0.0f);

    k.fma(cursor, block_threads, one);
    k.branch_to(fill_top, k.lt(cursor, staged));
    k.place(fill_done);

    // Everything above runs for every thread of the block, including the ones
    // whose pixel is off screen in an edge tile. A thread that branched past
    // this would be one the rest wait for and never see, and the scheduler
    // refuses that rather than hanging.
    k.barrier();

    // From here the walk is build_tiled_raster_program's, with load_shared in
    // place of load and a cursor that starts at zero — the tile's run begins at
    // the front of shared memory however far into the buffer it sat.
    const Reg<Scalar> zero = k.constant(0.0f);
    const Reg<Scalar> half = k.constant(0.5f);
    const Reg<Scalar> cx = k.add(px, half);
    const Reg<Scalar> cy = k.add(py, half);

    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        k.if_(k.gt(count, zero), [&] {
            const Reg<Scalar> shared_stride =
                k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
            const Reg<Scalar> shared_addr = k.copy(zero);
            const Reg<Scalar> i = k.copy(zero);

            const Label top = k.label();
            k.place(top);

            // Position then 1/w, three times — the layout pass 1 writes and
            // bin_triangles copies.
            const Reg<Scalar> x0 = k.load_shared(shared_addr, 0.0f);
            const Reg<Scalar> y0 = k.load_shared(shared_addr, 4.0f);
            const Reg<Scalar> z0 = k.load_shared(shared_addr, 8.0f);
            const Reg<Scalar> iw0 = k.load_shared(shared_addr, 12.0f);
            const Reg<Scalar> x1 = k.load_shared(shared_addr, 16.0f);
            const Reg<Scalar> y1 = k.load_shared(shared_addr, 20.0f);
            const Reg<Scalar> z1 = k.load_shared(shared_addr, 24.0f);
            const Reg<Scalar> iw1 = k.load_shared(shared_addr, 28.0f);
            const Reg<Scalar> x2 = k.load_shared(shared_addr, 32.0f);
            const Reg<Scalar> y2 = k.load_shared(shared_addr, 36.0f);
            const Reg<Scalar> z2 = k.load_shared(shared_addr, 40.0f);
            const Reg<Scalar> iw2 = k.load_shared(shared_addr, 44.0f);

            const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
            const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
            const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

            const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
            const Reg<Scalar> inv_area = k.rcp(area);
            const Reg<Scalar> w0 = k.mul(e0, inv_area);
            const Reg<Scalar> w1 = k.mul(e1, inv_area);
            const Reg<Scalar> w2 = k.mul(e2, inv_area);

            const Reg<Scalar> inside =
                k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

            const Reg<Scalar> depth = k.mul(w0, z0);
            k.fma(depth, w1, z1);
            k.fma(depth, w2, z2);

            const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));
            // Depth from the affine weights, colour from the corrected ones:
            // NDC z is linear in screen space and an attribute is not.
            k.if_(take, [&] {
                k.copy_into(best_z, depth);
                emit_shade(k, best, w0, w1, w2, iw0, iw1, iw2);
            });

            k.fma(shared_addr, shared_stride, one);
            k.fma(i, one, one);
            k.branch_to(top, k.lt(i, count));
        });

        // One pixel, once, after the whole scene has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });

    return k.build();
}

void run_shared_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_shared_raster_stage: an image with no pixels");
    }

    // Real hardware splits an overfull tile across passes. Doing that here
    // would have the fill, the barrier and the walk repeat per pass, which is a
    // different kernel; refusing is honest until a scene needs it.
    if (args.max_tile_triangles > SHARED_TRIANGLE_CAPACITY) {
        throw std::runtime_error("run_shared_raster_stage: a tile holds " +
                                 std::to_string(args.max_tile_triangles) +
                                 " triangles, and shared memory " + "stages " +
                                 std::to_string(SHARED_TRIANGLE_CAPACITY));
    }

    const dim3 block{TILE_WIDTH, TILE_HEIGHT, 1};
    const dim3 grid{args.tiles_x, (args.height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1};

    void* raw[] = {const_cast<TiledRasterStageArgs*>(&args)};
    rt.myrt_launch(build_shared_raster_program, grid, block, raw);
}

// ---------------------------------------------------------------------------
// Ray tracing
//
// The comparison the rasteriser exists to be measured against. Same scene, same
// image, and divergence from a different cause: the rasteriser splits a warp at
// a triangle's edge, the ray tracer wherever one lane's ray leaves early.
// ---------------------------------------------------------------------------

RayBasis ray_basis(const Camera& camera, float aspect)
{
    // Same handedness as look_at, and it has to be: the two cameras are
    // compared by rendering one scene through both.
    const Float3 forward = normalize(camera.target - camera.eye);
    const Float3 right = normalize(cross(forward, camera.up));
    const Float3 up = cross(right, forward);

    // Folding the field of view in here is what leaves the kernel with three
    // scales and two adds.
    const float half_extent = std::tan(radians(camera.fov_y_degrees) * 0.5f);

    return RayBasis{camera.eye, right * (half_extent * aspect), up * half_extent,
                    forward};
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

Program build_raytrace_program(void** args)
{
    const RaytraceStageArgs& a = *static_cast<const RaytraceStageArgs*>(args[0]);
    IRBuilder k;

    // Twelve moves, uniform across the launch — every lane ends up holding its
    // own copy of the same twelve numbers, as it does for the vertex stage's
    // matrix and for the tile table. The redundancy a scalar unit exists to
    // avoid, and the third place it turns up.
    const RayBasis& b = a.basis;
    const Reg<Vec3> origin = k.constant(b.origin.x, b.origin.y, b.origin.z);
    const Reg<Vec3> right = k.constant(b.right.x, b.right.y, b.right.z);
    const Reg<Vec3> up = k.constant(b.up.x, b.up.y, b.up.z);
    const Reg<Vec3> forward = k.constant(b.forward.x, b.forward.y, b.forward.z);
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        // The half-pixel offset and the scale fold into one constant each, as
        // kernels/ray_triangle.cpp does: two multiplies and two adds rather
        // than six. sy runs backwards because rows count down and the frame up.
        const Reg<Scalar> sx =
            k.add(k.mul(px, k.constant(2.0f / static_cast<float>(a.width))),
                  k.constant(1.0f / static_cast<float>(a.width) - 1.0f));
        const Reg<Scalar> sy =
            k.add(k.mul(py, k.constant(-2.0f / static_cast<float>(a.height))),
                  k.constant(1.0f - 1.0f / static_cast<float>(a.height)));
        const Reg<Vec3> dir = k.add(k.scale(right, sx), k.add(k.scale(up, sy), forward));
        // The running best, as in the rasteriser. Infinity rather than the
        // rasteriser's 2.0f: that worked because NDC depth is bounded, and t is
        // a ray parameter with no ceiling.
        const Reg<Scalar> best_t = k.constant(std::numeric_limits<float>::infinity());
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // One run of world triangles. count is a host value, so unlike the tiled
        // rasteriser there is no table to read first — and run_raytrace_stage
        // refuses a count of zero, so the loop can test at the bottom without a
        // guard.
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> eps = k.constant(INTERSECT_EPSILON);
        const Reg<Scalar> stride = k.constant(static_cast<float>(3 * WORLD_VERTEX_BYTES));
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));
        const Reg<Scalar> cursor = k.constant(static_cast<float>(a.triangles_offset));
        const Reg<Scalar> i = k.constant(0.0f);

        // Above k.place(top) because a constant below it is issued once per
        // triangle. Emitted whatever the mode: six moves outside the loop cost
        // less than the optional it would take to declare them conditionally.
        const Float3& lp = a.shading.light_position;
        const Float3& bc = a.shading.base_colour;
        const Reg<Vec3> light = k.constant(lp.x, lp.y, lp.z);
        const Reg<Vec3> base = k.constant(bc.x, bc.y, bc.z);

        // Every miss lands on next, and it is placed before the advance below:
        // a miss means "on to the next triangle", not "out of the loop", so the
        // cursor still has to move or the lane meets the same one for ever.
        const Label top = k.label();
        const Label next = k.label();
        k.place(top);

        // The vertices have to arrive as Vec3 for cross and dot to take them,
        // and load hands back a scalar it allocated itself. load_into is what
        // puts three of them in one range.
        const Reg<Vec3> v0 = k.vec3();
        const Reg<Vec3> v1 = k.vec3();
        const Reg<Vec3> v2 = k.vec3();
        for (uint32_t c = 0; c < 3; ++c) {
            const float component = static_cast<float>(c * sizeof(float));
            k.load_into(v0.component(c), cursor, component + 0.0f);
            k.load_into(v1.component(c), cursor, component + 12.0f);
            k.load_into(v2.component(c), cursor, component + 24.0f);
        }

        // Möller-Trumbore, a line for each line of intersect().
        const Reg<Vec3> e1 = k.sub(v1, v0);
        const Reg<Vec3> e2 = k.sub(v2, v0);
        const Reg<Vec3> h = k.cross(dir, e2);
        const Reg<Scalar> a_dot = k.dot(e1, h);

        // a < eps rather than |a| < eps rejects a back face along with a
        // parallel ray, the ISA having no absolute value.
        //
        // First of the four exits. Six conditions, but any() folds each pair
        // into one flag, so a warp splits four times rather than six.
        k.branch_to(next, k.lt(a_dot, eps));

        const Reg<Scalar> f = k.rcp(a_dot);
        const Reg<Vec3> s = k.sub(origin, v0);

        const Reg<Scalar> u = k.mul(f, k.dot(s, h));
        k.branch_to(next, k.any(k.lt(u, zero), k.gt(u, one)));

        const Reg<Vec3> q = k.cross(s, e1);
        const Reg<Scalar> v = k.mul(f, k.dot(dir, q));
        k.branch_to(next, k.any(k.lt(v, zero), k.gt(k.add(u, v), one)));

        const Reg<Scalar> t = k.mul(f, k.dot(e2, q));
        k.branch_to(next, k.lt(t, eps));

        // Nearest wins. No depth buffer and no atomics: the thread owns its
        // pixel, so the running best is a register.
        k.if_(k.lt(t, best_t), [&] {
            k.copy_into(best_t, t);

            // Read at build time, not by the device: a KernelFunc runs once per
            // launch, so only one of these arms reaches the instruction stream.
            if (a.shading.mode == ShadingMode::Diffuse) {
                // Shaded inside the loop, so a pixel that meets three triangles
                // shades three times. Carrying a normal and a point out to the
                // end would cost six registers to save work the twelve loads
                // above already dominate.

                // e1 and e2 are the intersection test's, not rebuilt from the
                // vertices: a normal wound off different edges than a_dot was
                // would cull one face and shade the other.
                const Reg<Vec3> normal = k.normalize(k.cross(e1, e2));

                // dir is not normalised and does not need to be — t is measured
                // in units of it, so the two cancel.
                const Reg<Vec3> point = k.add(origin, k.scale(dir, t));
                const Reg<Vec3> to_light = k.normalize(k.sub(light, point));
                const Reg<Scalar> diffuse = k.max(zero, k.dot(normal, to_light));
                const Reg<Vec3> shaded = k.scale(base, diffuse);

                k.copy_into(best.component(0), shaded.component(0));
                k.copy_into(best.component(1), shaded.component(1));
                k.copy_into(best.component(2), shaded.component(2));
            } else {
                k.copy_into(best.component(0), u);
                k.copy_into(best.component(1), v);
                k.copy_into(best.component(2), k.sub(k.sub(one, u), v));
            }
        });

        k.place(next);
        k.fma(cursor, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

        // One pixel, once, after the whole scene has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));
        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });
    return k.build();
}

void run_raytrace_stage(MyGPURuntime& rt, const RaytraceStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_raytrace_stage: an image with no pixels");
    }
    if (args.triangle_count == 0) {
        throw std::runtime_error("run_raytrace_stage: nothing to trace");
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE, args.height, 1};

    void* raw[] = {const_cast<RaytraceStageArgs*>(&args)};
    rt.myrt_launch(build_raytrace_program, grid, block, raw);
}
