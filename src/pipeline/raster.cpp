#include <cstdint>
#include <stdexcept>

#include "ir_builder.hpp"
#include "pipeline/raster.hpp"
#include "raster_emit.hpp"
#include "thread.hpp"  // WARP_SIZE, the launch width

// ---------------------------------------------------------------------------
// Pass 2 — coverage
//
// Three sign conventions stack up here — the triangle's winding, the y flip
// pass 1 applied, and where inside a pixel the sample sits — and each of them
// wrong still renders a picture that looks deliberate. Checked against
// shade_pixel in tests/reference.hpp, by tests written as geometric facts
// rather than as expected numbers.
// ---------------------------------------------------------------------------

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
