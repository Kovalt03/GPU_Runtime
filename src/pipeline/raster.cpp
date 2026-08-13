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

Program build_indexed_raster_program(void** args)
{
    const RasterStageArgs& a = *static_cast<const RasterStageArgs*>(args[0]);
    IRBuilder k;

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

        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // The cursor walks the index buffer: three entries a triangle, four
        // bytes each, where the flattened kernel stepped 48 through the
        // vertices themselves.
        const Reg<Scalar> idx_addr = k.constant(static_cast<float>(a.index_offset));
        const Reg<Scalar> stride = k.constant(static_cast<float>(3 * sizeof(float)));

        // screen_offset stops being a cursor and becomes a base: an index names
        // a vertex, and that vertex sits at base + index * stride. Neither
        // moves, so both are emitted out here — a constant below k.place(top)
        // is a move per triangle.

        const Reg<Scalar> screen_base = k.constant(static_cast<float>(a.screen_offset));
        const Reg<Scalar> vertex_bytes =
            k.constant(static_cast<float>(SCREEN_VERTEX_BYTES));

        const Reg<Scalar> i = k.constant(0.0f);
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));

        const Label top = k.label();
        k.place(top);

        // The dependent load the flattened kernel does not pay: a vertex's
        // address is not known until its index has arrived, so this is two
        // loads deep, three times over.
        //
        // An index counts vertices, not bytes — hence the multiply. The host
        // makes the same conversion in read_back_triangles, by
        // SCREEN_VERTEX_FLOATS rather than SCREEN_VERTEX_BYTES, addresses there
        // being float positions.
        const Reg<Scalar> i0 = k.load(idx_addr, 0.0f);
        const Reg<Scalar> v0 = k.add(screen_base, k.mul(i0, vertex_bytes));
        const Reg<Scalar> i1 = k.load(idx_addr, 4.0f);
        const Reg<Scalar> v1 = k.add(screen_base, k.mul(i1, vertex_bytes));
        const Reg<Scalar> i2 = k.load(idx_addr, 8.0f);
        const Reg<Scalar> v2 = k.add(screen_base, k.mul(i2, vertex_bytes));

        // Four loads per vertex off its own base, where the flattened kernel
        // read twelve off one. Fifteen global loads a triangle against twelve
        // is the whole cost of indexing on this pass.
        const Reg<Scalar> x0 = k.load(v0, 0.0f);
        const Reg<Scalar> y0 = k.load(v0, 4.0f);
        const Reg<Scalar> z0 = k.load(v0, 8.0f);
        const Reg<Scalar> iw0 = k.load(v0, 12.0f);

        const Reg<Scalar> x1 = k.load(v1, 0.0f);
        const Reg<Scalar> y1 = k.load(v1, 4.0f);
        const Reg<Scalar> z1 = k.load(v1, 8.0f);
        const Reg<Scalar> iw1 = k.load(v1, 12.0f);

        const Reg<Scalar> x2 = k.load(v2, 0.0f);
        const Reg<Scalar> y2 = k.load(v2, 4.0f);
        const Reg<Scalar> z2 = k.load(v2, 8.0f);
        const Reg<Scalar> iw2 = k.load(v2, 12.0f);

        // From here to the store this is build_raster_program line for line,
        // and its comments are not repeated — read them there. Copied rather
        // than shared because the two kernels are what the measurement
        // compares, and a helper that drifted would take both readings with it.
        // emit_edge and emit_shade already hold the parts worth sharing.

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
        k.if_(take, [&] {
            k.copy_into(best_z, depth);
            emit_shade(k, best, w0, w1, w2, iw0, iw1, iw2);
        });

        k.fma(idx_addr, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

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

// Same launch geometry as run_raster_stage, and for the same reason: 32 lanes
// on 32 adjacent pixels of one row is what makes an edge split a warp.
void run_indexed_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_indexed_raster_stage: an image with no pixels");
    }
    if (args.triangle_count == 0) {
        throw std::runtime_error("run_indexed_raster_stage: nothing to draw");
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE, args.height, 1};

    void* raw[] = {const_cast<RasterStageArgs*>(&args)};
    rt.myrt_launch(build_indexed_raster_program, grid, block, raw);
}

Program build_predicated_raster_program(void** args)
{
    const RasterStageArgs& a = *static_cast<const RasterStageArgs*>(args[0]);
    IRBuilder k;

    // The bounds check stays a branch. Predicating it would leave every lane
    // computing a pixel it must not write, and the store at the end has no flag
    // to suppress — this ISA has no predicated store.
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

        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        const Reg<Scalar> tri_addr = k.constant(static_cast<float>(a.screen_offset));
        const Reg<Scalar> stride =
            k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
        const Reg<Scalar> i = k.constant(0.0f);
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));

        const Label top = k.label();
        k.place(top);

        // Everything down to `take` is build_raster_program line for line, and
        // its comments are not repeated here. The two kernels decide coverage
        // identically and differ only in what they do with the answer.
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

        // Shaded for every lane, and into a scratch range rather than into
        // best: emit_shade overwrites what it is handed, and the blend below
        // still needs the old value.
        //
        // This is the cost. A lane the triangle does not cover runs the whole
        // shade and then multiplies it away.
        const Reg<Vec3> shaded = k.vec3();
        emit_shade(k, shaded, w0, w1, w2, iw0, iw1, iw2);

        // old + take * (new - old) is the same select an instruction cheaper,
        // and is not exact: subtracting old and adding it back rounds, so a
        // taken blend lands near new rather than on it. Sixty-four triangles of
        // that drift changed the image.
        const Reg<Scalar> keep = k.sub(one, take);
        const auto blend = [&](Reg<Scalar> dst, Reg<Scalar> src) {
            k.copy_into(dst, k.mul(dst, keep));
            k.fma(dst, take, src);
        };
        blend(best_z, depth);
        blend(best.component(0), shaded.component(0));
        blend(best.component(1), shaded.component(1));
        blend(best.component(2), shaded.component(2));

        k.fma(tri_addr, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

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

// Same launch geometry as run_raster_stage: 32 lanes on 32 adjacent pixels of
// one row is what puts a triangle edge inside a warp, and this variant exists
// to be measured against that.
void run_predicated_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_predicated_raster_stage: an image with no pixels");
    }
    if (args.triangle_count == 0) {
        throw std::runtime_error("run_predicated_raster_stage: nothing to draw");
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE, args.height, 1};

    void* raw[] = {const_cast<RasterStageArgs*>(&args)};
    rt.myrt_launch(build_predicated_raster_program, grid, block, raw);
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
