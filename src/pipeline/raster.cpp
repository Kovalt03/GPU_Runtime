#include <cstdint>
#include <functional>
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

namespace {

// Three screen vertices, however the kernel got hold of them. Fetching them is
// the one thing the indexed and flattened forms do differently; everything
// downstream reads this.
struct Triangle {
    Reg<Scalar> x0, y0, z0, iw0;
    Reg<Scalar> x1, y1, z1, iw1;
    Reg<Scalar> x2, y2, z2, iw2;
};

}  // namespace

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
        //
        // Indexed, it walks the index buffer at three entries a triangle;
        // flattened, the screen buffer at three vertices. Which one decides the
        // stride and what fetch below has to do to reach a vertex.
        const Reg<Scalar> cursor =
            k.constant(static_cast<float>(a.indexed ? a.index_offset : a.screen_offset));
        const Reg<Scalar> stride = k.constant(
            static_cast<float>(a.indexed ? 3 * sizeof(float) : 3 * SCREEN_VERTEX_BYTES));
        const Reg<Scalar> i = k.constant(0.0f);
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));

        // Above the loop, so a constant is issued once for the launch rather
        // than once a triangle. Emitted whatever the mode, six moves being
        // cheaper than the branch it would take to declare them conditionally.
        const bool diffuse = a.shading.mode == ShadingMode::Diffuse;
        const Reg<Vec3> light =
            k.constant(a.shading.light_position.x, a.shading.light_position.y,
                       a.shading.light_position.z);
        const Reg<Vec3> base_colour = k.constant(
            a.shading.base_colour.x, a.shading.base_colour.y, a.shading.base_colour.z);
        const Reg<Scalar> world_base = k.constant(static_cast<float>(a.world_offset));
        const Reg<Scalar> normal_base = k.constant(static_cast<float>(a.normal_offset));
        const Reg<Scalar> world_stride =
            k.constant(static_cast<float>(WORLD_VERTEX_BYTES));
        const Reg<Scalar> normal_stride =
            k.constant(static_cast<float>(FACE_NORMAL_BYTES));

        // Chosen here, on the host, so only one form reaches the instruction
        // stream — a KernelFunc runs once per launch and the flag costs no lane
        // anything. Whatever constants a form needs are emitted as it is built,
        // which is above the loop, so the other form does not pay for them.
        std::function<Triangle()> fetch;
        if (a.indexed) {
            // screen_offset stops being a cursor and becomes a base: an index
            // names a vertex, and that vertex sits at base + index * stride.
            const Reg<Scalar> screen_base =
                k.constant(static_cast<float>(a.screen_offset));
            const Reg<Scalar> vertex_bytes =
                k.constant(static_cast<float>(SCREEN_VERTEX_BYTES));

            fetch = [&k, cursor, screen_base, vertex_bytes] {
                // The dependent load the flattened form does not pay: a
                // vertex's address is not known until its index has arrived, so
                // this is two loads deep, three times over.
                //
                // An index counts vertices, not bytes — hence the multiply. The
                // host makes the same conversion in read_back_triangles, by
                // SCREEN_VERTEX_FLOATS rather than SCREEN_VERTEX_BYTES,
                // addresses there being float positions.
                const Reg<Scalar> i0 = k.load(cursor, 0.0f);
                const Reg<Scalar> v0 = k.add(screen_base, k.mul(i0, vertex_bytes));
                const Reg<Scalar> i1 = k.load(cursor, 4.0f);
                const Reg<Scalar> v1 = k.add(screen_base, k.mul(i1, vertex_bytes));
                const Reg<Scalar> i2 = k.load(cursor, 8.0f);
                const Reg<Scalar> v2 = k.add(screen_base, k.mul(i2, vertex_bytes));

                // Four loads per vertex off its own base, where the flattened
                // form reads twelve off one. Fifteen global loads a triangle
                // against twelve is the whole cost of indexing on this pass.
                Triangle t;
                t.x0 = k.load(v0, 0.0f);
                t.y0 = k.load(v0, 4.0f);
                t.z0 = k.load(v0, 8.0f);
                t.iw0 = k.load(v0, 12.0f);
                t.x1 = k.load(v1, 0.0f);
                t.y1 = k.load(v1, 4.0f);
                t.z1 = k.load(v1, 8.0f);
                t.iw1 = k.load(v1, 12.0f);
                t.x2 = k.load(v2, 0.0f);
                t.y2 = k.load(v2, 4.0f);
                t.z2 = k.load(v2, 8.0f);
                t.iw2 = k.load(v2, 12.0f);
                return t;
            };
        } else {
            fetch = [&k, cursor] {
                // Position then 1/w, three times — the layout pass 1 writes and
                // bin_triangles copies.
                Triangle t;
                t.x0 = k.load(cursor, 0.0f);
                t.y0 = k.load(cursor, 4.0f);
                t.z0 = k.load(cursor, 8.0f);
                t.iw0 = k.load(cursor, 12.0f);
                t.x1 = k.load(cursor, 16.0f);
                t.y1 = k.load(cursor, 20.0f);
                t.z1 = k.load(cursor, 24.0f);
                t.iw1 = k.load(cursor, 28.0f);
                t.x2 = k.load(cursor, 32.0f);
                t.y2 = k.load(cursor, 36.0f);
                t.z2 = k.load(cursor, 40.0f);
                t.iw2 = k.load(cursor, 44.0f);
                return t;
            };
        }

        // The loop splits no warp: every lane runs the same number of
        // iterations and reaches the backward branch together. Only the
        // coverage test inside it diverges, and only when the kernel branches.
        //
        // Every lane also loads the same twelve floats, the triangle not
        // depending on the pixel. Real hardware broadcasts these from a scalar
        // unit; here all 32 lanes issue the same load — the redundancy the
        // sixteen matrix moves already cost pass 1, and the reason the ISA
        // reserves an S_ prefix.
        const Label top = k.label();
        k.place(top);

        const Triangle t = fetch();

        // Each weight is the edge opposite its vertex, as in barycentric().
        const Reg<Scalar> e0 = emit_edge(k, t.x1, t.y1, t.x2, t.y2, cx, cy);
        const Reg<Scalar> e1 = emit_edge(k, t.x2, t.y2, t.x0, t.y0, cx, cy);
        const Reg<Scalar> e2 = emit_edge(k, t.x0, t.y0, t.x1, t.y1, cx, cy);

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

        const Reg<Scalar> depth = k.mul(w0, t.z0);
        k.fma(depth, w1, t.z1);
        k.fma(depth, w2, t.z2);

        // Both conditions folded into one flag: covered, and nearer than
        // anything kept so far.
        const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));

        // What a kept pixel is coloured with. Barycentric reads nothing beyond
        // the weights; Diffuse reads the world vertices pass 1 projected and the
        // triangle's normal, both of which are still on the device because the
        // geometry is.
        const auto shade = [&](Reg<Vec3> dst) {
            if (!diffuse) {
                emit_shade(k, dst, w0, w1, w2, t.iw0, t.iw1, t.iw2);
                return;
            }

            // The same perspective correction emit_shade applies to a colour.
            // A world position is an attribute like any other: affine weights
            // interpolate it wrongly across a projected triangle.
            const Reg<Scalar> a0 = k.mul(w0, t.iw0);
            const Reg<Scalar> a1 = k.mul(w1, t.iw1);
            const Reg<Scalar> a2 = k.mul(w2, t.iw2);
            const Reg<Scalar> inv_total = k.rcp(k.add(k.add(a0, a1), a2));

            // Three vertices of world space, wherever this form keeps them.
            // Flattened, a triangle is three consecutive vertices; indexed, the
            // indices have to be read again — the dependent load that form pays
            // everywhere else, here once more.
            Reg<Scalar> p0 = world_base;
            Reg<Scalar> p1 = world_base;
            Reg<Scalar> p2 = world_base;
            if (a.indexed) {
                p0 = k.add(world_base, k.mul(k.load(cursor, 0.0f), world_stride));
                p1 = k.add(world_base, k.mul(k.load(cursor, 4.0f), world_stride));
                p2 = k.add(world_base, k.mul(k.load(cursor, 8.0f), world_stride));
            } else {
                const Reg<Scalar> triangle_base =
                    k.add(world_base, k.mul(i, k.constant(3.0f * WORLD_VERTEX_BYTES)));
                p0 = triangle_base;
                p1 = k.add(triangle_base, world_stride);
                p2 = k.add(triangle_base, k.mul(world_stride, k.constant(2.0f)));
            }

            const Reg<Vec3> v0 = k.load_vec3(p0);
            const Reg<Vec3> v1 = k.load_vec3(p1);
            const Reg<Vec3> v2 = k.load_vec3(p2);

            Reg<Vec3> point = k.scale(v0, k.mul(a0, inv_total));
            point = k.add(point, k.scale(v1, k.mul(a1, inv_total)));
            point = k.add(point, k.scale(v2, k.mul(a2, inv_total)));

            // Read rather than derived. The ray tracer takes the cross product
            // of edges it already holds; this kernel would have to load three
            // vertices to do the same, and the host has already done it once.
            const Reg<Vec3> normal =
                k.load_vec3(k.add(normal_base, k.mul(i, normal_stride)));

            const Reg<Vec3> to_light = k.normalize(k.sub(light, point));
            const Reg<Scalar> lambert = k.max(zero, k.dot(normal, to_light));
            const Reg<Vec3> lit = k.scale(base_colour, lambert);
            k.copy_into(dst.component(0), lit.component(0));
            k.copy_into(dst.component(1), lit.component(1));
            k.copy_into(dst.component(2), lit.component(2));
        };

        emit_keep(k, a.predicated, take, best_z, best, depth, one, shade);

        k.fma(cursor, stride, one);
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
