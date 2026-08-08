#include <cstdint>
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

}  // namespace

// ---------------------------------------------------------------------------
// [1] Host reference
//
// Do this first. It is a dozen lines of arithmetic with no ISA in the way, and
// once it passes its tests the device version has something to be wrong
// against — which is the only practical way to debug a kernel here.
// ---------------------------------------------------------------------------

Float3 project_vertex(const Float4x4& view_projection, Float3 world, uint32_t width,
                      uint32_t height)
{
    Float3 screen;
    const Float4 clip = transform(view_projection, world, 1.0f);

    const float inv_w = 1.0f / clip.w;
    const Float3 ndc{clip.x * inv_w, clip.y * inv_w, clip.z * inv_w};

    screen.x = ((ndc.x + 1.0f) * 0.5f) * width;
    screen.y = (0.5f - ndc.y * 0.5f) * height;
    screen.z = ndc.z;

    return screen;
}

// ---------------------------------------------------------------------------
// [2] Pass 1 as a program
//
// The first kernel written entirely with IRBuilder, and the first use of
// V_MATVEC_MAT4_F32 outside its own test.
// ---------------------------------------------------------------------------

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

        // One address register serves both buffers, which holds only while the
        // strides agree.
        static_assert(WORLD_VERTEX_BYTES == SCREEN_VERTEX_BYTES,
                      "reusing addr for the store needs both strides to match");
        const float screen_base = static_cast<float>(a.screen_offset);
        k.store(addr, sx, screen_base + 0.0f);
        k.store(addr, sy, screen_base + 4.0f);
        k.store(addr, sz, screen_base + 8.0f);
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

// ---------------------------------------------------------------------------
// [3] Pass 2 — host reference
//
// Do this first, as with pass 1. Coverage has three sign conventions stacked on
// each other (winding, the y flip, pixel centres) and getting one wrong renders
// a picture that looks deliberate.
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

Float3 shade_pixel(Float3 v0, Float3 v1, Float3 v2, uint32_t px, uint32_t py)
{
    const float cx = static_cast<float>(px) + 0.5f;
    const float cy = static_cast<float>(py) + 0.5f;

    const Float3 w = barycentric(v0, v1, v2, cx, cy);

    // Tested the positive way round on purpose: a degenerate triangle makes
    // every weight NaN, and NaN fails >= as well as <, so it falls through to
    // the background instead of writing NaN into the frame.
    if (w.x >= 0.0f && w.y >= 0.0f && w.z >= 0.0f) {
        // (w1, w2, w0). The ray tracer colours a hit (u, v, 1 - u - v), where u
        // weights v1 and v weights v2 — so this ordering is what makes the two
        // renderers produce the same picture rather than one rotated in hue.
        return Float3{w.y, w.z, w.x};
    }
    return Float3{0.0f, 0.0f, 0.0f};
}

// ---------------------------------------------------------------------------
// [4] Pass 2 as a program
//
// The first kernel whose lanes take different paths for reasons that are not a
// bounds check. A warp covers 32 horizontally adjacent pixels, so a triangle
// edge crossing it splits the warp, and step 7 measured what that costs: one
// diverged lane is as expensive as sixteen.
// ---------------------------------------------------------------------------

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
        // The triangle's address is the same for every lane, so it is a host
        // constant that rides entirely in the loads' immediates — no register
        // and no arithmetic. Real hardware would broadcast these six values
        // from a scalar unit; here all 32 lanes issue the same load, the same
        // redundancy the sixteen matrix moves cost pass 1.
        //
        // Only x and y are read. The depth pass 1 wrote is what a second
        // triangle would be compared on, and there is only one.
        const float base =
            static_cast<float>(a.screen_offset + static_cast<size_t>(a.triangle_index) *
                                                     3 * SCREEN_VERTEX_BYTES);

        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> x0 = k.load(zero, base + 0.0f);
        const Reg<Scalar> y0 = k.load(zero, base + 4.0f);
        const Reg<Scalar> x1 = k.load(zero, base + 12.0f);
        const Reg<Scalar> y1 = k.load(zero, base + 16.0f);
        const Reg<Scalar> x2 = k.load(zero, base + 24.0f);
        const Reg<Scalar> y2 = k.load(zero, base + 28.0f);

        // The pixel centre, matching what the ray tracer samples.
        const Reg<Scalar> half = k.constant(0.5f);
        const Reg<Scalar> cx = k.add(px, half);
        const Reg<Scalar> cy = k.add(py, half);

        // [d] Three edge functions, using emit_edge above. Each weight is the
        //     edge opposite its vertex, as in barycentric():
        //
        //       e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
        //       e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
        //       e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);
        //
        const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
        const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
        const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

        // [e] area = e0 + e1 + e2, one k.rcp of it, three k.mul. Normalising
        //     rather than testing the signs directly is what keeps the winding
        //     from mattering — pass 1 reverses it on every triangle.
        //
        const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
        const Reg<Scalar> inv_area = k.rcp(area);
        const Reg<Scalar> w0 = k.mul(e0, inv_area);
        const Reg<Scalar> w1 = k.mul(e1, inv_area);
        const Reg<Scalar> w2 = k.mul(e2, inv_area);

        // [f] Coverage, and the divergence point:
        //
        //       inside = min(min(ge(w0, zero), ge(w1, zero)), ge(w2, zero))
        //       k.if_else(inside, colour is (w1, w2, w0), colour is black)
        //
        //     Tested the positive way round, as shade_pixel is, so a degenerate
        //     triangle's NaNs fail the comparison and leave the background.
        //
        //     A predicated form — selecting with V_MIN/V_MAX instead of
        //     branching — is the comparison the README promises, and belongs in
        //     its own change so the two can be measured against each other.
        //
        const Reg<Scalar> inside =
            k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

        // Allocated before the branch, because both arms have to leave their
        // answer in the same place for the stores below to find it.
        const Reg<Vec3> colour = k.vec3();
        k.if_else(
            inside,
            [&] {
                // (w1, w2, w0) — v0 blue, v1 red, v2 green, as the ray tracer
                // colours a hit.
                k.copy_into(colour.component(0), w1);
                k.copy_into(colour.component(1), w2);
                k.copy_into(colour.component(2), w0);
            },
            [&] {
                k.set(colour.component(0), 0.0f);
                k.set(colour.component(1), 0.0f);
                k.set(colour.component(2), 0.0f);
            });

        // The stores sit after the join, so every in-image lane runs them
        // together: the divergence above is the cost of choosing a colour, not
        // of writing one. Putting them inside the arms instead would double
        // what a split warp pays and make the branch look worse than it is when
        // a predicated version is measured against it.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, colour.component(0), frame_base + 0.0f);
        k.store(addr, colour.component(1), frame_base + 4.0f);
        k.store(addr, colour.component(2), frame_base + 8.0f);
    });

    // Registers: nine for the triangle, a handful of scalars, and no matrix.
    // Well under what pass 1 needs, which is the register pressure the two-pass
    // split was meant to relieve.
    return k.build();
}

void run_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_raster_stage: an image with no pixels");
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
