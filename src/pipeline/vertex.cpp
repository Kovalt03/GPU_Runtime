#include <cstdint>
#include <stdexcept>

#include "ir_builder.hpp"
#include "math3d.hpp"
#include "pipeline/vertex.hpp"
#include "thread.hpp"  // WARP_SIZE, the launch width

namespace {

// Rows and columns of the view-projection matrix, so the walk that bakes it
// into registers reads as the row-major rule rather than as a bare 4.
constexpr uint32_t MATRIX_DIMENSION = 4;

}  // namespace

// ---------------------------------------------------------------------------
// Pass 1 — projection
//
// One thread per vertex. Checked against project_vertex in
// tests/reference.hpp, which is arithmetic with no ISA in the way and so has
// something to be wrong against.
// ---------------------------------------------------------------------------

// The same projection as a program: the first kernel written entirely with
// IRBuilder, and the first use of V_MATVEC_MAT4_F32 outside its own test.

Program build_vertex_program(void** args)
{
    const VertexStageArgs& a = *static_cast<const VertexStageArgs*>(args[0]);
    IRBuilder k;

    // The uniform, either baked in as sixteen moves or read as one instruction
    // from the constant window. Row-major on both sides, so no transpose belongs
    // here — if a render comes out transposed, this is the first suspect.
    //
    // The two forms compute the same frame, and the baked one is what every
    // figure taken before the window existed used.
    const Reg<Mat4> mvp = [&] {
        if (a.uniform_offset != 0) {
            return k.load_const_mat4(k.const_base());
        }
        const Reg<Mat4> baked = k.mat4();
        for (uint32_t row = 0; row < MATRIX_DIMENSION; ++row) {
            for (uint32_t col = 0; col < MATRIX_DIMENSION; ++col) {
                k.set(baked.component(row * MATRIX_DIMENSION + col),
                      a.view_projection.at(row, col));
            }
        }
        return baked;
    }();

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
        const Reg<Scalar> out_addr = k.mul(
            id, k.constant(static_cast<float>(screen_vertex_bytes(a.varying_count))));

        const float screen_base = static_cast<float>(a.screen_offset);
        k.store(out_addr, sx, screen_base + 0.0f);
        k.store(out_addr, sy, screen_base + 4.0f);
        k.store(out_addr, sz, screen_base + 8.0f);

        // inv_w falls out of the divide above at no extra cost, and pass 2
        // cannot recover it once w is gone.
        k.store(out_addr, inv_w, screen_base + 12.0f);

        // The varyings, copied through. A vertex's attributes sit in a buffer of
        // their own and land in the slots after the four above, so that pass 2
        // reads a vertex once and has everything about it.
        if (a.varying_count > 0) {
            const Reg<Scalar> from = k.add(
                k.constant(static_cast<float>(a.attribute_offset)),
                k.mul(id,
                      k.constant(static_cast<float>(a.varying_count) * sizeof(float))));
            for (uint32_t i = 0; i < a.varying_count; ++i) {
                const float at = static_cast<float>(i * sizeof(float));
                k.store(out_addr, k.load(from, at),
                        screen_base + static_cast<float>(SCREEN_VERTEX_BYTES) + at);
            }
        }
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
    LaunchConfig config{grid, block};
    config.const_offset = args.uniform_offset;
    rt.myrt_launch(build_vertex_program, config, raw);
}
