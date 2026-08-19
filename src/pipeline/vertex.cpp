#include <array>
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
    // Which instance this block is drawing. Driven by whether matrices were
    // supplied rather than by how many: a draw of one instance still has a
    // transform to apply, and a launch with no instance buffer is the
    // uninstanced kernel that every figure before this was taken from.
    const bool instanced = a.instance_offset != 0;
    const Reg<Scalar> instance = k.block_y();

    const auto view_projection = [&] {
        // Instanced, the window is spent on the matrices that vary and the
        // view-projection is baked. Reading it from the window here would read
        // the first instance's model matrix instead, since that is what the
        // window points at.
        if (a.uniform_offset != 0 && !instanced) {
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
    };

    // Where a block's matrix sits: blockIdx.y scaled by a matrix, which is
    // uniform across the warp — see instance_offset in the header for why that
    // is the whole reason a per-instance matrix can go in the window at all.
    const auto instance_matrix = [&](size_t base) {
        return k.load_const_mat4(
            k.add(k.const_base(), k.mul(instance, k.constant(static_cast<float>(
                                                      MAT4_REGISTERS * sizeof(float))))),
            static_cast<float>(base) - static_cast<float>(a.uniform_offset));
    };

    // Composed, one matrix carries both and the kernel is the uninstanced one
    // with a different address. Per-vertex, the model matrix is applied first
    // and the view-projection after, which is two MATVECs a vertex.
    const bool composed = instanced && a.transform == InstanceTransform::ComposePass;
    const Reg<Mat4> mvp = composed    ? instance_matrix(a.composed_offset)
                          : instanced ? instance_matrix(a.instance_offset)
                                      : view_projection();

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

        // The varyings, loaded before the shader rather than after the matrices,
        // so that a shader can read them and write over them. A launch with no
        // shader stores them straight through and nothing has moved.
        std::array<Reg<Scalar>, MAX_VARYINGS> attributes{};
        const Reg<Scalar> attribute_addr =
            a.varying_count == 0
                ? id
                : k.add(k.constant(static_cast<float>(a.attribute_offset)),
                        k.mul(id, k.constant(static_cast<float>(a.varying_count) *
                                             sizeof(float))));
        for (uint32_t i = 0; i < a.varying_count; ++i) {
            attributes[i] = k.load(attribute_addr, static_cast<float>(i * sizeof(float)));
        }

        // The shader, if there is one. It writes where the vertex actually is,
        // and the matrices below act on that instead of on what was loaded.
        if (a.shade) {
            Vertex vertex;
            vertex.out = k.vec3();
            vertex.position = k.vec3();
            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(vertex.position.component(c), position.component(c));
                k.copy_into(vertex.out.component(c), position.component(c));
            }
            vertex.index = id;
            vertex.instance = instance;
            vertex.varyings = attributes;
            vertex.varying_count = a.varying_count;
            a.shade(k, vertex);

            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(position.component(c), vertex.out.component(c));
            }
            attributes = vertex.varyings;
        }

        // Instanced and not composed, mvp holds the model matrix alone and the
        // view-projection follows it. This is the extra MATVEC a vertex that a
        // composition pass exists to remove.
        const Reg<Vec4> model_space = k.transform(mvp, position);
        const Reg<Vec4> clip = (instanced && !composed)
                                   ? k.transform(view_projection(), model_space)
                                   : model_space;

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
        //
        // Instance-major: instance i's vertices are one run of vertex_count, so
        // pass 2 sees a single longer list at the stride it already used and
        // needs no change at all.
        const Reg<Scalar> slot =
            instanced ? k.add(id, k.mul(instance,
                                        k.constant(static_cast<float>(a.vertex_count))))
                      : id;
        const Reg<Scalar> out_addr = k.mul(
            slot, k.constant(static_cast<float>(screen_vertex_bytes(a.varying_count))));

        const float screen_base = static_cast<float>(a.screen_offset);
        k.store(out_addr, sx, screen_base + 0.0f);
        k.store(out_addr, sy, screen_base + 4.0f);
        k.store(out_addr, sz, screen_base + 8.0f);

        // inv_w falls out of the divide above at no extra cost, and pass 2
        // cannot recover it once w is gone.
        k.store(out_addr, inv_w, screen_base + 12.0f);

        // The varyings, into the slots after the four above, so that pass 2 reads
        // a vertex once and has everything about it. Loaded before the shader
        // and stored after it: what lands here is whatever the shader left.
        for (uint32_t i = 0; i < a.varying_count; ++i) {
            const float at = static_cast<float>(i * sizeof(float));
            k.store(out_addr, attributes[i],
                    screen_base + static_cast<float>(SCREEN_VERTEX_BYTES) + at);
        }
    });

    return k.build();
}

Program build_compose_program(void** args)
{
    const VertexStageArgs& a = *static_cast<const VertexStageArgs*>(args[0]);
    IRBuilder k;

    // One thread an instance, and the view-projection baked: this pass runs once
    // a frame over a handful of threads, so sixteen moves here are cheaper than
    // a window read and they leave the window free for the model matrices.
    const Reg<Mat4> vp = k.mat4();
    for (uint32_t row = 0; row < MATRIX_DIMENSION; ++row) {
        for (uint32_t col = 0; col < MATRIX_DIMENSION; ++col) {
            k.set(vp.component(row * MATRIX_DIMENSION + col),
                  a.view_projection.at(row, col));
        }
    }

    const Reg<Scalar> id = k.thread_x();
    const Reg<Scalar> live = k.lt(id, k.constant(static_cast<float>(a.instance_count)));

    k.if_(live, [&] {
        // Read and written through ordinary global memory, not the window: a
        // lane here wants its own matrix, so the address varies by lane and the
        // window's pricing would be a lie. The window is for pass 1, where a
        // block is one instance.
        const Reg<Scalar> addr =
            k.mul(id, k.constant(static_cast<float>(MAT4_REGISTERS * sizeof(float))));

        const Reg<Mat4> model = k.mat4();
        for (uint32_t i = 0; i < MAT4_REGISTERS; ++i) {
            k.load_into(model.component(i), addr,
                        static_cast<float>(a.instance_offset + i * sizeof(float)));
        }

        // view_projection * model, in that order: the model matrix acts first.
        const Reg<Mat4> folded = k.compose(vp, model);
        for (uint32_t i = 0; i < MAT4_REGISTERS; ++i) {
            k.store(addr, folded.component(i),
                    static_cast<float>(a.composed_offset + i * sizeof(float)));
        }
    });

    return k.build();
}

void run_compose_stage(MyGPURuntime& rt, const VertexStageArgs& args)
{
    if (args.instance_count == 0) {
        throw std::runtime_error("run_compose_stage: no instances to compose for");
    }
    if (args.composed_offset == 0) {
        throw std::runtime_error(
            "run_compose_stage: nowhere to put the folded matrices — set "
            "composed_offset");
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.instance_count + WARP_SIZE - 1) / WARP_SIZE, 1, 1};
    void* raw = const_cast<VertexStageArgs*>(&args);
    void* kernel_args[] = {raw};
    rt.myrt_launch(build_compose_program, grid, block, kernel_args);
}

void run_vertex_stage_indirect(MyGPURuntime& rt, const VertexStageArgs& args,
                               size_t grid_offset, StreamId stream)
{
    if (args.vertex_count == 0) {
        throw std::runtime_error("run_vertex_stage_indirect: a mesh with no vertices");
    }
    if (grid_offset == 0) {
        throw std::runtime_error(
            "run_vertex_stage_indirect: no grid to read — a culling pass writes one");
    }
    if (args.instance_offset == 0) {
        throw std::runtime_error(
            "run_vertex_stage_indirect: an indirect grid decides how many instances "
            "to draw, and there are no matrices to draw them from");
    }

    void* raw[] = {const_cast<VertexStageArgs*>(&args)};
    IndirectLaunchConfig config;
    config.grid_offset = grid_offset;
    config.block = dim3{WARP_SIZE, 1, 1};
    config.const_offset = args.uniform_offset;
    rt.myrt_launch_indirect(build_vertex_program, config, raw, stream);
}

void run_vertex_stage(MyGPURuntime& rt, const VertexStageArgs& args)
{
    // Caught here rather than left to myrt_launch, which would reject the empty
    // grid with a message about launch geometry and say nothing about vertices.
    if (args.vertex_count == 0) {
        throw std::runtime_error("run_vertex_stage: a mesh with no vertices");
    }
    if (args.instance_count == 0) {
        throw std::runtime_error("run_vertex_stage: a draw of no instances");
    }
    if (args.instance_count > 1 && args.instance_offset == 0) {
        throw std::runtime_error(
            "run_vertex_stage: " + std::to_string(args.instance_count) +
            " instances and no matrices for them — set instance_offset");
    }
    if (args.instance_offset != 0 && args.transform == InstanceTransform::ComposePass &&
        args.composed_offset == 0) {
        throw std::runtime_error(
            "run_vertex_stage: a composed transform and nowhere it was composed "
            "into — run_compose_stage writes composed_offset");
    }

    // 1D, one warp wide: a vertex has no second coordinate to spend, unlike a
    // pixel. The last block is partly out of range whenever vertex_count is not
    // a multiple of the warp, which is what the guard in the kernel is for.
    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.vertex_count + WARP_SIZE - 1) / WARP_SIZE, args.instance_count,
                    1};

    // myrt_launch takes void** after CUDA's convention, so the const has to
    // come off. The kernel only reads it, and args outlives the call.
    void* raw[] = {const_cast<VertexStageArgs*>(&args)};
    LaunchConfig config{grid, block};
    config.const_offset = args.uniform_offset;
    rt.myrt_launch(build_vertex_program, config, raw);
}
