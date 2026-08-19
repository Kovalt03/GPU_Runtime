#pragma once

#include <cstdint>

#include "isa.hpp"
#include "math3d.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"

// --- pass 1 -----------------------------------------------------------------

// Passed through myrt_launch's args as a single pointer, the way
// apps/ray_triangle.cpp passes its scene.
struct VertexStageArgs {
    Float4x4 view_projection;

    // Byte offsets from the base of device memory, from myrt_device_offset.
    // Not pointers: the ISA has no way to dereference a host address.
    size_t world_offset = 0;
    size_t screen_offset = 0;

    uint32_t vertex_count = 0;
    uint32_t width = 0;
    uint32_t height = 0;

    // Per-vertex attributes that ride through to the fragment stage, and how
    // many floats of them a vertex has.
    //
    // Zero is what every launch before them declared, and the screen vertex is
    // then exactly the four floats it always was. Pass 1 copies them rather than
    // computing them: a stage that computed its own would be a vertex shader,
    // which is a callback this does not have yet.
    size_t attribute_offset = 0;
    uint32_t varying_count = 0;

    // Emits the instructions that place a vertex, when there are any.
    //
    // Runs before the matrices, on the position as the buffer holds it, and
    // writes where the vertex actually is. Empty is what every launch before
    // this declared and the kernel it builds is the one that existed then — a
    // load, the matrices, a store.
    VertexFn shade;

    // How large the screen buffer is, when a caller knows. Zero means unchecked,
    // which is what every launch declaring no varyings can afford — the buffer
    // and the vertex are the same four floats then, and there is nothing to get
    // wrong.
    size_t screen_bytes = 0;

    // Where the view-projection is, if it is not baked into the program.
    //
    // Zero means bake it: sixteen moves, and a program that serves one matrix.
    // Anything else is a byte offset into device memory, and the kernel reads it
    // through the constant window — one instruction, and a program that serves
    // any matrix. That is the difference between a demo and a runtime: without
    // it, an object with its own transform needs its own build.
    size_t uniform_offset = 0;

    // --- instancing ---------------------------------------------------------
    // One model matrix an instance, sixteen floats each, in the constant window
    // beside the view-projection.
    //
    // The launch grows a second grid dimension rather than a longer first one:
    // blockIdx.y is the instance, so **a block is one instance**, and the matrix
    // a lane wants is the same across its warp. That is what lets it ride the
    // constant window at all — the window is charged once for a warp, and the
    // justification is warp-uniformity rather than the address being a launch
    // constant. A block coordinate preserves it; a lane coordinate would not.
    //
    // One instance is what every launch before this declared, and the kernel it
    // builds is the one that existed then.
    uint32_t instance_count = 1;
    size_t instance_offset = 0;

    // Where the model matrix is folded into the view-projection.
    InstanceTransform transform = InstanceTransform::PerVertex;

    // Where the composition pass put its results, when there was one. Read
    // instead of instance_offset, and the kernel then does one MATVEC where the
    // per-vertex form does two.
    size_t composed_offset = 0;
};

// Composes view_projection with every model matrix — one thread an instance,
// one V_MATMUL_MAT4_F32 each.
//
// A launch of its own rather than work folded into pass 1, and that is the whole
// point: composing costs four MATVECs, so doing it once an instance beats doing
// an extra MATVEC at every vertex only when an instance has more than four of
// them. A pass makes the "once" real.
Program build_compose_program(void** args);
void run_compose_stage(MyGPURuntime& rt, const VertexStageArgs& args);

// Builds pass 1. Matches KernelFunc, so it is handed to myrt_launch directly;
// args[0] must point at a VertexStageArgs that outlives the launch.
//
// The matrix arrives as 16 V_MOV_F32, one per element — a uniform being baked
// into the program. Because a KernelFunc runs once per launch rather than once
// per thread, this is a recompile per frame, which costs nothing here and is
// what lets the camera change without any uniform-upload path existing.
Program build_vertex_program(void** args);

// Runs pass 1 over vertex_count threads. The launch is 1D: a vertex has no
// meaningful second coordinate, unlike a pixel.
//
// Threads past vertex_count exit without writing, since a grid only divides
// evenly into warps by accident.
void run_vertex_stage(MyGPURuntime& rt, const VertexStageArgs& args);

// The same, with the instance count read from device memory when the launch
// reaches the machine rather than when it is queued.
//
// `grid_offset` names three floats a culling pass has already raised, so the
// host never learns how many instances survived. It is enqueued rather than run,
// since waiting for it would put the number back in the host's hands — the whole
// arrangement is that nobody outside the device knows it.
//
// instance_count is still read: it sizes the constant window and is the upper
// bound the buffers were allocated for. What comes from the device is how many
// of them are drawn.
void run_vertex_stage_indirect(MyGPURuntime& rt, const VertexStageArgs& args,
                               size_t grid_offset, StreamId stream = DEFAULT_STREAM);
