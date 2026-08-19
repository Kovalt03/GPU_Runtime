#pragma once

#include <cstdint>
#include <vector>

#include "bvh.hpp"
#include "isa.hpp"
#include "math3d.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"

// --- deciding on the device what to draw --------------------------------------
//
// Everything before this had the host say how much work a launch does. A draw of
// sixty-four instances launched sixty-four instances' worth whether or not the
// camera could see them, and culling them first meant reading their boxes back,
// testing them on the host and uploading what survived — which is a
// synchronisation in the middle of a frame.
//
// The pieces to avoid that were already here and had nothing to point at:
// myrt_launch_indirect reads a grid from device memory when the launch reaches
// the machine, and V_ATOM_ADD_GLOBAL_F32 gives each surviving lane a slot
// nobody else got. What was missing was something to cull, which instancing
// supplied.

// The six planes of the view frustum, each as (normal, -distance) so that a
// point is outside when dot(plane.xyz, p) + plane.w < 0.
//
// Extracted from the view-projection by the Gribb-Hartmann identity: a row of
// the matrix added to or subtracted from the last row is a plane, because a clip
// coordinate is inside when -w <= x <= w and each of those six inequalities is
// one combination. No geometry is involved, which is why it belongs here rather
// than beside the camera.
struct Frustum {
    Float4 plane[6];
};

Frustum frustum_of(const Float4x4& view_projection);

// Whether a box is entirely outside one of the planes, tested on the host.
//
// The same arithmetic the kernel runs, and here for the same reason
// nodes_visited is: a figure that says "the device culled 40 of 64" is worth
// nothing unless something independent agrees which 40.
bool outside_frustum(const Frustum& frustum, const Box& box);

struct CullStageArgs {
    Frustum frustum;

    // Byte offsets into device memory. `boxes` is six floats an instance, in the
    // order the matrices were uploaded; `matrices` is sixteen.
    size_t boxes_offset = 0;
    size_t matrices_offset = 0;

    // Where the survivors' matrices go, compacted. Sixteen floats each, and the
    // first `grid.y` of them are the ones to draw.
    size_t survivors_offset = 0;

    // Three floats an indirect launch reads its grid from. The host writes
    // {blocks_x, 0, 1} before the cull; this pass raises the middle one with an
    // atomic, so the counter and the grid are the same three words and there is
    // no second kernel to publish one into the other.
    //
    // That is how a draw-indirect argument buffer is laid out on real hardware,
    // and for the same reason.
    size_t grid_offset = 0;

    uint32_t instance_count = 0;
};

// Builds the cull. One thread an instance.
Program build_cull_program(void** args);

// Runs it. Enqueued rather than run to completion when a stream is given, which
// is the whole point: the launch after it reads a grid this one has not written
// yet, and neither the host nor the queue has to know the number.
void run_cull_stage(MyGPURuntime& rt, const CullStageArgs& args,
                    StreamId stream = DEFAULT_STREAM);
