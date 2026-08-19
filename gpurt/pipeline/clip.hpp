#pragma once

#include <cstdint>

#include "isa.hpp"
#include "math3d.hpp"
#include "runtime.hpp"

// --- pass 0 -----------------------------------------------------------------
// Near-plane clipping, one thread a triangle, before anything is projected.
//
// A triangle that crosses the near plane cannot be projected: w passes through
// zero and the screen coordinates run off to infinity. Every scene here has kept
// the camera outside the geometry, so it has never happened — and the moment a
// camera moves through a wall it happens everywhere at once.
//
// Clipped in **world space** rather than clip space, which is not what hardware
// does and is what suits this pipeline: the output is world triangles, so pass 1,
// the lighting and the ray tracer all read it without knowing a clip happened.
// Clip space would mean handing pass 1 something already projected and giving it
// a second form to accept.
//
// The surviving triangles are **compacted**: each thread takes its slots with an
// atomic add and writes there, so the output has no holes and the passes after it
// read a count rather than a mask. That count is a device value the host never
// sees — the same shape GPU-driven culling has, and the first use of the value
// V_ATOM_ADD_GLOBAL_F32 hands back.

struct ClipStageArgs {
    // Byte offsets from the base of device memory.
    size_t world_offset = 0;    // the triangles to clip, three vertices each
    size_t output_offset = 0;   // where the survivors go, same layout
    size_t counter_offset = 0;  // one float: how many triangles came out

    uint32_t triangle_count = 0;

    // The plane, as dot(normal, p) + d >= 0 for a point in front of it. Read from
    // the constant window rather than baked, so that one program serves any
    // camera — which is what [u] was for.
    size_t plane_offset = 0;
};

// A triangle crossing the plane becomes one or two; one behind it becomes none.
// Two is the worst case, and what an output buffer has to be sized for.
inline constexpr uint32_t CLIP_WORST_CASE = 2;

// The plane a camera's near clip sits on, in world space.
Float4 near_plane_of(const Camera& camera);

// Builds pass 0. args[0] must point at a ClipStageArgs that outlives the launch.
Program build_clip_program(void** args);

// Runs it over one thread a triangle. The counter must be zero before the launch
// — the atomic adds to it, and a stale value would place the survivors past the
// end of the buffer.
void run_clip_stage(MyGPURuntime& rt, const ClipStageArgs& args);
