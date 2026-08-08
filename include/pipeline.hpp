#pragma once

#include <cstdint>

#include "isa.hpp"
#include "math3d.hpp"
#include "runtime.hpp"

// The graphics layer. Everything below is a thin wrapper over myrt_launch: the
// runtime stays general-purpose the way CUDA is, and the knowledge that these
// numbers are triangles lives only here.
//
// Rasterisation runs in two passes, because the two halves are indexed
// differently. The MVP transform is per vertex and coverage is per pixel, and a
// single kernel would have to pick one and waste threads on the other. Real
// hardware splits them for the same reason, into a vertex stage and a fragment
// stage, and this runtime can already launch twice.
//
//   pass 1  vertex_program   1 thread = 1 vertex   world -> screen
//   pass 2  raster_program   1 thread = 1 pixel    coverage, nearest wins
//
// Only pass 1 exists so far.

// --- device buffer layout ---------------------------------------------------
// Kernels address device memory by byte offset, so the strides are named once
// here rather than spelled out at every load.

// x, y, z in world space.
inline constexpr uint32_t WORLD_VERTEX_FLOATS = 3;
inline constexpr uint32_t WORLD_VERTEX_BYTES = WORLD_VERTEX_FLOATS * sizeof(float);

// x, y in pixels, plus the NDC depth that pass 2 compares. Not four: the
// perspective divide happens in pass 1, so w has served its purpose by the time
// anything is written.
inline constexpr uint32_t SCREEN_VERTEX_FLOATS = 3;
inline constexpr uint32_t SCREEN_VERTEX_BYTES = SCREEN_VERTEX_FLOATS * sizeof(float);

// --- pass 1 -----------------------------------------------------------------

// Passed through myrt_launch's args as a single pointer, the way
// kernels/ray_triangle.cpp passes its scene.
struct VertexStageArgs {
    Float4x4 view_projection;

    // Byte offsets from the base of device memory, from myrt_device_offset.
    // Not pointers: the ISA has no way to dereference a host address.
    size_t world_offset = 0;
    size_t screen_offset = 0;

    uint32_t vertex_count = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

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

// The same projection on the host, and the reference pass 1 is checked against.
// Lives here rather than in the test because pass 2 will want it too, to decide
// what a whole frame should have looked like.
//
//   clip   = view_projection * (world, 1)
//   ndc    = clip.xyz / clip.w
//   screen = ((ndc.x + 1) / 2 * width, (1 - ndc.y) / 2 * height, ndc.z)
//
// y is flipped because NDC counts upward from the bottom and an image counts
// downward from the top row.
Float3 project_vertex(const Float4x4& view_projection, Float3 world, uint32_t width,
                      uint32_t height);
