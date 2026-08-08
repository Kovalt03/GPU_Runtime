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

// --- pass 2 -----------------------------------------------------------------
// Coverage, one thread per pixel. This is the first kernel whose threads
// disagree about anything that matters: lanes covering a triangle take a
// different path from lanes outside it, and a warp spanning an edge pays for
// both. That is the divergence this project exists to measure.

// RGB per pixel, the layout kernels/ray_triangle.cpp already writes, so the two
// renderers can be compared image against image rather than by description.
inline constexpr uint32_t PIXEL_FLOATS = 3;
inline constexpr uint32_t PIXEL_BYTES = PIXEL_FLOATS * sizeof(float);

struct RasterStageArgs {
    // Byte offsets from the base of device memory. screen_offset is where pass
    // 1 left its output; nothing is transferred between the two passes.
    size_t screen_offset = 0;
    size_t framebuffer_offset = 0;

    uint32_t width = 0;
    uint32_t height = 0;

    // Which triangle in the screen buffer to draw. One for now — a loop over
    // several needs a depth comparison to decide which one wins, and that is a
    // separate change.
    uint32_t triangle_index = 0;
};

// Builds pass 2. args[0] must point at a RasterStageArgs that outlives the
// launch.
//
// No matrix here, which is the other half of why the pipeline is split: the
// transform is per vertex and coverage is per pixel, so this kernel never pays
// the sixteen registers pass 1 spends on a uniform.
Program build_raster_program(void** args);

// Runs pass 2 over width x height threads. The launch is 2D, and 32 threads
// wide along x so that one warp covers 32 horizontally adjacent pixels — which
// is what makes a triangle edge split a warp rather than fall between them.
void run_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args);

// Twice the signed area of the triangle (a, b, p): the z component of a 2D
// cross product. Its sign says which side of the line ab the point falls on,
// and the three of them together decide coverage.
float edge_function(Float3 a, Float3 b, float px, float py);

// Barycentric weights of (px, py), in vertex order.
//
// Normalised by the signed area, which is what lets coverage be "all three are
// >= 0" whatever way the triangle winds. That is not a nicety here: the y flip
// in pass 1 reverses the winding of every triangle, so a test written against
// one order would draw nothing at all.
Float3 barycentric(Float3 v0, Float3 v1, Float3 v2, float px, float py);

// The colour pass 2 writes for one pixel, and the reference the kernel is
// compared against. Samples the pixel centre, matching the ray tracer — an edge
// landing on a corner otherwise leaves neighbouring pixels disagreeing and
// frays the edge.
//
// Ordered (w1, w2, w0) so each vertex comes out a pure primary in the same
// arrangement kernels/ray_triangle.cpp produces. Black where the triangle does
// not cover the pixel.
Float3 shade_pixel(Float3 v0, Float3 v1, Float3 v2, uint32_t px, uint32_t py);
