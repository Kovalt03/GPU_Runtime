#pragma once

#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "math3d.hpp"
#include "runtime.hpp"
#include "thread.hpp"  // SHARED_MEM_FLOATS, the staging budget

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

// --- device buffer layout ---------------------------------------------------
// Kernels address device memory by byte offset, so the strides are named once
// here rather than spelled out at every load.

// x, y, z in world space.
inline constexpr uint32_t WORLD_VERTEX_FLOATS = 3;
inline constexpr uint32_t WORLD_VERTEX_BYTES = WORLD_VERTEX_FLOATS * sizeof(float);

// x, y in pixels, the NDC depth pass 2 compares, and 1/w.
//
// w is kept because the divide is exactly what makes screen-space interpolation
// wrong for anything but depth. Barycentric weights taken from projected
// vertices are affine; an attribute carried across a perspective-projected
// triangle needs them weighted by 1/w and renormalised. Depth does not, being
// linear in screen space by construction — which is the whole reason a depth
// buffer stores NDC z.
//
// Costs a float per vertex and a reciprocal per pixel, and without it the
// rasteriser and the ray tracer only agree on a scene whose triangles all sit
// at one depth.
inline constexpr uint32_t SCREEN_VERTEX_FLOATS = 4;
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

// A triangle as pass 1 leaves it: x and y in pixels, z the NDC depth.
struct ScreenTriangle {
    Float3 v0;
    Float3 v1;
    Float3 v2;

    // One per vertex, in that order. Named rather than packed into a Float3,
    // which would read as a vector and is three unrelated scalars.
    float inv_w0 = 1.0f;
    float inv_w1 = 1.0f;
    float inv_w2 = 1.0f;
};

// --- pass 2 -----------------------------------------------------------------
// Coverage, one thread per pixel, and the first kernel whose lanes disagree
// about anything that matters: a warp spanning a triangle edge pays for both
// paths. That is the divergence this project exists to measure.
//
// Each pixel walks every triangle, which costs O(pixels x triangles) against
// the O(fragments) real hardware pays — it bins triangles into tiles first, so
// a pixel only sees the few that reach it. Kept this way on purpose, as the
// baseline a tiled version is measured against; benchmarks/RESULTS.md has the
// numbers.

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

    // How many triangles the screen buffer holds. Each thread walks all of
    // them and keeps the nearest that covers it.
    uint32_t triangle_count = 0;
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

// --- tiling -----------------------------------------------------------------
// The fix for the walk above: sort triangles into screen tiles once, so a pixel
// only ever visits the few that reach it. O(pixels x triangles) becomes
// O(pixels x triangles per tile), which is what real hardware buys with a
// binning stage.
//
// One ThreadBlock covers one tile, so blockIdx *is* the tile index and each
// block can find its own list. That is what block_x/block_y were added for.
//
// The binning runs on the host. Real hardware does it in a geometry stage and
// keeps the lists on chip; doing it here changes where the work happens but not
// what it saves, and the saving is what is being measured.

// 32 wide so a warp still covers 32 adjacent pixels of one row — the
// arrangement every divergence figure so far was measured against, so the
// comparison stays honest.
//
// Height is free to be more than one because nothing here uses shared memory:
// each thread reads its tile's triangles from global on its own, and no two
// warps have to agree on anything. Hoisting a tile into shared memory would
// need a barrier, which the ISA does not have, so it is a separate step.
inline constexpr uint32_t TILE_WIDTH = 32;
inline constexpr uint32_t TILE_HEIGHT = 8;

// What the host hands the device.
//
// Triangles are copied into each tile's run rather than referenced by index. An
// index would cost a second, dependent global load per triangle — 100 units to
// save 36 bytes — and the duplication is bounded by how many tiles a triangle
// spans.
struct TileBinning {
    // One screen triangle per entry, tile by tile, in tile order — laid out
    // exactly as pass 1 writes three consecutive vertices, so a kernel reads a
    // binned triangle with the offsets it already has.
    std::vector<float> vertices;

    // Two floats per tile: where its run starts, counted in triangles, and how
    // many it holds. Floats because the ISA has no integer registers.
    std::vector<float> table;

    uint32_t tiles_x = 0;
    uint32_t tiles_y = 0;

    uint32_t tile_count() const
    {
        return tiles_x * tiles_y;
    }
};

// Assigns each triangle to every tile its bounding box overlaps.
//
// A bounding box over-counts — a thin diagonal claims tiles it only passes
// near — which costs a few wasted coverage tests and never a missing pixel. An
// exact test would be a triangle/rectangle intersection per pair, and the tiles
// it saves are the cheapest ones to have kept.
//
// A tile's run is TILE_TRIANGLE_FLOATS per entry.
TileBinning bin_triangles(const std::vector<ScreenTriangle>& triangles, uint32_t width,
                          uint32_t height);

struct TiledRasterStageArgs {
    // Byte offsets from the base of device memory.
    size_t tile_vertices_offset = 0;
    size_t tile_table_offset = 0;
    size_t framebuffer_offset = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tiles_x = 0;

    // The fullest tile in the binning. Only the shared-memory variant needs
    // it, to refuse a tile it cannot hold; the global-memory one ignores it.
    uint32_t max_tile_triangles = 0;
};

// Builds the tiled pass 2. Same picture as build_raster_program, reached by
// reading a shorter list.
Program build_tiled_raster_program(void** args);

// Runs it, one block per tile. The grid is the tile grid, which is what makes
// blockIdx the tile index.
void run_tiled_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args);

// --- tiling, through shared memory ------------------------------------------
// The same frame again. What changes is where the tile's triangles are read
// from: every pixel of a block walks the same list, so reading it from global
// once per pixel is 32 lanes issuing the same load. The block loads it once
// into shared memory instead, at 8 units a load rather than 100.
//
// That is what BARRIER was added for. The threads that fill shared memory are
// not the ones that read each entry, so without a rendezvous between the two a
// fast warp reads a slot a slow one has not written.

// 4096 floats of shared memory, nine per triangle. A tile holding more than
// this cannot be staged in one pass, and real hardware has the same problem —
// it splits the tile across passes. Refused here instead.
// A binned triangle is laid out exactly as three consecutive screen vertices,
// so all three raster kernels read it with one set of offsets.
inline constexpr uint32_t TILE_TRIANGLE_FLOATS = 3 * SCREEN_VERTEX_FLOATS;

inline constexpr uint32_t SHARED_TRIANGLE_CAPACITY =
    SHARED_MEM_FLOATS / TILE_TRIANGLE_FLOATS;

// Builds the shared-memory pass 2. args[0] must point at a
// TiledRasterStageArgs, whose max_tile_triangles must not exceed
// SHARED_TRIANGLE_CAPACITY.
//
// The fill and the barrier sit *outside* the bounds check, unlike everything
// else in this file. Every thread of the block has to reach a barrier, and the
// edge blocks of a frame hold threads whose pixel is off screen — guarding the
// barrier along with the pixel work would have those threads branch past it
// and the scheduler would refuse the launch.
Program build_shared_raster_program(void** args);

// Runs it, one block per tile, as run_tiled_raster_stage does. Throws when a
// tile holds more triangles than shared memory can stage.
void run_shared_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args);

// --- ray tracing ------------------------------------------------------------
// The same scene by the other route, and the comparison the project was built
// for: two renderers over one image, diverging for different reasons.
//
// One launch, not two. Rays are cast in world space and the triangles are
// already there, so nothing has to be projected first — the vertex stage has no
// counterpart here. That is the structural difference between the two paths,
// and the reason the ray tracer needs no matrix at all.

// Three vertices in world space. A separate type from ScreenTriangle for the
// same reason Float3 is separate from Reg<Vec3>: the two hold identical numbers
// and mixing them is the mistake worth making impossible.
struct WorldTriangle {
    Float3 v0;
    Float3 v1;
    Float3 v2;
};

// The camera as the kernel wants it. right and up arrive already scaled by the
// field of view and the aspect, so a ray is
//
//   direction = right * sx + up * sy + forward
//
// for sx and sy in [-1, 1] — five instructions and no matrix. A matrix would
// cost about the same to apply and sixteen moves to set up, and would make the
// one path that needs no transform carry one anyway.
struct RayBasis {
    Float3 origin;
    Float3 right;
    Float3 up;
    Float3 forward;
};

RayBasis ray_basis(const Camera& camera, float aspect);

// The tolerance both intersection tests use, the host one and the kernel. One
// symbol rather than a literal in each, because the two are compared pixel for
// pixel and an edge lands on a different side the moment they drift apart.
//
// Not an argument: it is a property of the test, not of a scene, and passing it
// in would just move the chance of two callers disagreeing one level up.
//
// It does two jobs. Against the determinant it rejects a ray parallel to the
// triangle — and, being a signed comparison rather than an absolute one, a back
// face along with it. Against t it keeps a hit behind the origin from counting.
inline constexpr float INTERSECT_EPSILON = 1e-6f;

// How a hit is coloured.
//
// Barycentric is the debug colouring both renderers share, and the only one the
// rasteriser can produce: pass 1 keeps screen x, y and depth, so the world
// position and normal a light needs are gone by the time it runs. Lighting is
// therefore the point at which the two paths stop being directly comparable,
// which is why it is a mode rather than a replacement.
enum class ShadingMode {
    Barycentric,
    Diffuse,
};

struct Shading {
    ShadingMode mode = ShadingMode::Barycentric;

    // World space. Only read in Diffuse.
    Float3 light_position{2.0f, 3.0f, 1.0f};
    Float3 base_colour{1.0f, 1.0f, 1.0f};
};

struct RaytraceStageArgs {
    RayBasis basis;
    Shading shading;

    // Byte offsets from the base of device memory. Three floats a vertex, in
    // world space and never rewritten — no 1/w, there being no projection on
    // this path to produce one.
    size_t triangles_offset = 0;
    size_t framebuffer_offset = 0;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t triangle_count = 0;
};

// Builds the ray tracer. args[0] must point at a RaytraceStageArgs.
//
// Möller-Trumbore leaves for the next triangle from four separate tests, which
// is what Label exists for — nesting if_ four deep would pay for the inverted
// condition each time and read as a staircase.
Program build_raytrace_program(void** args);

// Runs it, one thread per pixel. The launch geometry matches the untiled
// rasteriser's exactly, since the divergence figures are meant to be compared
// and a different warp shape would compare two things at once.
void run_raytrace_stage(MyGPURuntime& rt, const RaytraceStageArgs& args);
