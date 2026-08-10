#pragma once

#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"
#include "thread.hpp"  // SHARED_MEM_FLOATS, the staging budget

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
