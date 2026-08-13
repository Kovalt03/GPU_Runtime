#pragma once

#include <cstdint>

#include "isa.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"

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

    // Where the index buffer sits, for the indexed variant below. Three entries
    // a triangle, each a vertex slot in the screen buffer.
    //
    // Stored as floats: the ISA has no integer registers, and a float carries
    // whole numbers exactly to 2^24, which is more vertices than a scene here
    // will hold.
    size_t index_offset = 0;
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

// The same pass 2, reading an index buffer rather than three consecutive
// vertices per triangle.
//
// This is the half of indexing that costs something. Pass 1 saves a transform
// for every triangle that shares a corner; here each triangle pays three
// dependent loads — the index first, then the vertex it names — where the
// flattened walk knew the address already.
//
// The two exist side by side because that trade is the measurement. Neither is
// the successor of the other.
Program build_indexed_raster_program(void** args);

void run_indexed_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args);

// The same pass 2 with no branch in it — the second of the two ways a SIMT
// machine can handle lanes that disagree.
//
// build_raster_program decides coverage and then jumps, so a warp split across
// a triangle's edge issues both sides and masks half its lanes on each. This
// kernel shades for every lane and keeps the result arithmetically:
//
//   kept = covered * new + (1 - covered) * old
//
// covered is 1.0 or 0.0, so one term survives and the other is multiplied away.
// Every lane runs the same instructions, and divergence measures exactly zero.
//
// What that costs is a shade for lanes the triangle never covered. On the
// scenes in render_bench the branch wins by about 2% throughout, because it
// skips the shade outright whenever no lane of a warp is covered — which is
// most warps, for triangles that do not fill the frame.
//
// Predication pays where warps straddle edges often, and the route it is
// attached to straddles them least: the walk runs 1.3% diverged where the tiled
// route runs 7.4%. Only the flattened walk has a predicated form today, so that
// comparison is the one available, not the one that settles the question.
Program build_predicated_raster_program(void** args);

void run_predicated_raster_stage(MyGPURuntime& rt, const RasterStageArgs& args);
