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

    // Where the index buffer sits, when indexed is set. Three entries a
    // triangle, each a vertex slot in the screen buffer.
    //
    // Stored as floats: the ISA has no integer registers, and a float carries
    // whole numbers exactly to 2^24, which is more vertices than a scene here
    // will hold.
    size_t index_offset = 0;

    // The two axes pass 2 varies along, both read on the host when the kernel
    // is built. A KernelFunc runs once per launch, so choosing here costs no
    // lane anything and only the chosen form reaches the instruction stream.
    //
    // They are flags rather than four separate builders because the axes are
    // independent and each touches one block: indexed changes how a triangle's
    // vertices are addressed, predicated changes what happens once coverage is
    // known. Copies would have to be kept in step by hand, and a fix landing in
    // three of four would leave the measurements quietly disagreeing — which is
    // the failure separate copies were meant to prevent.

    // Vertices come from the index buffer rather than three consecutive slots.
    // Pass 1 then transforms each unique vertex once; this pass pays three
    // dependent loads a triangle for it.
    bool indexed = false;

    // Coverage is blended rather than branched on: no lane is masked, and every
    // lane pays for a shade it may discard. emit_keep holds both forms and what
    // the trade measured out at.
    bool predicated = false;
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
