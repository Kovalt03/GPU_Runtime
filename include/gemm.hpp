#pragma once

#include <cstdint>
#include <vector>

#include "isa.hpp"
#include "pipeline/raster_tiled.hpp"  // TileStaging
#include "runtime.hpp"

// A tiled matrix multiply, and the first kernel here that is not a renderer.
//
// It exists because two instructions had nowhere to be measured. V_MMA_16X16X16_F32
// had no kernel at all, and cp.async had one whose staged data was read 256 times
// — so hiding the fetch could never matter. A matrix multiply is the other end of
// both: a staged tile is consumed by a fixed, small number of operations, which is
// the ratio the asynchronous copy was designed for.
//
// C = A * B, row-major, with every dimension a multiple of the tile.

// One block computes a 16 x 64 strip of C with four warps, each holding a 16 x 16
// accumulator in registers for the whole K loop.
//
// The thread block is 2 x 16 x 4 so that every index the kernel needs is a
// coordinate rather than a division: x is which half of a fragment row a lane
// holds, y is the row, z is the warp. The ISA has no integer divide, and this is
// what that constraint looks like when it reaches a real kernel.
inline constexpr uint32_t GEMM_TILE = MMA_TILE;                  // 16
inline constexpr uint32_t GEMM_WARPS = 4;                        // a block's warps
inline constexpr uint32_t GEMM_TILE_N = GEMM_TILE * GEMM_WARPS;  // 64 columns a block

// What a k-step stages: one A tile and the four B tiles beside it.
inline constexpr uint32_t GEMM_A_FLOATS = GEMM_TILE * GEMM_TILE;
inline constexpr uint32_t GEMM_B_FLOATS = GEMM_TILE * GEMM_TILE_N;
inline constexpr uint32_t GEMM_STAGE_FLOATS = GEMM_A_FLOATS + GEMM_B_FLOATS;

// Copies a warp issues a k-step, which S_CP_ASYNC_WAIT has to be told and so has
// to be a whole number. Two of A and eight of B for every thread.
inline constexpr uint32_t GEMM_COPIES_A_WARP =
    GEMM_STAGE_FLOATS / (GEMM_WARPS * WARP_SIZE);

struct GemmArgs {
    // Byte offsets from the base of device memory.
    size_t a_offset = 0;
    size_t b_offset = 0;
    size_t c_offset = 0;

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;

    // Whether the warp multiplies its tile in one instruction or the lanes do it
    // one fused multiply-add at a time. The same arithmetic either way, and the
    // same staging — only the inner loop differs.
    bool matrix_unit = true;

    // Whether a fragment is loaded in one instruction or eight.
    //
    // Only the matrix route has fragments; the arithmetic one reads a float at a
    // time by construction. False is what the first figures were taken under.
    bool wide_fragments = false;

    // How the tiles reach shared memory. AsyncDoubleBuffered issues the next
    // k-step's copies before this one is multiplied, which is the overlap the
    // renderer had no room for.
    TileStaging staging = TileStaging::Synchronous;
};

// Builds the kernel. args[0] must point at a GemmArgs that outlives the launch.
Program build_gemm_program(void** args);

// Runs it over the grid the shape implies. Throws unless m and k are multiples of
// 16 and n is a multiple of 64 — a tail would be a different kernel, and refusing
// is honest until a problem needs one.
void run_gemm(MyGPURuntime& rt, const GemmArgs& args);

// The same product on the host, for the tests to check against. Row-major, and
// deliberately the obvious three loops: a reference that shared the kernel's
// blocking could be wrong in the same way it is.
std::vector<float> gemm_reference(const std::vector<float>& a,
                                  const std::vector<float>& b, uint32_t m, uint32_t n,
                                  uint32_t k);
