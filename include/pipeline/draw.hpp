#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"
#include "mesh.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"

// The four routes from world triangles to a frame, end to end: allocate,
// upload, run the passes, read the framebuffer back.
//
// They exist as one implementation because measuring them against each other is
// the point. A benchmark and a test that each drove the pipeline their own way
// would drift, and the figures in benchmarks/RESULTS.md would stop describing
// what the tests check.

// Where a draw lands, and through which camera.
//
// The camera is kept rather than a matrix because the two renderers need
// different things from it — the rasteriser a view-projection, the ray tracer a
// basis of three vectors — and a caller handed both could pass a mismatched
// pair. Comparing them only means anything if they saw the same scene from the
// same place.
struct DrawTarget {
    uint32_t width = 0;
    uint32_t height = 0;
    Camera camera;

    float aspect() const
    {
        return static_cast<float>(width) / static_cast<float>(height);
    }
};

// The three rasteriser routes each run pass 1, then myrt_sync, then pass 2, and
// return the framebuffer a pixel at a time.
//
// The sync between the passes clears the counters, so a caller reads pass 2
// alone. Pass 1 puts three vertices in a 32-lane warp and leaves 29 of them
// masked, which would swamp any figure taken across both.

// Every pixel walks every triangle. The baseline the other two are measured
// against, and O(pixels x triangles).
std::vector<Float3> draw_walk(MyGPURuntime& rt, const std::vector<Float3>& world,
                              const DrawTarget& target);

// Triangles binned into screen tiles on the host, one block per tile, each
// pixel walking only its own tile's list.
std::vector<Float3> draw_tiled(MyGPURuntime& rt, const std::vector<Float3>& world,
                               const DrawTarget& target);

// As draw_tiled, with each tile's triangles staged through shared memory once
// per block instead of read from global by every pixel.
//
// Throws when a tile holds more triangles than SHARED_TRIANGLE_CAPACITY, which
// is the same refusal run_shared_raster_stage makes — reported here so the
// caller learns which scene it was.
std::vector<Float3> draw_shared(MyGPURuntime& rt, const std::vector<Float3>& world,
                                const DrawTarget& target);

// The other renderer entirely: one thread per pixel, casting a ray at the world
// triangles where they already are.
//
// One launch, not two. There is no vertex stage to sync away, so what a caller
// reads is the whole draw — which is the right comparison, the rasteriser's
// pass 1 existing only because it has to project first.
//
// Defaults to Barycentric so a frame can be compared against the rasteriser's;
// Diffuse is the mode the rasteriser cannot follow it into.
std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const std::vector<Float3>& world,
                                  const DrawTarget& target,
                                  const Shading& shading = Shading{});

// The same four, taking a mesh.
//
// They flatten on the host and call the overloads above, so an indexed mesh
// renders identically to the vertex list it expands to — which is what the
// tests assert. Pass 1 still transforms one thread per flattened vertex; making
// it transform each unique vertex once is a separate change, with a number
// attached to it.
std::vector<Float3> draw_walk(MyGPURuntime& rt, const Mesh& mesh,
                              const DrawTarget& target);
std::vector<Float3> draw_tiled(MyGPURuntime& rt, const Mesh& mesh,
                               const DrawTarget& target);
std::vector<Float3> draw_shared(MyGPURuntime& rt, const Mesh& mesh,
                                const DrawTarget& target);
std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const Mesh& mesh,
                                  const DrawTarget& target,
                                  const Shading& shading = Shading{});
