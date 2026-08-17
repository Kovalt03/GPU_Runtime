#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"
#include "mesh.hpp"
#include "pipeline/raster_tiled.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

// The four routes from world triangles to a frame, end to end: allocate,
// upload, run the passes, read the framebuffer back.
//
// They exist as one implementation because measuring them against each other is
// the point. A benchmark and a test that each drove the pipeline their own way
// would drift, and the figures in benchmarks/RESULTS.md would stop describing
// what the tests check.

// How much device memory a draw will ask for, stated rather than inferred.
//
// A graphics API makes the host do this — glBufferData and CreateBuffer both
// take a byte count — because only the host knows how many vertices it is
// about to send and how large a target it is drawing into. Inferring the sizes
// inside the upload is what let a path that never writes a pixel quietly
// reserve a framebuffer, and a benchmark size the runtime for a scene it could
// not hold.
//
// device_bytes() is what MyGPURuntime has to be given. Callers that go on to
// bin triangles add binned_bytes() to it.
struct BufferPlan {
    uint32_t world_vertices = 0;

    // One per world vertex: pass 1 runs a thread each and writes x, y, depth
    // and 1/w. Zero for a route with no vertex stage — the ray tracer reads
    // world triangles where they lie and never projects.
    uint32_t screen_vertices = 0;

    // Three per triangle, when the geometry arrives indexed.
    uint32_t indices = 0;

    // Zero on a path that measures rather than renders.
    uint32_t width = 0;
    uint32_t height = 0;

    size_t device_bytes() const;

    // What binning costs at worst: a triangle is copied into every tile its
    // bounding box reaches, so the bound is one entry per tile per triangle.
    static size_t binned_bytes(uint32_t width, uint32_t height, uint32_t triangles);
};

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
                              const DrawTarget& target, bool predicated = false);

// Triangles binned into screen tiles on the host, one block per tile, each
// pixel walking only its own tile's list.
std::vector<Float3> draw_tiled(MyGPURuntime& rt, const std::vector<Float3>& world,
                               const DrawTarget& target, bool predicated = false);

// As draw_tiled, with each tile's triangles staged through shared memory once
// per block instead of read from global by every pixel.
//
// Throws when a tile holds more triangles than SHARED_TRIANGLE_CAPACITY, which
// is the same refusal run_shared_raster_stage makes — reported here so the
// caller learns which scene it was.
std::vector<Float3> draw_shared(MyGPURuntime& rt, const std::vector<Float3>& world,
                                const DrawTarget& target, bool predicated = false);

// draw_walk with the coverage branch blended instead. Kept as a name because
// the branch against the blend is what the flag was added to measure; every
// route here takes it, so the question can be put to any of them.
//
// Four distinct pass-2 programs answer it, not six: bin_triangles de-indexes on
// the host, so the tiled and shared routes run the same kernel whether they
// were handed a mesh or a vertex list, and only the walk has an indexed form.
std::vector<Float3> draw_predicated(MyGPURuntime& rt, const std::vector<Float3>& world,
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
                                  const Shading& shading = Shading{},
                                  bool predicated = false);

// The same four, taking a mesh. Each renders identically to the vertex list the
// mesh expands to, which is what the tests assert.
//
// All four run pass 1 over the unique vertices, so a corner shared by six
// triangles is transformed once. What they do with the index buffer afterwards
// differs: the walk carries it into pass 2 and pays three dependent loads a
// triangle, the tiled pair have already been de-indexed by bin_triangles, and
// the ray tracer never had a vertex stage to save — see its overload.
std::vector<Float3> draw_walk(MyGPURuntime& rt, const Mesh& mesh,
                              const DrawTarget& target, bool predicated = false);

std::vector<Float3> draw_predicated(MyGPURuntime& rt, const Mesh& mesh,
                                    const DrawTarget& target);
std::vector<Float3> draw_tiled(MyGPURuntime& rt, const Mesh& mesh,
                               const DrawTarget& target, bool predicated = false);
std::vector<Float3> draw_shared(MyGPURuntime& rt, const Mesh& mesh,
                                const DrawTarget& target, bool predicated = false);
std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const Mesh& mesh,
                                  const DrawTarget& target,
                                  const Shading& shading = Shading{});

// --- geometry that outlives a draw ------------------------------------------
// Every overload above uploads, draws and releases. That is what one rendered
// frame wants and what no real program does: a vertex buffer is uploaded once
// and drawn from every frame after.
//
// Two things follow from the difference. The same mesh drawn twice through the
// overloads above is two uploads at two addresses, so each draw asks the cache a
// question nobody has asked before and every miss it reports is compulsory. And
// a depth prepass cannot be written at all, its two passes having to read one
// copy of the geometry.

// What a route needs of the geometry beyond the vertices themselves.
enum class VertexStage {
    // The rasteriser routes: pass 1 projects each vertex and needs somewhere to
    // put the result.
    Projects,

    // The ray tracer, which reads world triangles where they lie. Naming it
    // keeps a path that never projects from reserving a buffer it never writes.
    None,
};

// Vertices on the device, and where pass 1 will leave them. Buffers a draw
// writes but does not own — the framebuffer — are in DeviceFrame instead, so
// that one geometry can be drawn into two frames and from two cameras.
struct DeviceGeometry {
    void* world = nullptr;   // world-space vertices; the unique ones if indexed
    void* screen = nullptr;  // pass 1's output, one slot a world vertex
    void* index = nullptr;   // null when the geometry arrived flattened

    // One unit normal a triangle, for the routes that light what they draw.
    // Uploaded with the vertices rather than derived per pixel, and only for
    // geometry that will go through a vertex stage — the ray tracer takes the
    // cross product of edges it has already loaded.
    void* normals = nullptr;

    uint32_t vertex_count = 0;
    uint32_t triangle_count = 0;

    // The index list stays on the host too, because binning runs here: the
    // tiled routes resolve indices while building their tile lists. Hardware
    // bins on the device and needs no second copy.
    std::vector<uint32_t> indices;

    bool indexed() const
    {
        return index != nullptr;
    }
};

DeviceGeometry upload(MyGPURuntime& rt, const std::vector<Float3>& world,
                      VertexStage stage = VertexStage::Projects);

// An indexed upload, which is what makes pass 1 transform a shared corner once.
// The ray tracer cannot draw one — see its overload below.
DeviceGeometry upload(MyGPURuntime& rt, const Mesh& mesh);

// Where a draw puts its pixels, kept apart from the geometry because it belongs
// to the target rather than to the model.
struct DeviceFrame {
    void* pixels = nullptr;

    // One float a pixel, written by a depth prepass and read by the pass that
    // follows it. Allocated with the frame rather than on demand: it is the same
    // size as a quarter of the colour buffer and a route that wants one wants it
    // for the frame's whole life.
    void* depth = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
};

DeviceFrame allocate_frame(MyGPURuntime& rt, const DrawTarget& target);
std::vector<Float3> read_back(MyGPURuntime& rt, const DeviceFrame& frame);

// Hands the buffers back and leaves the handle empty, so releasing twice is
// harmless. The first callers of myrt_free outside the tests and ray_triangle.
void release(MyGPURuntime& rt, DeviceGeometry& geometry);
void release(MyGPURuntime& rt, DeviceFrame& frame);

// The routes over what is already resident. The camera stays in the target: the
// same buffers drawn from two viewpoints is the point of holding them.
//
// The tiled pair still allocate per draw, their tile lists being a function of
// the camera as well as the geometry, and that is where binning belongs.
// shading last rather than beside the ray tracer's, which takes it before
// predicated: adding it there would have moved every existing call site's
// arguments by one and turned a compile error into a silent reinterpretation.
//
// Diffuse is the walk's alone. The tiled pair read their triangles from tile
// lists that carry screen positions only, and would need those lists to grow by
// a world position a vertex and a normal a triangle; they refuse it instead.
std::vector<Float3> draw_walk(MyGPURuntime& rt, const DeviceGeometry& geometry,
                              const DeviceFrame& frame, const DrawTarget& target,
                              bool predicated = false,
                              const Shading& shading = Shading{});
// Both take the shading argument and both refuse Diffuse, which is a better
// answer than drawing an unlit frame and letting a caller believe otherwise.
std::vector<Float3> draw_tiled(MyGPURuntime& rt, const DeviceGeometry& geometry,
                               const DeviceFrame& frame, const DrawTarget& target,
                               bool predicated = false,
                               const Shading& shading = Shading{});
std::vector<Float3> draw_shared(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                const DeviceFrame& frame, const DrawTarget& target,
                                bool predicated = false,
                                const Shading& shading = Shading{});

// Throws on indexed geometry rather than resolving it. The index buffer exists
// to feed a vertex stage and this route has none, so an indexed upload is not a
// form of the geometry it can read — upload(rt, mesh.flattened(), VertexStage::None)
// is what it wants.
std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                  const DeviceFrame& frame, const DrawTarget& target,
                                  const Shading& shading = Shading{},
                                  bool predicated = false);

// The walk again, with a depth prepass in front of it.
//
// Two launches over the same geometry: the first keeps the nearest depth a pixel
// and colours nothing, the second colours only the triangle that depth names. A
// pixel is therefore shaded once however many triangles cover it, which is what
// makes it worth having on a scene with depth complexity — and it walks every
// triangle twice, which is what makes it a trade rather than a saving.
//
// Unlike the routes above this does not sync between its passes, so the counters
// a caller reads hold both. Against draw_walk's single pass, which is the
// comparison the trade is about.
std::vector<Float3> draw_early_z(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                 const DeviceFrame& frame, const DrawTarget& target,
                                 const Shading& shading = Shading{});

// What pass 1 costs on its own.
//
// The draw routes clear the counters between their passes so a caller reads
// pass 2 alone, which leaves the vertex stage — the half indexing actually
// saves — with nowhere to be seen. This runs it and nothing else.
SchedulerStats vertex_stage_cost(const std::vector<Float3>& vertices,
                                 const DrawTarget& target);
