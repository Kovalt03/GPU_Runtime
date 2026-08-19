#pragma once

#include <cstdint>
#include <vector>

#include "bvh.hpp"
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
// One object drawn from shared geometry: where it is, and nothing else yet.
//
// A material would belong here too and is not here, because nothing reads one.
// It arrives with the ray tracer's second level, where the instance a ray hits
// is what decides the shader — which is the divergence SER exists for, and the
// first in this repository that will not have been built to be measured.
struct Instance {
    Float4x4 model;
};

struct DeviceGeometry {
    void* world = nullptr;   // world-space vertices; the unique ones if indexed
    void* screen = nullptr;  // pass 1's output, one slot a world vertex
    void* index = nullptr;   // null when the geometry arrived flattened

    // One unit normal a triangle, for the routes that light what they draw.
    // Uploaded with the vertices rather than derived per pixel, and only for
    // geometry that will go through a vertex stage — the ray tracer takes the
    // cross product of edges it has already loaded.
    void* normals = nullptr;

    // A tree over the triangles, when one was asked for. The ray tracer walks it
    // instead of the whole list; nothing else looks at it.
    //
    // The triangles are uploaded in the tree's order, not the caller's, because a
    // leaf is a range. So a geometry built with an acceleration structure has a
    // different `world` from one without, and the frames still have to match —
    // which is the check that says the permutation was applied consistently.
    void* bvh = nullptr;
    uint32_t bvh_depth = 0;
    TraversalOrder bvh_order = TraversalOrder::Unordered;

    // The second level, when the geometry was uploaded to be drawn many times.
    // The tree above holds one copy in its own space; these hold where the
    // copies went and how to get a ray into each.
    void* tlas = nullptr;
    void* tlas_instances = nullptr;
    uint32_t tlas_depth = 0;

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

// The same, with a bounding volume hierarchy built over the triangles and
// uploaded beside them.
//
// Separate from the overload above rather than a defaulted argument: building
// one is host work proportional to the scene, and a benchmark that did it
// without meaning to would be timing the builder.
DeviceGeometry upload_accelerated(MyGPURuntime& rt, const std::vector<Float3>& world,
                                  BvhSplit split = BvhSplit::SAH,
                                  uint32_t leaf_size = BVH_DEFAULT_LEAF,
                                  TraversalOrder order = TraversalOrder::Unordered);

// Several pieces of geometry, each placed as often as a caller likes, with a
// tree over the placements.
//
// One lower-level tree a piece of geometry however many copies of it there are:
// what a ray walks at an instance is that tree, reached by moving the ray into
// the instance's space rather than by building geometry there. Instancing pays
// for itself in memory before it pays for anything in time.
//
// `instances` names geometries by index and carries a material each. The
// material reaches a fragment shader and nothing else reads it — what makes
// lanes of a warp take different paths has to come from the scene.
DeviceGeometry upload_scene(MyGPURuntime& rt,
                            const std::vector<std::vector<Float3>>& geometries,
                            const std::vector<TlasInstance>& instances,
                            BvhSplit split = BvhSplit::SAH,
                            uint32_t leaf_size = BVH_DEFAULT_LEAF,
                            TraversalOrder order = TraversalOrder::Unordered);

// One geometry placed many times, which is the case above with a single entry
// and every instance naming it. Kept because most of the tests and both
// benchmarks want exactly that and say so more clearly this way.
DeviceGeometry upload_instanced_accelerated(
    MyGPURuntime& rt, const std::vector<Float3>& world,
    const std::vector<Instance>& instances, BvhSplit split = BvhSplit::SAH,
    uint32_t leaf_size = BVH_DEFAULT_LEAF,
    TraversalOrder order = TraversalOrder::Unordered);

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
// Fills the frame's colour and depth, which is what a frame needs before
// anything can be drawn into it rather than over it.
//
// Every mode but DepthUse::Test starts a pixel empty and ends by overwriting it,
// so none of them ever needed this — a thread owned its pixel outright and the
// frame's previous contents were nobody's business. A depth-tested draw reads
// what is there, and what is there has to have been put there.
//
// depth defaults beyond the far plane, so the first covering triangle wins.
void clear_frame(MyGPURuntime& rt, const DeviceFrame& frame, const DrawTarget& target,
                 Float3 colour = Float3{0.0f, 0.0f, 0.0f}, float depth = 2.0f);

// The same, queued on a stream and not waited for.
//
// A clear has nothing to do with the frame being drawn, so it need not happen
// between two draws — it needs only to have happened before the frame it clears
// is drawn into. Whoever draws next drains the queue, which is why a synchronous
// draw route is enough to collect it.
void queue_clear(MyGPURuntime& rt, const DeviceFrame& frame, const DrawTarget& target,
                 Float3 colour, float depth, StreamId stream);

// A draw that lands in the frame rather than over it: the depth buffer decides,
// and a pixel this draw does not cover keeps what the last one left.
//
// clear_frame first, or the frame starts from whatever the allocation held.
std::vector<Float3> draw_depth_tested(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                      const DeviceFrame& frame, const DrawTarget& target,
                                      const Shading& shading = Shading{});

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
// async_staging says how the tile reaches shared memory: through a register and
// a wait a float, or a chunk at a time through copies the warp does not wait for.
// Same frame either way — it is a cost question, and benchmarks/RESULTS.md has
// the answer, along with the reason it is a small one here.
std::vector<Float3> draw_shared(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                const DeviceFrame& frame, const DrawTarget& target,
                                bool predicated = false,
                                const Shading& shading = Shading{},
                                bool async_staging = false);

// Throws on indexed geometry rather than resolving it. The index buffer exists
// to feed a vertex stage and this route has none, so an indexed upload is not a
// form of the geometry it can read — upload(rt, mesh.flattened(), VertexStage::None)
// is what it wants.
std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                  const DeviceFrame& frame, const DrawTarget& target,
                                  const Shading& shading = Shading{},
                                  bool predicated = false);

// The same geometry drawn several times, each with its own transform.
//
// One pass 1 over vertex_count x instances, laid out instance-major, so pass 2
// sees a single longer vertex list at the stride it already used. Every raster
// route below therefore draws instances without knowing they exist.
//
// `transform` picks where the model matrix meets the view-projection, which is a
// crossing rather than a winner — see InstanceTransform.
std::vector<Float3> draw_instanced(
    MyGPURuntime& rt, const DeviceGeometry& geometry, const DeviceFrame& frame,
    const DrawTarget& target, const std::vector<Instance>& instances,
    InstanceTransform transform = InstanceTransform::PerVertex);

// How much room the buffers of an instanced draw need. Pass 1 writes a screen
// vertex an instance, so the screen buffer is the one that grows.
size_t instanced_screen_bytes(uint32_t vertex_count, uint32_t instances,
                              uint32_t varying_count = 0);

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
// uniforms says how the matrix reaches the kernel: baked into the program as
// sixteen moves, or read from the constant window in one instruction. The frame
// is the same and the programs are not — a baked one serves one matrix, and the
// figures taken before the window existed all used it.
enum class Uniforms {
    Baked,
    Window,
};

SchedulerStats vertex_stage_cost(const std::vector<Float3>& vertices,
                                 const DrawTarget& target,
                                 Uniforms uniforms = Uniforms::Baked);

// Pass 1 again, over instances, and the composition pass ahead of it when there
// is one. The sum of both launches, since the whole question is whether the one
// pays for itself in the other.
//
// Separate from a draw route for the reason above, and more sharply here: pass 2
// grows with the instances too and swamps the difference completely — the two
// arms come out identical to the last lane operation through draw_instanced.
SchedulerStats instanced_vertex_cost(const std::vector<Float3>& vertices,
                                     const DrawTarget& target,
                                     const std::vector<Instance>& instances,
                                     InstanceTransform transform);
