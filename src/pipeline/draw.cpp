#include <cstdint>
#include <utility>
#include <vector>

#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/raster_tiled.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/vertex.hpp"

namespace {

// One device buffer per thing the passes address by offset. Held together so
// the three routes cannot disagree about which one is which.
struct Buffers {
    void* world = nullptr;
    void* screen = nullptr;
    void* frame = nullptr;
    size_t frame_bytes = 0;

    // Null for the routes that take a flattened vertex list, which name their
    // triangles by position rather than by index.
    void* index = nullptr;
};

// Allocates exactly what the plan names, and nothing it does not. A zero
// count means the buffer is not bound at all, and its pointer stays null.
Buffers upload(MyGPURuntime& rt, const std::vector<Float3>& world, const BufferPlan& plan)
{
    Buffers b;
    const size_t world_bytes =
        static_cast<size_t>(plan.world_vertices) * WORLD_VERTEX_BYTES;
    b.frame_bytes = static_cast<size_t>(plan.width) * plan.height * PIXEL_BYTES;

    b.world = rt.myrt_malloc(world_bytes);
    if (plan.screen_vertices > 0) {
        b.screen = rt.myrt_malloc(static_cast<size_t>(plan.screen_vertices) *
                                  SCREEN_VERTEX_BYTES);
    }
    if (b.frame_bytes > 0) {
        b.frame = rt.myrt_malloc(b.frame_bytes);
    }

    std::vector<float> flat;
    flat.reserve(world.size() * WORLD_VERTEX_FLOATS);
    for (const Float3& v : world) {
        flat.push_back(v.x);
        flat.push_back(v.y);
        flat.push_back(v.z);
    }
    rt.myrt_memcpy(b.world, flat.data(), world_bytes, Direction::HostToDevice);
    return b;
}

// A plan for the routes that project first: one screen vertex per world vertex,
// and a framebuffer the size of the target.
BufferPlan raster_plan(size_t vertices, const DrawTarget& target)
{
    BufferPlan plan;
    plan.world_vertices = static_cast<uint32_t>(vertices);
    plan.screen_vertices = static_cast<uint32_t>(vertices);
    plan.width = target.width;
    plan.height = target.height;
    return plan;
}

// The same, for an indexed mesh.
//
// The vertex half delegates, and handing over mesh.vertices rather than
// mesh.flattened() is what sizes both the world buffer and the screen buffer by
// the unique count — a cube reserves eight screen slots where the flattened
// list needed thirty-six.
Buffers upload(MyGPURuntime& rt, const Mesh& mesh, const DrawTarget& target)
{
    BufferPlan plan = raster_plan(mesh.vertex_count(), target);
    plan.indices = static_cast<uint32_t>(mesh.indices.size());
    Buffers b = upload(rt, mesh.vertices, plan);

    // Indices reach the device as floats: the ISA has no integer registers, and
    // the kernel multiplies an index by a vertex stride to get an address, so
    // it wants a float there anyway. Whole numbers are exact to 2^24, which is
    // more vertices than a scene here will hold.
    std::vector<float> as_floats;
    as_floats.reserve(mesh.indices.size());
    for (uint32_t i : mesh.indices) {
        as_floats.push_back(static_cast<float>(i));
    }

    const size_t index_bytes = as_floats.size() * sizeof(float);
    b.index = rt.myrt_malloc(index_bytes);
    rt.myrt_memcpy(b.index, as_floats.data(), index_bytes, Direction::HostToDevice);
    return b;
}

// vertex_count rather than the vertices themselves, because that is all pass 1
// depends on: build_vertex_program transforms slot i into slot i and never asks
// whether i is shared.
void run_pass_one(MyGPURuntime& rt, uint32_t vertex_count, const DrawTarget& target,
                  const Buffers& b)
{
    VertexStageArgs args;
    args.view_projection = target.camera.view_projection(target.aspect());
    args.world_offset = rt.myrt_device_offset(b.world);
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.vertex_count = vertex_count;
    args.width = target.width;
    args.height = target.height;
    run_vertex_stage(rt, args);

    // Clears the counters as well as waiting, which is what leaves a caller
    // reading pass 2 alone. Silent: this ends a pass, not a kernel run, and the
    // caller has its own reason to print or not.
    rt.myrt_sync(false);
}

// Binning reads the projected vertices, so it happens after pass 1 — on the
// host, which means reading them back first. Real hardware bins in a geometry
// stage without the round trip.
std::vector<ScreenTriangle> read_back_triangles(MyGPURuntime& rt,
                                                const std::vector<Float3>& world,
                                                const Buffers& b)
{
    std::vector<float> screen(world.size() * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(screen.data(), b.screen, screen.size() * sizeof(float),
                   Direction::DeviceToHost);

    std::vector<ScreenTriangle> triangles;
    for (size_t i = 0; i + TILE_TRIANGLE_FLOATS <= screen.size();
         i += TILE_TRIANGLE_FLOATS) {
        triangles.push_back(
            ScreenTriangle{Float3{screen[i + 0], screen[i + 1], screen[i + 2]},
                           Float3{screen[i + 4], screen[i + 5], screen[i + 6]},
                           Float3{screen[i + 8], screen[i + 9], screen[i + 10]},
                           screen[i + 3], screen[i + 7], screen[i + 11]});
    }
    return triangles;
}

// The same, resolving indices instead of slicing.
//
// The flattened version can walk the buffer twelve floats at a time because a
// triangle *is* three consecutive vertices there. Here the buffer holds each
// vertex once, so the only thing that says which three belong together is
// mesh.indices — and an index counts vertices, not floats, which is what the
// multiply by SCREEN_VERTEX_FLOATS is for.
//
// This is the only place the two forms differ. bin_triangles and everything
// below it sees a list of ScreenTriangle and never learns an index existed.
std::vector<ScreenTriangle> read_back_triangles(MyGPURuntime& rt, const Mesh& mesh,
                                                const Buffers& b)
{
    std::vector<float> screen(mesh.vertex_count() * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(screen.data(), b.screen, screen.size() * sizeof(float),
                   Direction::DeviceToHost);

    const auto at = [&screen](uint32_t i) {
        const size_t v = static_cast<size_t>(i) * SCREEN_VERTEX_FLOATS;
        return std::pair<Float3, float>{
            Float3{screen[v + 0], screen[v + 1], screen[v + 2]}, screen[v + 3]};
    };

    std::vector<ScreenTriangle> triangles;
    for (size_t t = 0; t < mesh.triangle_count(); ++t) {
        const auto [v0, iw0] = at(mesh.indices[t * 3 + 0]);
        const auto [v1, iw1] = at(mesh.indices[t * 3 + 1]);
        const auto [v2, iw2] = at(mesh.indices[t * 3 + 2]);
        triangles.push_back(ScreenTriangle{v0, v1, v2, iw0, iw1, iw2});
    }
    return triangles;
}

std::vector<Float3> download(MyGPURuntime& rt, const Buffers& b)
{
    std::vector<float> out(b.frame_bytes / sizeof(float), 0.0f);
    rt.myrt_memcpy(out.data(), b.frame, b.frame_bytes, Direction::DeviceToHost);

    std::vector<Float3> frame;
    frame.reserve(out.size() / PIXEL_FLOATS);
    for (size_t i = 0; i < out.size(); i += PIXEL_FLOATS) {
        frame.push_back(Float3{out[i + 0], out[i + 1], out[i + 2]});
    }
    return frame;
}

// The half the two tiled routes share: bin, upload the runs, fill in the
// offsets. max_tile_triangles is filled whether or not the caller needs it —
// the global-memory kernel ignores it.
TiledRasterStageArgs bin_and_upload(MyGPURuntime& rt, const std::vector<Float3>& world,
                                    const DrawTarget& target, const Buffers& b)
{
    const TileBinning binning =
        bin_triangles(read_back_triangles(rt, world, b), target.width, target.height);

    void* verts = rt.myrt_malloc(binning.vertices.size() * sizeof(float));
    void* table = rt.myrt_malloc(binning.table.size() * sizeof(float));
    rt.myrt_memcpy(verts, binning.vertices.data(),
                   binning.vertices.size() * sizeof(float), Direction::HostToDevice);
    rt.myrt_memcpy(table, binning.table.data(), binning.table.size() * sizeof(float),
                   Direction::HostToDevice);

    uint32_t fullest = 0;
    for (uint32_t t = 0; t < binning.tile_count(); ++t) {
        const uint32_t count = static_cast<uint32_t>(binning.table[t * 2 + 1]);
        if (count > fullest) {
            fullest = count;
        }
    }

    TiledRasterStageArgs args;
    args.tile_vertices_offset = rt.myrt_device_offset(verts);
    args.tile_table_offset = rt.myrt_device_offset(table);
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.width = target.width;
    args.height = target.height;
    args.tiles_x = binning.tiles_x;
    args.max_tile_triangles = fullest;
    return args;
}

TiledRasterStageArgs bin_and_upload(MyGPURuntime& rt, const Mesh& mesh,
                                    const DrawTarget& target, const Buffers& b)
{
    const TileBinning binning =
        bin_triangles(read_back_triangles(rt, mesh, b), target.width, target.height);

    void* verts = rt.myrt_malloc(binning.vertices.size() * sizeof(float));
    void* table = rt.myrt_malloc(binning.table.size() * sizeof(float));
    rt.myrt_memcpy(verts, binning.vertices.data(),
                   binning.vertices.size() * sizeof(float), Direction::HostToDevice);
    rt.myrt_memcpy(table, binning.table.data(), binning.table.size() * sizeof(float),
                   Direction::HostToDevice);

    uint32_t fullest = 0;
    for (uint32_t t = 0; t < binning.tile_count(); ++t) {
        const uint32_t count = static_cast<uint32_t>(binning.table[t * 2 + 1]);
        if (count > fullest) {
            fullest = count;
        }
    }

    TiledRasterStageArgs args;
    args.tile_vertices_offset = rt.myrt_device_offset(verts);
    args.tile_table_offset = rt.myrt_device_offset(table);
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.width = target.width;
    args.height = target.height;
    args.tiles_x = binning.tiles_x;
    args.max_tile_triangles = fullest;
    return args;
}

}  // namespace

size_t BufferPlan::device_bytes() const
{
    return static_cast<size_t>(world_vertices) * WORLD_VERTEX_BYTES +
           static_cast<size_t>(screen_vertices) * SCREEN_VERTEX_BYTES +
           static_cast<size_t>(indices) * sizeof(float) +
           static_cast<size_t>(width) * height * PIXEL_BYTES;
}

size_t BufferPlan::binned_bytes(uint32_t width, uint32_t height, uint32_t triangles)
{
    const size_t tiles = static_cast<size_t>((width + TILE_WIDTH - 1) / TILE_WIDTH) *
                         ((height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    // A triangle is copied into every tile its bounding box reaches, so the
    // bound is one entry per tile per triangle — plus two floats a tile saying
    // where its run starts and how long it is.
    return tiles * triangles * TILE_TRIANGLE_FLOATS * sizeof(float) +
           tiles * 2 * sizeof(float);
}

std::vector<Float3> draw_walk(MyGPURuntime& rt, const std::vector<Float3>& world,
                              const DrawTarget& target, bool predicated)
{
    const Buffers b = upload(rt, world, raster_plan(world.size(), target));
    run_pass_one(rt, static_cast<uint32_t>(world.size()), target, b);

    RasterStageArgs args;
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = static_cast<uint32_t>(world.size() / 3);
    args.predicated = predicated;
    run_raster_stage(rt, args);

    return download(rt, b);
}

std::vector<Float3> draw_predicated(MyGPURuntime& rt, const std::vector<Float3>& world,
                                    const DrawTarget& target)
{
    return draw_walk(rt, world, target, true);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const std::vector<Float3>& world,
                               const DrawTarget& target, bool predicated)
{
    const Buffers b = upload(rt, world, raster_plan(world.size(), target));
    run_pass_one(rt, static_cast<uint32_t>(world.size()), target, b);

    TiledRasterStageArgs args = bin_and_upload(rt, world, target, b);
    args.predicated = predicated;
    run_tiled_raster_stage(rt, args);
    return download(rt, b);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const std::vector<Float3>& world,
                                const DrawTarget& target, bool predicated)
{
    const Buffers b = upload(rt, world, raster_plan(world.size(), target));
    run_pass_one(rt, static_cast<uint32_t>(world.size()), target, b);

    TiledRasterStageArgs args = bin_and_upload(rt, world, target, b);
    args.predicated = predicated;
    run_shared_raster_stage(rt, args);
    return download(rt, b);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const std::vector<Float3>& world,
                                  const DrawTarget& target, const Shading& shading)
{
    // screen_vertices stays zero: there is no vertex stage here, so nothing
    // ever projects and no buffer holds the result.
    BufferPlan plan;
    plan.world_vertices = static_cast<uint32_t>(world.size());
    plan.width = target.width;
    plan.height = target.height;
    const Buffers b = upload(rt, world, plan);

    RaytraceStageArgs args;
    args.basis = ray_basis(target.camera, target.aspect());
    args.shading = shading;
    args.triangles_offset = rt.myrt_device_offset(b.world);
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = static_cast<uint32_t>(world.size() / 3);
    run_raytrace_stage(rt, args);

    return download(rt, b);
}

std::vector<Float3> draw_walk(MyGPURuntime& rt, const Mesh& mesh,
                              const DrawTarget& target, bool predicated)
{
    const Buffers b = upload(rt, mesh, target);

    // The whole of what indexing buys is this one argument: a cube runs eight
    // threads here where the flattened list ran thirty-six, and a transform is
    // the most expensive instruction in the set.
    run_pass_one(rt, mesh.vertex_count(), target, b);

    RasterStageArgs args;
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.index_offset = rt.myrt_device_offset(b.index);
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = mesh.triangle_count();
    args.indexed = true;
    args.predicated = predicated;
    run_raster_stage(rt, args);
    return download(rt, b);
}

std::vector<Float3> draw_predicated(MyGPURuntime& rt, const Mesh& mesh,
                                    const DrawTarget& target)
{
    return draw_walk(rt, mesh, target, true);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const Mesh& mesh,
                               const DrawTarget& target, bool predicated)
{
    // Line for line the flattened route, and the kernel is the same program:
    // bin_triangles copies each triangle into every tile it reaches, so what
    // reaches the device is already de-indexed. Indexing costs this route
    // nothing and saves it a transform per shared corner.
    const Buffers b = upload(rt, mesh, target);
    run_pass_one(rt, mesh.vertex_count(), target, b);
    TiledRasterStageArgs args = bin_and_upload(rt, mesh, target, b);
    args.predicated = predicated;
    run_tiled_raster_stage(rt, args);
    return download(rt, b);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const Mesh& mesh,
                                const DrawTarget& target, bool predicated)
{
    const Buffers b = upload(rt, mesh, target);
    run_pass_one(rt, mesh.vertex_count(), target, b);

    TiledRasterStageArgs args = bin_and_upload(rt, mesh, target, b);
    args.predicated = predicated;
    run_shared_raster_stage(rt, args);
    return download(rt, b);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const Mesh& mesh,
                                  const DrawTarget& target, const Shading& shading)
{
    // Flattened on purpose, and it stays that way — this is not the rasteriser
    // route waiting its turn.
    //
    // Indexing pays for itself in a vertex stage: a corner shared by six
    // triangles is transformed once instead of six times. The ray tracer has no
    // vertex stage at all. It reads world triangles where they already lie, so
    // there is no transform to save and the index buffer would be pure cost —
    // three dependent loads a triangle on top of the nine it already makes.
    //
    // Real ray tracing does take an index buffer — DXR and Vulkan RT both name
    // one in their geometry description. What differs is when it is read. The
    // acceleration-structure builder consumes it once, and per-ray traversal
    // then reads the structure's own leaves rather than following indices. The
    // post-transform cache an index buffer feeds on the raster side has no
    // counterpart, there being no per-ray vertex transform to cache.
    //
    // This tracer is a step below even that: no acceleration structure, a
    // linear walk of every triangle. Indexing a brute-force walk is cost with
    // nothing on the other side of it.
    return draw_raytrace(rt, mesh.flattened(), target, shading);
}

SchedulerStats vertex_stage_cost(const std::vector<Float3>& vertices,
                                 const DrawTarget& target)
{
    // No width or height in the plan, so no framebuffer: this runs pass 1 and
    // never writes a pixel. Saying so is the point of BufferPlan — this path
    // used to reach for a raster upload and reserve three megabytes of frame at
    // 512, which the runtime it had been given could not hold.
    BufferPlan plan;
    plan.world_vertices = static_cast<uint32_t>(vertices.size());
    plan.screen_vertices = plan.world_vertices;

    // Its own runtime, so the reading is a total rather than a difference, and
    // run_pass_one's sync is what leaves the counters holding this pass alone.
    MyGPURuntime rt(plan.device_bytes() + (1u << 16));
    const Buffers b = upload(rt, vertices, plan);

    VertexStageArgs args;
    args.view_projection = target.camera.view_projection(target.aspect());
    args.world_offset = rt.myrt_device_offset(b.world);
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.vertex_count = plan.world_vertices;
    args.width = target.width;
    args.height = target.height;
    run_vertex_stage(rt, args);

    return rt.stats();
}
