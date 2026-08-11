#include <cstdint>
#include <vector>

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
};

Buffers upload(MyGPURuntime& rt, const std::vector<Float3>& world,
               const DrawTarget& target)
{
    Buffers b;
    const size_t world_bytes = world.size() * WORLD_VERTEX_BYTES;
    b.frame_bytes = static_cast<size_t>(target.width) * target.height * PIXEL_BYTES;

    b.world = rt.myrt_malloc(world_bytes);
    b.screen = rt.myrt_malloc(world.size() * SCREEN_VERTEX_BYTES);
    b.frame = rt.myrt_malloc(b.frame_bytes);

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

void run_pass_one(MyGPURuntime& rt, const std::vector<Float3>& world,
                  const DrawTarget& target, const Buffers& b)
{
    VertexStageArgs args;
    args.view_projection = target.camera.view_projection(target.aspect());
    args.world_offset = rt.myrt_device_offset(b.world);
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.vertex_count = static_cast<uint32_t>(world.size());
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

}  // namespace

std::vector<Float3> draw_walk(MyGPURuntime& rt, const std::vector<Float3>& world,
                              const DrawTarget& target)
{
    const Buffers b = upload(rt, world, target);
    run_pass_one(rt, world, target, b);

    RasterStageArgs args;
    args.screen_offset = rt.myrt_device_offset(b.screen);
    args.framebuffer_offset = rt.myrt_device_offset(b.frame);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = static_cast<uint32_t>(world.size() / 3);
    run_raster_stage(rt, args);

    return download(rt, b);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const std::vector<Float3>& world,
                               const DrawTarget& target)
{
    const Buffers b = upload(rt, world, target);
    run_pass_one(rt, world, target, b);
    run_tiled_raster_stage(rt, bin_and_upload(rt, world, target, b));
    return download(rt, b);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const std::vector<Float3>& world,
                                const DrawTarget& target)
{
    const Buffers b = upload(rt, world, target);
    run_pass_one(rt, world, target, b);
    run_shared_raster_stage(rt, bin_and_upload(rt, world, target, b));
    return download(rt, b);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const std::vector<Float3>& world,
                                  const DrawTarget& target, const Shading& shading)
{
    // upload allocates a screen buffer this route never touches. Left as it is
    // rather than split: the counters ignore allocation, and one upload path is
    // one fewer place for the two renderers to receive different vertices.
    const Buffers b = upload(rt, world, target);

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
                              const DrawTarget& target)
{
    return draw_walk(rt, mesh.flattened(), target);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const Mesh& mesh,
                               const DrawTarget& target)
{
    return draw_tiled(rt, mesh.flattened(), target);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const Mesh& mesh,
                                const DrawTarget& target)
{
    return draw_shared(rt, mesh.flattened(), target);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const Mesh& mesh,
                                  const DrawTarget& target, const Shading& shading)
{
    return draw_raytrace(rt, mesh.flattened(), target, shading);
}
