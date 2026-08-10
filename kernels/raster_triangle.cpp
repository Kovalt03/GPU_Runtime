// The same triangle kernels/ray_triangle.cpp renders, by the other route:
// project the vertices, then test coverage per pixel.
//
// This is the application level of the pipeline — the layer a user of the
// runtime writes, holding a scene and a camera and calling the two stages. It
// also exists to be looked at. The unit tests compare the device against a host
// reference, which cannot catch a sign convention that is wrong in both; two
// renderers built on different mathematics arriving at the same picture can.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pipeline/raster.hpp"
#include "pipeline/types.hpp"
#include "pipeline/vertex.hpp"
#include "ppm.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

namespace {

// Identical to the ray tracer's Scene: apex up, base along the bottom, at
// z = -2 and large enough that its edges cross plenty of warps.
const std::vector<Float3> TRIANGLE = {
    Float3{0.0f, 0.5f, -2.0f},    // apex
    Float3{-0.5f, -0.5f, -2.0f},  // bottom left
    Float3{0.5f, -0.5f, -2.0f},   // bottom right
};

// The same triangle again, each copy a little further away. Not a scene worth
// looking at — it exists so the cost of walking N triangles can be measured
// before there is a mesh loader. One copy reproduces the ray tracer exactly,
// and the nearest always wins, so the picture never changes however many are
// added: any growth in the counters is the walk and nothing else.
std::vector<Float3> scene(uint32_t triangles)
{
    std::vector<Float3> world;
    for (uint32_t t = 0; t < triangles; ++t) {
        const float back = -0.01f * static_cast<float>(t);
        for (const Float3& v : TRIANGLE) {
            world.push_back(Float3{v.x, v.y, v.z + back});
        }
    }
    return world;
}

// Issued work, which is reproducible; the GIOPS figure alongside it is not,
// depending as it does on how fast the host happens to be.
void report(const char* label, const SchedulerStats& before, const SchedulerStats& after)
{
    SchedulerStats d;
    d.warp_steps = after.warp_steps - before.warp_steps;
    d.active_lane_ops = after.active_lane_ops - before.active_lane_ops;
    d.weighted_lane_ops = after.weighted_lane_ops - before.weighted_lane_ops;

    std::printf("%-14s %12llu %14llu %14llu %9.1f%%\n", label,
                static_cast<unsigned long long>(d.warp_steps),
                static_cast<unsigned long long>(d.active_lane_ops),
                static_cast<unsigned long long>(d.weighted_lane_ops),
                100.0 * d.divergence_rate());
}

Camera scene_camera()
{
    Camera cam;
    cam.eye = Float3{0.0f, 0.0f, 0.0f};
    cam.target = Float3{0.0f, 0.0f, -1.0f};
    cam.up = Float3{0.0f, 1.0f, 0.0f};

    // The ray tracer builds each ray as (x - 0.5, 0.5 - y, -1) over the unit
    // square, so its frustum is half a unit high at unit depth: fov_y is
    // 2 * atan(0.5). Matching it is what makes the two images comparable.
    //
    // It applies no aspect correction, so the two agree exactly only on a
    // square frame. This one does correct, since a projection matrix that
    // ignored the aspect would be wrong rather than merely different.
    cam.fov_y_degrees = 2.0f * std::atan(0.5f) * 180.0f / 3.14159265358979323846f;
    cam.near_z = 0.1f;
    cam.far_z = 100.0f;
    return cam;
}

}  // namespace

int main(int argc, char** argv)
{
    const uint32_t width = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 256;
    const uint32_t height =
        (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : width;
    const uint32_t triangles = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3])) : 1;

    const std::vector<Float3> world = scene(triangles);

    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t world_bytes = world.size() * WORLD_VERTEX_BYTES;
    const size_t screen_bytes = world.size() * SCREEN_VERTEX_BYTES;
    const size_t frame_bytes = pixels * PIXEL_BYTES;

    MyGPURuntime rt(frame_bytes + screen_bytes + (1u << 20));

    // Three separate allocations, which is why the offsets are asked for rather
    // than assumed: only the first of them sits at zero.
    void* world_dev = rt.myrt_malloc(world_bytes);
    void* screen_dev = rt.myrt_malloc(screen_bytes);
    void* frame_dev = rt.myrt_malloc(frame_bytes);

    std::vector<float> world_flat;
    for (const Float3& v : world) {
        world_flat.push_back(v.x);
        world_flat.push_back(v.y);
        world_flat.push_back(v.z);
    }
    rt.myrt_memcpy(world_dev, world_flat.data(), world_bytes, Direction::HostToDevice);

    const Camera camera = scene_camera();
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    std::printf("rasterising %ux%u — %zu pixels, %u triangles\n\n", width, height, pixels,
                triangles);
    std::printf("%-14s %12s %14s %14s %10s\n", "", "warp steps", "lane ops", "weighted",
                "divergence");

    // Pass 1: one thread per vertex.
    VertexStageArgs vertex_args;
    vertex_args.view_projection = camera.view_projection(aspect);
    vertex_args.world_offset = rt.myrt_device_offset(world_dev);
    vertex_args.screen_offset = rt.myrt_device_offset(screen_dev);
    vertex_args.vertex_count = static_cast<uint32_t>(world.size());
    vertex_args.width = width;
    vertex_args.height = height;

    const SchedulerStats at_start = rt.stats();
    run_vertex_stage(rt, vertex_args);
    const SchedulerStats after_vertex = rt.stats();
    report("pass 1 vertex", at_start, after_vertex);

    // Pass 2: one thread per pixel. Reported separately, because pass 1 runs a
    // single warp with 29 of its lanes idle and would otherwise drown out the
    // figure worth reading.
    RasterStageArgs raster_args;
    raster_args.screen_offset = rt.myrt_device_offset(screen_dev);
    raster_args.framebuffer_offset = rt.myrt_device_offset(frame_dev);
    raster_args.width = width;
    raster_args.height = height;
    raster_args.triangle_count = triangles;

    run_raster_stage(rt, raster_args);
    report("pass 2 raster", after_vertex, rt.stats());
    std::printf("\n");
    rt.myrt_sync();

    std::vector<float> host_frame(pixels * PIXEL_FLOATS, 0.0f);
    rt.myrt_memcpy(host_frame.data(), frame_dev, frame_bytes, Direction::DeviceToHost);

    const std::string path = "output/raster.ppm";
    write_ppm(path, host_frame, width, height);
    std::printf("wrote %s\n", path.c_str());
    std::printf("compare against output/result.ppm from ray_triangle\n");
    return 0;
}
