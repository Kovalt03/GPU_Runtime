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

#include "pipeline.hpp"
#include "ppm.hpp"
#include "runtime.hpp"

namespace {

// Identical to the ray tracer's Scene: apex up, base along the bottom, at
// z = -2 and large enough that its edges cross plenty of warps.
const std::vector<Float3> TRIANGLE = {
    Float3{0.0f, 0.5f, -2.0f},    // apex
    Float3{-0.5f, -0.5f, -2.0f},  // bottom left
    Float3{0.5f, -0.5f, -2.0f},   // bottom right
};

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

    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t world_bytes = TRIANGLE.size() * WORLD_VERTEX_BYTES;
    const size_t screen_bytes = TRIANGLE.size() * SCREEN_VERTEX_BYTES;
    const size_t frame_bytes = pixels * PIXEL_BYTES;

    MyGPURuntime rt(frame_bytes + screen_bytes + (1u << 20));

    // Three separate allocations, which is why the offsets are asked for rather
    // than assumed: only the first of them sits at zero.
    void* world_dev = rt.myrt_malloc(world_bytes);
    void* screen_dev = rt.myrt_malloc(screen_bytes);
    void* frame_dev = rt.myrt_malloc(frame_bytes);

    std::vector<float> world_flat;
    for (const Float3& v : TRIANGLE) {
        world_flat.push_back(v.x);
        world_flat.push_back(v.y);
        world_flat.push_back(v.z);
    }
    rt.myrt_memcpy(world_dev, world_flat.data(), world_bytes, Direction::HostToDevice);

    const Camera camera = scene_camera();
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    std::printf("rasterising %ux%u — %zu pixels, %zu triangles\n", width, height, pixels,
                TRIANGLE.size() / 3);

    // Pass 1: one thread per vertex.
    VertexStageArgs vertex_args;
    vertex_args.view_projection = camera.view_projection(aspect);
    vertex_args.world_offset = rt.myrt_device_offset(world_dev);
    vertex_args.screen_offset = rt.myrt_device_offset(screen_dev);
    vertex_args.vertex_count = static_cast<uint32_t>(TRIANGLE.size());
    vertex_args.width = width;
    vertex_args.height = height;

    run_vertex_stage(rt, vertex_args);
    std::printf("pass 1 (vertex)  ");
    rt.myrt_sync();

    // Pass 2: one thread per pixel. Reported separately, because pass 1 runs a
    // single warp with 29 of its lanes idle and would otherwise drown out the
    // figure worth reading.
    RasterStageArgs raster_args;
    raster_args.screen_offset = rt.myrt_device_offset(screen_dev);
    raster_args.framebuffer_offset = rt.myrt_device_offset(frame_dev);
    raster_args.width = width;
    raster_args.height = height;
    raster_args.triangle_index = 0;

    run_raster_stage(rt, raster_args);
    std::printf("pass 2 (raster)  ");
    rt.myrt_sync();

    std::vector<float> host_frame(pixels * PIXEL_FLOATS, 0.0f);
    rt.myrt_memcpy(host_frame.data(), frame_dev, frame_bytes, Direction::DeviceToHost);

    const std::string path = "output/raster.ppm";
    write_ppm(path, host_frame, width, height);
    std::printf("wrote %s\n", path.c_str());
    std::printf("compare against output/result.ppm from ray_triangle\n");
    return 0;
}
