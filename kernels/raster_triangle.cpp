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

#include "pipeline/draw.hpp"
#include "pipeline/raster_tiled.hpp"  // TILE_WIDTH, the binning geometry
#include "pipeline/types.hpp"
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

// Issued work, which is reproducible; the GIOPS figure the runtime prints
// alongside it is not, depending as it does on how fast the host happens to be.
//
// Pass 2 only: each draw syncs between its passes, which clears the counters.
// Pass 1 puts three vertices in a 32-lane warp and reads 45% diverged from the
// 29 idle ones, which says nothing about the rasteriser.
void report(const char* label, const SchedulerStats& s)
{
    std::printf("%-14s %12llu %14llu %14llu %9.1f%%\n", label,
                static_cast<unsigned long long>(s.warp_steps),
                static_cast<unsigned long long>(s.active_lane_ops),
                static_cast<unsigned long long>(s.weighted_lane_ops),
                100.0 * s.divergence_rate());
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

    // Room for both framebuffers, the screen vertices, and the binned runs — a
    // triangle is copied into every tile it reaches, so the worst case is one
    // entry per tile per triangle.
    const size_t tiles = ((width + TILE_WIDTH - 1) / TILE_WIDTH) *
                         ((height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    const size_t budget = pixels * PIXEL_BYTES + world.size() * SCREEN_VERTEX_BYTES +
                          tiles * triangles * TILE_TRIANGLE_FLOATS * sizeof(float) +
                          (1u << 20);

    const DrawTarget target{width, height, scene_camera()};

    std::printf("rasterising %ux%u — %zu pixels, %u triangles\n\n", width, height, pixels,
                triangles);
    std::printf("%-14s %12s %14s %14s %10s\n", "", "warp steps", "lane ops", "weighted",
                "divergence");

    // A runtime each, so neither reading has to be a difference of two totals.
    MyGPURuntime walk_rt(budget);
    const std::vector<Float3> walk_frame = draw_walk(walk_rt, world, target);
    report("walk", walk_rt.stats());

    MyGPURuntime tiled_rt(budget);
    const std::vector<Float3> tiled_frame = draw_tiled(tiled_rt, world, target);
    report("tiled", tiled_rt.stats());

    const double saved =
        100.0 * (1.0 - static_cast<double>(tiled_rt.stats().weighted_lane_ops) /
                           static_cast<double>(walk_rt.stats().weighted_lane_ops));
    std::printf("\nbinning removed %.1f%% of the issued work\n", saved);

    // The claim the two routes make, checked rather than asserted in prose: the
    // tile a pixel belongs to changes which triangles it sees, never which
    // colour it ends up.
    size_t differing = 0;
    for (size_t i = 0; i < walk_frame.size(); ++i) {
        if (walk_frame[i].x != tiled_frame[i].x || walk_frame[i].y != tiled_frame[i].y ||
            walk_frame[i].z != tiled_frame[i].z) {
            ++differing;
        }
    }
    std::printf("frames agree: %s\n\n", (differing == 0) ? "yes" : "NO");

    // Written from the tiled frame, the two being identical — so the file on
    // disk is the optimised path's output and not merely claimed to match it.
    std::vector<float> flat;
    flat.reserve(pixels * PIXEL_FLOATS);
    for (const Float3& p : tiled_frame) {
        flat.push_back(p.x);
        flat.push_back(p.y);
        flat.push_back(p.z);
    }

    const std::string path = "output/raster.ppm";
    write_ppm(path, flat, width, height);
    std::printf("wrote %s\n", path.c_str());
    std::printf("compare against output/result.ppm from ray_triangle\n");
    return (differing == 0) ? 0 : 1;
}
