#pragma once

#include <cstdint>
#include <vector>

#include "math3d.hpp"
#include "pipeline/draw.hpp"

// The scenes the benchmark programs share.
//
// render_bench measures these under the default cost model and model_bench puts
// the same ones to the others, so the two tables are read side by side in
// RESULTS.md. A scene builder copied into both would eventually differ in one of
// them and quietly make that pair meaningless — which is the failure the scenes
// were committed to code to avoid in the first place.

inline constexpr uint32_t BENCH_WIDTH = 64;
inline constexpr uint32_t BENCH_HEIGHT = 32;

// Matched to the tests, so a figure here and an assertion there describe the
// same picture.
inline DrawTarget bench_target()
{
    Camera cam;
    cam.eye = Float3{0.0f, 0.0f, 3.0f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    return DrawTarget{BENCH_WIDTH, BENCH_HEIGHT, cam};
}

inline void push_triangle(std::vector<Float3>& out, Float3 centre, float half)
{
    out.push_back(Float3{centre.x, centre.y + half, centre.z});
    out.push_back(Float3{centre.x - half, centre.y - half, centre.z});
    out.push_back(Float3{centre.x + half, centre.y - half, centre.z});
}

// An n x n grid of small triangles across the frame — the shape binning is for,
// since most tiles end up holding none at all.
//
// The 4 x 4 case reproduces the scene the tests use, which is the one row of the
// old tables that could be checked against anything.
inline std::vector<Float3> spread(uint32_t n)
{
    std::vector<Float3> world;
    for (uint32_t gy = 0; gy < n; ++gy) {
        for (uint32_t gx = 0; gx < n; ++gx) {
            const float t = (n == 1) ? 0.0f : static_cast<float>(gx) / (n - 1);
            const float u = (n == 1) ? 0.0f : static_cast<float>(gy) / (n - 1);
            push_triangle(world, Float3{-2.4f + 4.8f * t, -1.2f + 2.4f * u, 0.0f}, 0.25f);
        }
    }
    return world;
}

// Triangles piled at the centre, each a little further away.
//
// half is what decides whether binning has anything to remove: 0.5 reaches four
// of the eight tiles, 4.0 reaches all of them and leaves the binning nothing —
// which is the case it loses.
inline std::vector<Float3> stacked(uint32_t count, float half)
{
    std::vector<Float3> world;
    for (uint32_t i = 0; i < count; ++i) {
        push_triangle(world, Float3{0.0f, 0.0f, -0.01f * static_cast<float>(i)}, half);
    }
    return world;
}
