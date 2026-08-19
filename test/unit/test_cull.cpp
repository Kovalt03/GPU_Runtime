#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "bvh.hpp"
#include "math3d.hpp"
#include "pipeline/cull.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/vertex.hpp"
#include "runtime.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// A row of small boxes running far past the frustum on both sides, so a good
// fraction is outside whatever the camera is.
std::vector<Box> spread(uint32_t count, float step = 0.13f)
{
    std::vector<Box> boxes;
    for (uint32_t i = 0; i < count; ++i) {
        const float x =
            -static_cast<float>(count) * step * 0.5f + static_cast<float>(i) * step;
        const float y = (static_cast<float>(i % 5) - 2.0f) * 0.4f;
        boxes.push_back(
            Box{Float3{x - 0.1f, y - 0.1f, 0.0f}, Float3{x + 0.1f, y + 0.1f, 0.05f}});
    }
    return boxes;
}

Float4x4 translation(Float3 by)
{
    Float4x4 m = Float4x4::identity();
    m.at(0, 3) = by.x;
    m.at(1, 3) = by.y;
    m.at(2, 3) = by.z;
    return m;
}

// Everything a cull needs on the device, and what it wrote afterwards.
struct Culled {
    uint32_t survivors = 0;
    std::vector<float> matrices;
    uint64_t instructions = 0;
};

Culled cull_on_device(const Frustum& frustum, const std::vector<Box>& boxes)
{
    const uint32_t count = static_cast<uint32_t>(boxes.size());
    std::vector<float> box_floats;
    std::vector<float> matrices;
    for (const Box& box : boxes) {
        for (float v : {box.lo.x, box.lo.y, box.lo.z, box.hi.x, box.hi.y, box.hi.z}) {
            box_floats.push_back(v);
        }
        // A matrix whose translation names the box, so a survivor can be
        // identified by what was copied rather than by its slot.
        const Float4x4 m = translation(box.lo);
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                matrices.push_back(m.at(r, c));
            }
        }
    }

    MyGPURuntime rt(1u << 24);
    void* device_boxes = rt.myrt_malloc(box_floats.size() * sizeof(float));
    void* device_matrices = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* survivors = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    rt.myrt_memcpy(device_boxes, box_floats.data(), box_floats.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(device_matrices, matrices.data(), matrices.size() * sizeof(float),
                   Direction::HostToDevice);

    // grid.y starts at zero because it is the counter the cull raises.
    const float start[3] = {1.0f, 0.0f, 1.0f};
    rt.myrt_memcpy(grid, start, sizeof(start), Direction::HostToDevice);

    CullStageArgs args;
    args.frustum = frustum;
    args.boxes_offset = rt.myrt_device_offset(device_boxes);
    args.matrices_offset = rt.myrt_device_offset(device_matrices);
    args.survivors_offset = rt.myrt_device_offset(survivors);
    args.grid_offset = rt.myrt_device_offset(grid);
    args.instance_count = count;
    run_cull_stage(rt, args);
    rt.myrt_wait();

    float wrote[3] = {0.0f, 0.0f, 0.0f};
    rt.myrt_memcpy(wrote, grid, sizeof(wrote), Direction::DeviceToHost);

    Culled culled;
    culled.survivors = static_cast<uint32_t>(wrote[1]);
    culled.instructions = rt.stats().warp_steps;
    culled.matrices.resize(matrices.size());
    rt.myrt_memcpy(culled.matrices.data(), survivors,
                   culled.matrices.size() * sizeof(float), Direction::DeviceToHost);
    return culled;
}

}  // namespace

TEST(Cull, TheDeviceKeepsWhatTheHostWouldHave)
{
    // The count on its own says nothing: a cull that kept an arbitrary 40 of 64
    // would report the same number as one that kept the right 40. The set is
    // compared, and the host test is written separately from the kernel so that
    // agreeing means something.
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const Frustum frustum = frustum_of(target.camera.view_projection(target.aspect()));
    const std::vector<Box> boxes = spread(64);

    uint32_t expected = 0;
    std::vector<Float3> should_survive;
    for (const Box& box : boxes) {
        if (!outside_frustum(frustum, box)) {
            ++expected;
            should_survive.push_back(box.lo);
        }
    }
    ASSERT_GT(expected, 0u) << "the camera saw nothing, so nothing was compared";
    ASSERT_LT(expected, boxes.size()) << "the camera saw everything, so nothing was cut";

    const Culled culled = cull_on_device(frustum, boxes);
    EXPECT_EQ(culled.survivors, expected);

    // Compacted, so the first `survivors` matrices are the ones kept — in
    // whatever order the atomic handed out slots, which is why this compares
    // sets rather than sequences.
    std::vector<Float3> kept;
    for (uint32_t i = 0; i < culled.survivors; ++i) {
        const float* m = &culled.matrices[i * 16];
        kept.push_back(Float3{m[3], m[7], m[11]});
    }
    ASSERT_EQ(kept.size(), should_survive.size());

    for (const Float3& wanted : should_survive) {
        bool found = false;
        for (const Float3& got : kept) {
            found = found || (std::fabs(got.x - wanted.x) < 1e-5f &&
                              std::fabs(got.y - wanted.y) < 1e-5f);
        }
        EXPECT_TRUE(found) << "a box the host kept is not in the compacted list";
    }
}

TEST(Cull, AFrustumHoldsWhatTheCameraCanSee)
{
    // Extracted from the matrix rather than built from the camera, so this is
    // what says the extraction is right: a point at the eye is behind the near
    // plane, one in front of it is inside, and one far off to the side is not.
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const Frustum frustum = frustum_of(target.camera.view_projection(target.aspect()));

    const auto tiny = [](Float3 at) {
        return Box{Float3{at.x - 0.01f, at.y - 0.01f, at.z - 0.01f},
                   Float3{at.x + 0.01f, at.y + 0.01f, at.z + 0.01f}};
    };

    EXPECT_FALSE(outside_frustum(frustum, tiny(Float3{0.0f, 0.0f, 0.0f})));
    EXPECT_TRUE(outside_frustum(frustum, tiny(Float3{0.0f, 0.0f, 10.0f})))
        << "a point behind the camera survived";
    EXPECT_TRUE(outside_frustum(frustum, tiny(Float3{40.0f, 0.0f, 0.0f})))
        << "a point far to the side survived";

    // A box straddling a plane is kept: the test is whether it is wholly
    // outside, and half of one is not.
    EXPECT_FALSE(outside_frustum(
        frustum, Box{Float3{-40.0f, -0.1f, -0.1f}, Float3{0.1f, 0.1f, 0.1f}}));
}

TEST(Cull, TheGridTheCullWroteIsTheGridTheDrawRuns)
{
    // The point of the whole arrangement: pass 1 draws as many instances as
    // survived, and the number reaches it without passing through the host.
    // Here it is read back afterwards only to check, which is exactly what the
    // arrangement exists to avoid having to do.
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const Float4x4 view_projection = target.camera.view_projection(target.aspect());
    const Frustum frustum = frustum_of(view_projection);
    const std::vector<Box> boxes = spread(64);

    uint32_t expected = 0;
    for (const Box& box : boxes) {
        expected += outside_frustum(frustum, box) ? 0u : 1u;
    }

    const std::vector<Float3> triangle{
        Float3{-0.1f, -0.1f, 0.0f}, Float3{0.1f, -0.1f, 0.0f}, Float3{0.0f, 0.1f, 0.05f}};

    std::vector<float> box_floats;
    std::vector<float> matrices;
    for (const Box& box : boxes) {
        for (float v : {box.lo.x, box.lo.y, box.lo.z, box.hi.x, box.hi.y, box.hi.z}) {
            box_floats.push_back(v);
        }
        const Float4x4 m = translation(Float3{box.lo.x + 0.1f, box.lo.y + 0.1f, 0.0f});
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                matrices.push_back(m.at(r, c));
            }
        }
    }

    MyGPURuntime rt(1u << 26);
    void* device_boxes = rt.myrt_malloc(box_floats.size() * sizeof(float));
    void* device_matrices = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* survivors = rt.myrt_malloc(matrices.size() * sizeof(float));
    void* grid = rt.myrt_malloc(3 * sizeof(float));
    void* world = rt.myrt_malloc(triangle.size() * sizeof(Float3));
    void* screen =
        rt.myrt_malloc(instanced_screen_bytes(3, static_cast<uint32_t>(boxes.size())));
    rt.myrt_memcpy(device_boxes, box_floats.data(), box_floats.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(device_matrices, matrices.data(), matrices.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(world, triangle.data(), triangle.size() * sizeof(Float3),
                   Direction::HostToDevice);
    const float start[3] = {1.0f, 0.0f, 1.0f};
    rt.myrt_memcpy(grid, start, sizeof(start), Direction::HostToDevice);

    CullStageArgs cull;
    cull.frustum = frustum;
    cull.boxes_offset = rt.myrt_device_offset(device_boxes);
    cull.matrices_offset = rt.myrt_device_offset(device_matrices);
    cull.survivors_offset = rt.myrt_device_offset(survivors);
    cull.grid_offset = rt.myrt_device_offset(grid);
    cull.instance_count = static_cast<uint32_t>(boxes.size());
    run_cull_stage(rt, cull);

    VertexStageArgs pass1;
    pass1.view_projection = view_projection;
    pass1.world_offset = rt.myrt_device_offset(world);
    pass1.screen_offset = rt.myrt_device_offset(screen);
    pass1.vertex_count = 3;
    pass1.width = WIDTH;
    pass1.height = HEIGHT;
    pass1.instance_count = static_cast<uint32_t>(boxes.size());
    pass1.instance_offset = rt.myrt_device_offset(survivors);
    pass1.uniform_offset = pass1.instance_offset;

    // Enqueued behind the cull on the same stream, so the grid it reads is the
    // one the cull has raised by the time it runs.
    run_vertex_stage_indirect(rt, pass1, cull.grid_offset);
    rt.myrt_wait();

    // 1/w is nonzero for every vertex pass 1 touched and zero for the rest, the
    // buffer never having been written.
    std::vector<float> written(3ull * boxes.size() * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(written.data(), screen, written.size() * sizeof(float),
                   Direction::DeviceToHost);
    uint32_t drawn = 0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        drawn += written[i * 3 * SCREEN_VERTEX_FLOATS + 3] != 0.0f ? 1u : 0u;
    }
    EXPECT_EQ(drawn, expected);
}

TEST(Cull, ACullWithNowhereToPutItsAnswerIsRefused)
{
    MyGPURuntime rt(1u << 20);
    CullStageArgs args;
    args.instance_count = 0;
    EXPECT_THROW(run_cull_stage(rt, args), std::runtime_error);

    args.instance_count = 4;
    EXPECT_THROW(run_cull_stage(rt, args), std::runtime_error);

    // And a draw asked to take its grid from nowhere.
    VertexStageArgs pass1;
    pass1.vertex_count = 3;
    pass1.instance_offset = 64;
    EXPECT_THROW(run_vertex_stage_indirect(rt, pass1, 0), std::runtime_error);

    pass1.instance_offset = 0;
    EXPECT_THROW(run_vertex_stage_indirect(rt, pass1, 128), std::runtime_error);
}
