#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "math3d.hpp"
#include "pipeline.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// One pixel of slack. The device runs the same arithmetic in a different order,
// and pass 2 only ever compares these against pixel centres.
constexpr float PIXEL_EPS = 1e-2f;

Camera default_camera()
{
    Camera cam;
    cam.eye = Float3{0.0f, 0.0f, 3.0f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    return cam;
}

Float4x4 default_vp()
{
    return default_camera().view_projection(static_cast<float>(WIDTH) /
                                            static_cast<float>(HEIGHT));
}

// Uploads world vertices, runs pass 1, reads the screen vertices back.
std::vector<Float3> run_stage(const std::vector<Float3>& world, uint32_t thread_count,
                              float sentinel = -1.0f)
{
    MyGPURuntime rt(1u << 20);

    const size_t world_bytes = world.size() * WORLD_VERTEX_BYTES;
    const size_t screen_bytes = thread_count * SCREEN_VERTEX_BYTES;

    void* world_dev = rt.myrt_malloc(world_bytes);
    void* screen_dev = rt.myrt_malloc(screen_bytes);

    std::vector<float> flat;
    for (const Float3& v : world) {
        flat.push_back(v.x);
        flat.push_back(v.y);
        flat.push_back(v.z);
    }
    rt.myrt_memcpy(world_dev, flat.data(), world_bytes, Direction::HostToDevice);

    // A value no projection produces, so an untouched slot is recognisable.
    std::vector<float> initial(thread_count * SCREEN_VERTEX_FLOATS, sentinel);
    rt.myrt_memcpy(screen_dev, initial.data(), screen_bytes, Direction::HostToDevice);

    VertexStageArgs args;
    args.view_projection = default_vp();
    args.world_offset = rt.myrt_device_offset(world_dev);
    args.screen_offset = rt.myrt_device_offset(screen_dev);
    args.vertex_count = static_cast<uint32_t>(world.size());
    args.width = WIDTH;
    args.height = HEIGHT;

    run_vertex_stage(rt, args);

    std::vector<float> out(thread_count * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(out.data(), screen_dev, screen_bytes, Direction::DeviceToHost);

    std::vector<Float3> screen;
    for (uint32_t i = 0; i < thread_count; ++i) {
        screen.push_back(Float3{out[i * 3 + 0], out[i * 3 + 1], out[i * 3 + 2]});
    }
    return screen;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host reference — wrong here means wrong everywhere downstream
// ---------------------------------------------------------------------------

TEST(Pipeline, ProjectsTheCameraTargetToTheImageCentre)
{
    const Float3 screen = project_vertex(default_vp(), Float3{0, 0, 0}, WIDTH, HEIGHT);
    EXPECT_NEAR(screen.x, WIDTH * 0.5f, PIXEL_EPS);
    EXPECT_NEAR(screen.y, HEIGHT * 0.5f, PIXEL_EPS);
}

TEST(Pipeline, FlipsYBetweenNdcAndImageRows)
{
    // NDC counts upward from the bottom, an image counts downward from the top.
    // Getting this wrong renders a correct picture upside down, which is easy
    // to miss on a symmetric scene.
    const Float4x4 vp = default_vp();
    const Float3 high = project_vertex(vp, Float3{0.0f, 0.5f, 0.0f}, WIDTH, HEIGHT);
    const Float3 low = project_vertex(vp, Float3{0.0f, -0.5f, 0.0f}, WIDTH, HEIGHT);

    EXPECT_LT(high.y, low.y) << "a vertex higher in the world sits on an earlier row";
    EXPECT_NEAR(high.x, low.x, PIXEL_EPS) << "and neither moved sideways";
}

TEST(Pipeline, ForeshortensWithDistance)
{
    const Float4x4 vp = default_vp();
    const Float3 near_v = project_vertex(vp, Float3{1.0f, 0.0f, 0.0f}, WIDTH, HEIGHT);
    const Float3 far_v = project_vertex(vp, Float3{1.0f, 0.0f, -5.0f}, WIDTH, HEIGHT);

    const float near_dx = near_v.x - WIDTH * 0.5f;
    const float far_dx = far_v.x - WIDTH * 0.5f;
    EXPECT_LT(far_dx, near_dx) << "the same offset subtends less at greater depth";
    EXPECT_GT(far_dx, 0.0f) << "and stays on the same side of centre";
}

// ---------------------------------------------------------------------------
// Pass 1 as a program
// ---------------------------------------------------------------------------

TEST(Pipeline, VertexProgramFitsInTheRegisterFile)
{
    // The matrix alone claims 16, which makes this the widest kernel so far.
    // Bump allocation throws when it runs out, so this asserts the margin
    // rather than waiting to discover it.
    VertexStageArgs args;
    args.view_projection = default_vp();
    args.vertex_count = 3;
    args.width = WIDTH;
    args.height = HEIGHT;

    void* raw[] = {&args};
    const Program p = build_vertex_program(raw);

    EXPECT_FALSE(p.empty());
    EXPECT_EQ(p.back().op, Opcode::RET);

    uint32_t matvecs = 0;
    for (const Instruction& i : p) {
        if (i.op == Opcode::V_MATVEC_MAT4_F32) {
            ++matvecs;
        }
    }
    EXPECT_EQ(matvecs, 1u) << "one transform per vertex, not one per component";
}

TEST(Pipeline, VertexStageMatchesTheHostProjection)
{
    // The test the whole pass exists to pass: the device and the host agree on
    // where a vertex lands.
    const std::vector<Float3> world = {
        Float3{0.0f, 0.0f, 0.0f},
        Float3{0.5f, 0.5f, 0.0f},
        Float3{-0.5f, -0.5f, -1.0f},
    };

    const std::vector<Float3> got = run_stage(world, static_cast<uint32_t>(world.size()));
    ASSERT_EQ(got.size(), world.size());

    const Float4x4 vp = default_vp();
    for (size_t i = 0; i < world.size(); ++i) {
        const Float3 want = project_vertex(vp, world[i], WIDTH, HEIGHT);
        EXPECT_NEAR(got[i].x, want.x, PIXEL_EPS) << "vertex " << i << " x";
        EXPECT_NEAR(got[i].y, want.y, PIXEL_EPS) << "vertex " << i << " y";
        EXPECT_NEAR(got[i].z, want.z, PIXEL_EPS) << "vertex " << i << " depth";
    }
}

TEST(Pipeline, ThreadsPastTheVertexCountWriteNothing)
{
    // A launch rounds up to whole warps, so with 3 vertices 29 lanes have no
    // work. Without the guard they read past the buffer and write past it too.
    const std::vector<Float3> world = {
        Float3{0.0f, 0.0f, 0.0f},
        Float3{0.5f, 0.5f, 0.0f},
        Float3{-0.5f, -0.5f, -1.0f},
    };

    constexpr float SENTINEL = -1.0f;
    const std::vector<Float3> got = run_stage(world, 8, SENTINEL);
    ASSERT_EQ(got.size(), 8u);

    for (size_t i = world.size(); i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i].x, SENTINEL) << "slot " << i << " should be untouched";
        EXPECT_FLOAT_EQ(got[i].y, SENTINEL) << "slot " << i;
        EXPECT_FLOAT_EQ(got[i].z, SENTINEL) << "slot " << i;
    }
}

TEST(Pipeline, VertexStageRejectsAnEmptyMesh)
{
    MyGPURuntime rt(1u << 20);
    VertexStageArgs args;
    args.view_projection = default_vp();
    args.vertex_count = 0;
    args.width = WIDTH;
    args.height = HEIGHT;

    EXPECT_THROW(run_vertex_stage(rt, args), std::runtime_error);
}
