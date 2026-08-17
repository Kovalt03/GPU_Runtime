#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "math3d.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/raster_tiled.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/types.hpp"
#include "pipeline/vertex.hpp"
#include "reference.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// One pixel of slack. The device runs the same arithmetic in a different order,
// and pass 2 only ever compares these against pixel centres.
constexpr float PIXEL_EPS = 1e-2f;

void expect_near(Float3 got, Float3 want, float eps, const char* what)
{
    EXPECT_NEAR(got.x, want.x, eps) << what << " .x";
    EXPECT_NEAR(got.y, want.y, eps) << what << " .y";
    EXPECT_NEAR(got.z, want.z, eps) << what << " .z";
}

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
// external lets a caller read the counters. run_stage does not sync, so what
// the runtime holds afterwards is pass 1 alone — which is the only way to see
// it, draw_* clearing the counters between its passes on purpose.
std::vector<Float3> run_stage(const std::vector<Float3>& world, uint32_t thread_count,
                              float sentinel = -1.0f, MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 20);
    MyGPURuntime& rt = (external != nullptr) ? *external : local;

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
        const size_t at = static_cast<size_t>(i) * SCREEN_VERTEX_FLOATS;
        screen.push_back(Float3{out[at + 0], out[at + 1], out[at + 2]});
    }
    return screen;
}

// The same, as whole triangles — which is what the raster path consumes, and
// the only form that carries the reciprocals of w. A reference built without
// them is affine, and would agree with the kernel only on a scene whose
// triangles each sit at one depth.
std::vector<ScreenTriangle> project_all_triangles(const std::vector<Float3>& world)
{
    const Float4x4 vp = default_vp();
    std::vector<ScreenTriangle> out;
    for (size_t i = 0; i + 2 < world.size(); i += 3) {
        out.push_back(
            project_triangle(vp, world[i], world[i + 1], world[i + 2], WIDTH, HEIGHT));
    }
    return out;
}

// The three routes, each on its own runtime unless the caller wants to read the
// counters. The bodies live in pipeline/draw.cpp so that what these tests check
// and what benchmarks/raster_bench measures cannot come apart.

DrawTarget default_target(uint32_t width = WIDTH, uint32_t height = HEIGHT)
{
    return DrawTarget{width, height, default_camera()};
}

std::vector<Float3> render_triangle(const std::vector<Float3>& world,
                                    MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 22);
    return draw_walk((external != nullptr) ? *external : local, world, default_target());
}

std::vector<Float3> render_triangle_tiled(const std::vector<Float3>& world,
                                          MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 24);
    return draw_tiled((external != nullptr) ? *external : local, world, default_target());
}

// The frame size is a parameter because the interesting case for the barrier is
// a frame that does not divide evenly into tiles: the edge blocks then hold
// threads whose pixel is off screen, and those threads still have to reach it.
std::vector<Float3> render_triangle_shared(const std::vector<Float3>& world,
                                           uint32_t width = WIDTH,
                                           uint32_t height = HEIGHT,
                                           MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 24);
    return draw_shared((external != nullptr) ? *external : local, world,
                       default_target(width, height));
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

// ---------------------------------------------------------------------------
// Pass 2 — host reference
//
// Coverage stacks three sign conventions: the triangle's winding, the y flip
// pass 1 applied, and where inside a pixel the sample sits. Each of them wrong
// still renders something, so each is pinned separately.
// ---------------------------------------------------------------------------

namespace {

// A triangle already in screen space, so these tests need no camera. Wound
// counter-clockwise on paper; after the y flip it is clockwise on screen, which
// is exactly the case the winding-independent coverage test has to survive.
constexpr Float3 T0{10.0f, 10.0f, 0.0f};
constexpr Float3 T1{50.0f, 10.0f, 0.0f};
constexpr Float3 T2{10.0f, 26.0f, 0.0f};

// Well inside, by inspection: a third of the way along both short legs.
constexpr float INSIDE_X = 20.0f;
constexpr float INSIDE_Y = 14.0f;

}  // namespace

TEST(Pipeline, EdgeFunctionChangesSignAcrossItsLine)
{
    // Points on the line itself give zero, which is what puts a pixel exactly
    // on an edge into one triangle rather than neither or both.
    EXPECT_NEAR(edge_function(T0, T1, T0.x, T0.y), 0.0f, PIXEL_EPS);
    EXPECT_NEAR(edge_function(T0, T1, T1.x, T1.y), 0.0f, PIXEL_EPS);

    const float below = edge_function(T0, T1, 30.0f, 0.0f);
    const float above = edge_function(T0, T1, 30.0f, 20.0f);
    EXPECT_LT(below * above, 0.0f) << "opposite sides give opposite signs";
}

TEST(Pipeline, BarycentricWeightsSumToOne)
{
    const Float3 w = barycentric(T0, T1, T2, INSIDE_X, INSIDE_Y);
    EXPECT_NEAR(w.x + w.y + w.z, 1.0f, PIXEL_EPS);
}

TEST(Pipeline, BarycentricIsOneAtItsOwnVertex)
{
    // Which weight belongs to which vertex is the ordering shade_pixel depends
    // on to colour v0 blue and v1 red.
    const Float3 at0 = barycentric(T0, T1, T2, T0.x, T0.y);
    EXPECT_NEAR(at0.x, 1.0f, PIXEL_EPS);
    EXPECT_NEAR(at0.y, 0.0f, PIXEL_EPS);
    EXPECT_NEAR(at0.z, 0.0f, PIXEL_EPS);

    const Float3 at2 = barycentric(T0, T1, T2, T2.x, T2.y);
    EXPECT_NEAR(at2.z, 1.0f, PIXEL_EPS);
}

TEST(Pipeline, CoverageSurvivesAWindingFlip)
{
    // Swapping two vertices reverses the winding and negates every edge value.
    // Normalising by the signed area is what keeps "all three >= 0" meaning the
    // same thing — and pass 1 flips the winding of every triangle it projects.
    const Float3 forward = barycentric(T0, T1, T2, INSIDE_X, INSIDE_Y);
    const Float3 reversed = barycentric(T0, T2, T1, INSIDE_X, INSIDE_Y);

    EXPECT_GT(forward.x, 0.0f);
    EXPECT_GT(forward.y, 0.0f);
    EXPECT_GT(forward.z, 0.0f);

    EXPECT_GT(reversed.x, 0.0f) << "the same point is inside either way round";
    EXPECT_GT(reversed.y, 0.0f);
    EXPECT_GT(reversed.z, 0.0f);
}

TEST(Pipeline, ShadePixelMakesEachVertexAPrimary)
{
    // v0 blue, v1 red, v2 green — the arrangement kernels/ray_triangle.cpp
    // already produces, so the two renderers can be compared by eye as well as
    // by number.
    // Each sample sits just inside its own corner. The triangle has its right
    // angle at T0, and the hypotenuse runs y = 30 - 0.4x, so a point has to
    // clear that as well as the two legs — 49,10 does not.
    const Float3 blue = shade_pixel(T0, T1, T2, 10, 10);
    EXPECT_GT(blue.z, blue.x);
    EXPECT_GT(blue.z, blue.y);

    const Float3 red = shade_pixel(T0, T1, T2, 44, 11);
    EXPECT_GT(red.x, red.y);
    EXPECT_GT(red.x, red.z);

    const Float3 green = shade_pixel(T0, T1, T2, 11, 23);
    EXPECT_GT(green.y, green.x);
    EXPECT_GT(green.y, green.z);
}

TEST(Pipeline, ShadePixelIsBlackOutsideTheTriangle)
{
    const Float3 outside = shade_pixel(T0, T1, T2, 60, 30);
    EXPECT_FLOAT_EQ(outside.x, 0.0f);
    EXPECT_FLOAT_EQ(outside.y, 0.0f);
    EXPECT_FLOAT_EQ(outside.z, 0.0f);
}

TEST(Pipeline, ShadePixelSamplesTheCentreNotTheCorner)
{
    // The ray tracer samples (px + 0.5, py + 0.5). A rasteriser sampling the
    // corner instead lands half a pixel off, which shows up as a fringe along
    // every edge rather than as an obviously wrong image.
    //
    // Pixel (9, 10) has its corner exactly on the T0 corner and its centre
    // outside, so the two conventions disagree here.
    const Float3 got = shade_pixel(T0, T1, T2, 9, 10);
    EXPECT_FLOAT_EQ(got.x, 0.0f)
        << "the centre of pixel 9 is at x = 9.5, left of the edge";
}

// ---------------------------------------------------------------------------
// Pass 2 as a program
// ---------------------------------------------------------------------------

TEST(Pipeline, RasterProgramCarriesNoMatrix)
{
    // The point of splitting the pipeline: coverage never pays the sixteen
    // registers a uniform matrix costs, so this kernel is far narrower than
    // pass 1 even though it does more branching.
    RasterStageArgs args;
    args.width = WIDTH;
    args.height = HEIGHT;

    void* raw[] = {&args};
    const Program p = build_raster_program(raw);

    ASSERT_FALSE(p.empty());
    EXPECT_EQ(p.back().op, Opcode::RET);
    for (const Instruction& i : p) {
        EXPECT_NE(i.op, Opcode::V_MATVEC_MAT4_F32);
    }
}

TEST(Pipeline, PerspectiveCorrectionIsIdentityAtOneDepth)
{
    // Every vertex at the same distance leaves the weights alone, which is why
    // the two renderers agreed on a flat scene before any of this existed — and
    // why such a scene proves nothing about interpolation.
    const Float3 affine{0.2f, 0.3f, 0.5f};
    const Float3 got = perspective_correct(affine, 0.25f, 0.25f, 0.25f);

    EXPECT_NEAR(got.x, affine.x, PIXEL_EPS);
    EXPECT_NEAR(got.y, affine.y, PIXEL_EPS);
    EXPECT_NEAR(got.z, affine.z, PIXEL_EPS);
}

TEST(Pipeline, PerspectiveCorrectionPullsTowardsTheNearerVertex)
{
    // A vertex twice as close has twice the 1/w, so it takes more of the weight
    // than the screen-space split suggests. Halfway along an edge in pixels is
    // not halfway along it in the world.
    const Float3 affine{0.5f, 0.5f, 0.0f};
    const Float3 got = perspective_correct(affine, 1.0f, 0.5f, 0.5f);

    EXPECT_NEAR(got.x + got.y + got.z, 1.0f, PIXEL_EPS) << "still a partition";
    EXPECT_GT(got.x, affine.x) << "the nearer vertex gains";
    EXPECT_LT(got.y, affine.y);
}

TEST(Pipeline, RasterStageMatchesTheHostShading)
{
    // Both passes end to end: project a world triangle, rasterise it, and check
    // every pixel against what the host says it should have been.
    const std::vector<Float3> world = {
        Float3{0.0f, 0.5f, 0.0f},
        Float3{-0.5f, -0.5f, 0.0f},
        Float3{0.5f, -0.5f, 0.0f},
    };

    const std::vector<Float3> frame = render_triangle(world);
    ASSERT_EQ(frame.size(), static_cast<size_t>(WIDTH) * HEIGHT);

    const ScreenTriangle t = project_all_triangles(world)[0];
    uint32_t covered = 0;
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            const Float3 want =
                shade_pixel(t.v0, t.v1, t.v2, x, y, t.inv_w0, t.inv_w1, t.inv_w2);
            const Float3 got = frame[y * WIDTH + x];
            ASSERT_NEAR(got.x, want.x, PIXEL_EPS) << "pixel " << x << "," << y << " r";
            ASSERT_NEAR(got.y, want.y, PIXEL_EPS) << "pixel " << x << "," << y << " g";
            ASSERT_NEAR(got.z, want.z, PIXEL_EPS) << "pixel " << x << "," << y << " b";
            if (want.x + want.y + want.z > 0.0f) {
                ++covered;
            }
        }
    }
    EXPECT_GT(covered, 0u) << "a frame of black would satisfy every check above";
}

TEST(Pipeline, RasterStageDivergesOnlyAtTheEdges)
{
    // The measurement the project exists for. A triangle covering part of the
    // frame splits the warps its edges cross and no others, so the rate is
    // above zero and nowhere near one.
    const std::vector<Float3> world = {
        Float3{0.0f, 0.5f, 0.0f},
        Float3{-0.5f, -0.5f, 0.0f},
        Float3{0.5f, -0.5f, 0.0f},
    };

    MyGPURuntime rt(1u << 22);
    render_triangle(world, &rt);

    EXPECT_GT(rt.divergence_rate(), 0.0) << "an edge has to split some warp";
    EXPECT_LT(rt.divergence_rate(), 1.0);
}

TEST(Pipeline, RasterStageRejectsAnEmptyImage)
{
    MyGPURuntime rt(1u << 20);
    RasterStageArgs args;
    args.width = 0;
    args.height = HEIGHT;

    EXPECT_THROW(run_raster_stage(rt, args), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Several triangles — depth, and which one wins
// ---------------------------------------------------------------------------

namespace {

// Two overlapping triangles at fixed depths, already in screen space. The
// second is nearer, so wherever they overlap it is the one that shows.
ScreenTriangle far_triangle()
{
    return ScreenTriangle{Float3{5.0f, 5.0f, 0.5f}, Float3{40.0f, 5.0f, 0.5f},
                          Float3{5.0f, 25.0f, 0.5f}};
}

ScreenTriangle near_triangle()
{
    return ScreenTriangle{Float3{15.0f, 5.0f, -0.5f}, Float3{50.0f, 5.0f, -0.5f},
                          Float3{15.0f, 25.0f, -0.5f}};
}

}  // namespace

TEST(Pipeline, DepthInterpolatesLinearlyAcrossTheTriangle)
{
    // Exact in screen space because pass 1 already divided z by w. Anything
    // else carried across a projected triangle would need the divide undone.
    const Float3 v0{0.0f, 0.0f, -1.0f};
    const Float3 v1{10.0f, 0.0f, 0.0f};
    const Float3 v2{0.0f, 10.0f, 1.0f};

    EXPECT_NEAR(interpolate_depth(v0, v1, v2, Float3{1.0f, 0.0f, 0.0f}), -1.0f,
                PIXEL_EPS);
    EXPECT_NEAR(interpolate_depth(v0, v1, v2, Float3{0.0f, 0.0f, 1.0f}), 1.0f, PIXEL_EPS);
    EXPECT_NEAR(interpolate_depth(v0, v1, v2, Float3{0.5f, 0.5f, 0.0f}), -0.5f,
                PIXEL_EPS);
}

TEST(Pipeline, NearestTriangleWinsWhereTheyOverlap)
{
    const std::vector<ScreenTriangle> both = {far_triangle(), near_triangle()};

    // Inside both: the nearer one is drawn whichever order the buffer holds.
    const Float3 got = shade_pixel_nearest(both, 20, 10);
    const Float3 want =
        shade_pixel(near_triangle().v0, near_triangle().v1, near_triangle().v2, 20, 10);
    EXPECT_NEAR(got.x, want.x, PIXEL_EPS);
    EXPECT_NEAR(got.y, want.y, PIXEL_EPS);
    EXPECT_NEAR(got.z, want.z, PIXEL_EPS);

    const std::vector<ScreenTriangle> reversed = {near_triangle(), far_triangle()};
    const Float3 other = shade_pixel_nearest(reversed, 20, 10);
    EXPECT_NEAR(other.x, got.x, PIXEL_EPS) << "buffer order must not decide it";
}

TEST(Pipeline, EachTriangleShowsWhereOnlyItCovers)
{
    const std::vector<ScreenTriangle> both = {far_triangle(), near_triangle()};

    // Left of the near triangle, inside the far one only.
    const Float3 only_far = shade_pixel_nearest(both, 7, 10);
    EXPECT_GT(only_far.x + only_far.y + only_far.z, 0.0f);

    // Right of the far one, inside the near only.
    const Float3 only_near = shade_pixel_nearest(both, 45, 7);
    EXPECT_GT(only_near.x + only_near.y + only_near.z, 0.0f);
}

TEST(Pipeline, NoTriangleLeavesTheBackground)
{
    const std::vector<ScreenTriangle> both = {far_triangle(), near_triangle()};
    const Float3 outside = shade_pixel_nearest(both, 60, 30);
    EXPECT_FLOAT_EQ(outside.x, 0.0f);
    EXPECT_FLOAT_EQ(outside.y, 0.0f);
    EXPECT_FLOAT_EQ(outside.z, 0.0f);

    EXPECT_FLOAT_EQ(shade_pixel_nearest({}, 20, 10).x, 0.0f) << "an empty buffer too";
}

TEST(Pipeline, RasterStageDrawsTwoTrianglesInDepthOrder)
{
    // End to end, and the check DOC/09 asks for: two triangles that occlude
    // each other have to come out in the right order.
    //
    // The nearer one is at z = 0, the farther at z = -1 in world space, with
    // the camera down +z looking at the origin.
    const std::vector<Float3> world = {
        Float3{-0.6f, 0.4f, -1.0f}, Float3{-0.6f, -0.4f, -1.0f},
        Float3{0.4f, 0.4f, -1.0f},

        Float3{-0.4f, 0.4f, 0.0f},  Float3{-0.4f, -0.4f, 0.0f},
        Float3{0.6f, 0.4f, 0.0f},
    };

    const std::vector<Float3> frame = render_triangle(world);
    ASSERT_EQ(frame.size(), static_cast<size_t>(WIDTH) * HEIGHT);

    const std::vector<ScreenTriangle> triangles = project_all_triangles(world);

    uint32_t covered = 0;
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            const Float3 want = shade_pixel_nearest(triangles, x, y);
            const Float3 got = frame[y * WIDTH + x];
            ASSERT_NEAR(got.x, want.x, PIXEL_EPS) << "pixel " << x << "," << y << " r";
            ASSERT_NEAR(got.y, want.y, PIXEL_EPS) << "pixel " << x << "," << y << " g";
            ASSERT_NEAR(got.z, want.z, PIXEL_EPS) << "pixel " << x << "," << y << " b";
            if (want.x + want.y + want.z > 0.0f) {
                ++covered;
            }
        }
    }
    EXPECT_GT(covered, 0u) << "a frame of black would satisfy every check above";
}

// ---------------------------------------------------------------------------
// Tiling — the same frame from a shorter list
// ---------------------------------------------------------------------------

namespace {

// Two triangles far enough apart that they land in different tiles, which is
// what makes the binning observable at all.
std::vector<ScreenTriangle> spread_triangles()
{
    return {
        ScreenTriangle{Float3{2.0f, 2.0f, 0.0f}, Float3{20.0f, 2.0f, 0.0f},
                       Float3{2.0f, 6.0f, 0.0f}},
        ScreenTriangle{Float3{40.0f, 20.0f, 0.0f}, Float3{60.0f, 20.0f, 0.0f},
                       Float3{40.0f, 28.0f, 0.0f}},
    };
}

uint32_t tile_triangle_count(const TileBinning& binning, uint32_t tx, uint32_t ty)
{
    return static_cast<uint32_t>(binning.table[(ty * binning.tiles_x + tx) * 2 + 1]);
}

}  // namespace

TEST(Pipeline, BinningCoversTheWholeScreen)
{
    const TileBinning binning = bin_triangles(spread_triangles(), WIDTH, HEIGHT);

    // Rounded up, so the last tile hangs off the edge and its threads fall out
    // on the bounds check the kernel already has.
    EXPECT_EQ(binning.tiles_x, (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);
    EXPECT_EQ(binning.tiles_y, (HEIGHT + TILE_HEIGHT - 1) / TILE_HEIGHT);

    // One (first, count) pair per tile.
    EXPECT_EQ(binning.table.size(), binning.tile_count() * 2);
}

TEST(Pipeline, BinningPutsEachTriangleWhereItReaches)
{
    const TileBinning binning = bin_triangles(spread_triangles(), WIDTH, HEIGHT);

    // The first triangle spans x 2..20, y 2..6 — tile (0, 0) only.
    EXPECT_GT(tile_triangle_count(binning, 0, 0), 0u);

    // The second spans x 40..60, y 20..28 — tile (1, 2) and (1, 3).
    EXPECT_GT(tile_triangle_count(binning, 1, 2), 0u);

    // And nothing reaches the top right.
    EXPECT_EQ(tile_triangle_count(binning, 1, 0), 0u);
}

TEST(Pipeline, BinningStoresOneScreenTrianglePerEntry)
{
    const TileBinning binning = bin_triangles(spread_triangles(), WIDTH, HEIGHT);

    uint32_t entries = 0;
    for (uint32_t t = 0; t < binning.tile_count(); ++t) {
        entries += static_cast<uint32_t>(binning.table[t * 2 + 1]);
    }
    EXPECT_EQ(binning.vertices.size(), entries * TILE_TRIANGLE_FLOATS);
    EXPECT_GE(entries, 2u) << "each triangle reaches at least one tile";
}

TEST(Pipeline, BinningDropsATriangleThatIsWhollyOffScreen)
{
    const std::vector<ScreenTriangle> away = {
        ScreenTriangle{Float3{-90.0f, -90.0f, 0.0f}, Float3{-70.0f, -90.0f, 0.0f},
                       Float3{-90.0f, -70.0f, 0.0f}},
    };
    const TileBinning binning = bin_triangles(away, WIDTH, HEIGHT);
    EXPECT_TRUE(binning.vertices.empty());
}

TEST(Pipeline, TiledRasterDrawsTheSameFrameAsTheWalk)
{
    // The point of the whole change: fewer triangles visited, identical output.
    const std::vector<Float3> world = {
        Float3{-0.6f, 0.4f, -1.0f}, Float3{-0.6f, -0.4f, -1.0f},
        Float3{0.4f, 0.4f, -1.0f},

        Float3{-0.4f, 0.4f, 0.0f},  Float3{-0.4f, -0.4f, 0.0f},
        Float3{0.6f, 0.4f, 0.0f},
    };

    const std::vector<Float3> walked = render_triangle(world);
    const std::vector<Float3> tiled = render_triangle_tiled(world);
    ASSERT_EQ(tiled.size(), walked.size());

    for (size_t i = 0; i < walked.size(); ++i) {
        ASSERT_NEAR(tiled[i].x, walked[i].x, PIXEL_EPS) << "pixel " << i << " r";
        ASSERT_NEAR(tiled[i].y, walked[i].y, PIXEL_EPS) << "pixel " << i << " g";
        ASSERT_NEAR(tiled[i].z, walked[i].z, PIXEL_EPS) << "pixel " << i << " b";
    }
}

TEST(Pipeline, TiledRasterIssuesLessWorkThanTheWalk)
{
    // The measurement, asserted so a regression is a failing test rather than a
    // number nobody re-reads.
    //
    // Sixteen small triangles spread over the frame, which is the shape binning
    // is for: each tile ends up with two of them instead of all sixteen.
    //
    // Triangles large enough to cover the whole frame would land in every tile
    // and leave the binning nothing to remove, at which point it costs 1.4%
    // rather than saving anything. benchmarks/RESULTS.md records both ends.
    std::vector<Float3> world;
    for (uint32_t gy = 0; gy < 4; ++gy) {
        for (uint32_t gx = 0; gx < 4; ++gx) {
            const float cx = -2.4f + 1.6f * static_cast<float>(gx);
            const float cy = -1.2f + 0.8f * static_cast<float>(gy);
            world.push_back(Float3{cx, cy + 0.25f, 0.0f});
            world.push_back(Float3{cx - 0.25f, cy - 0.25f, 0.0f});
            world.push_back(Float3{cx + 0.25f, cy - 0.25f, 0.0f});
        }
    }

    MyGPURuntime walked_rt(1u << 24);
    render_triangle(world, &walked_rt);
    const uint64_t walked = walked_rt.stats().weighted_lane_ops;

    MyGPURuntime tiled_rt(1u << 24);
    render_triangle_tiled(world, &tiled_rt);
    const uint64_t tiled = tiled_rt.stats().weighted_lane_ops;

    EXPECT_LT(tiled, walked) << "binning has to save something";
    EXPECT_LT(tiled * 2, walked) << "and enough to be worth the change";
}

// ---------------------------------------------------------------------------
// Tiling through shared memory
// ---------------------------------------------------------------------------

namespace {

// Sixteen small triangles spread over the frame, the shape binning is for.
std::vector<Float3> scattered_scene()
{
    std::vector<Float3> world;
    for (uint32_t gy = 0; gy < 4; ++gy) {
        for (uint32_t gx = 0; gx < 4; ++gx) {
            const float cx = -2.4f + 1.6f * static_cast<float>(gx);
            const float cy = -1.2f + 0.8f * static_cast<float>(gy);
            world.push_back(Float3{cx, cy + 0.25f, 0.0f});
            world.push_back(Float3{cx - 0.25f, cy - 0.25f, 0.0f});
            world.push_back(Float3{cx + 0.25f, cy - 0.25f, 0.0f});
        }
    }
    return world;
}

}  // namespace

TEST(Pipeline, SharedRasterDrawsTheSameFrameAsTheTiledWalk)
{
    // Where the triangles are read from is the only thing that changed.
    const std::vector<Float3> world = scattered_scene();

    const std::vector<Float3> tiled = render_triangle_tiled(world);
    const std::vector<Float3> shared = render_triangle_shared(world);
    ASSERT_EQ(shared.size(), tiled.size());

    for (size_t i = 0; i < tiled.size(); ++i) {
        ASSERT_NEAR(shared[i].x, tiled[i].x, PIXEL_EPS) << "pixel " << i << " r";
        ASSERT_NEAR(shared[i].y, tiled[i].y, PIXEL_EPS) << "pixel " << i << " g";
        ASSERT_NEAR(shared[i].z, tiled[i].z, PIXEL_EPS) << "pixel " << i << " b";
    }
}

TEST(Pipeline, SharedRasterIssuesLessWorkThanTheTiledWalk)
{
    // Staging costs a fill and a barrier once per block; it saves 92 units on
    // every one of the twelve loads each pixel makes per triangle.
    const std::vector<Float3> world = scattered_scene();

    MyGPURuntime tiled_rt(1u << 24);
    render_triangle_tiled(world, &tiled_rt);
    const uint64_t tiled = tiled_rt.stats().weighted_lane_ops;

    MyGPURuntime shared_rt(1u << 24);
    render_triangle_shared(world, WIDTH, HEIGHT, &shared_rt);
    const uint64_t shared = shared_rt.stats().weighted_lane_ops;

    EXPECT_LT(shared, tiled) << "staging has to pay for itself";
    EXPECT_LT(shared * 2, tiled) << "and by more than the fill costs";
}

TEST(Pipeline, SharedRasterSurvivesAFrameThatDoesNotFillItsTiles)
{
    // The test the barrier placement exists for. 50x20 leaves edge blocks whose
    // last threads are off screen; if the fill and the barrier sat inside the
    // bounds check, those threads would branch past it and the scheduler would
    // refuse the launch.
    const std::vector<Float3> world = scattered_scene();

    std::vector<Float3> frame;
    ASSERT_NO_THROW(frame = render_triangle_shared(world, 50, 20));
    ASSERT_EQ(frame.size(), 50u * 20u);

    uint32_t covered = 0;
    for (const Float3& pixel : frame) {
        if (pixel.x + pixel.y + pixel.z > 0.0f) {
            ++covered;
        }
    }
    EXPECT_GT(covered, 0u) << "a blank frame would satisfy the check above";
}

TEST(Pipeline, SharedRasterRejectsATileItCannotStage)
{
    MyGPURuntime rt(1u << 20);
    TiledRasterStageArgs args;
    args.width = WIDTH;
    args.height = HEIGHT;
    args.tiles_x = 2;
    args.max_tile_triangles = SHARED_TRIANGLE_CAPACITY + 1;

    EXPECT_THROW(run_shared_raster_stage(rt, args), std::runtime_error);
}

TEST(Pipeline, SharedRasterStagesThroughSharedMemory)
{
    // Cheap structural check: the triangles have to arrive in shared memory,
    // and the block has to meet before anyone reads them back.
    TiledRasterStageArgs args;
    args.width = WIDTH;
    args.height = HEIGHT;
    args.tiles_x = 2;
    args.max_tile_triangles = 4;

    void* raw[] = {&args};
    const Program p = build_shared_raster_program(raw);

    uint32_t stores = 0;
    uint32_t loads = 0;
    uint32_t barriers = 0;
    for (const Instruction& i : p) {
        stores += (i.op == Opcode::V_ST_SHARED_F32) ? 1 : 0;
        loads += (i.op == Opcode::V_LD_SHARED_F32) ? 1 : 0;
        barriers += (i.op == Opcode::BARRIER) ? 1 : 0;
    }
    EXPECT_GT(stores, 0u) << "the fill has to write shared memory";
    EXPECT_EQ(loads, TILE_TRIANGLE_FLOATS)
        << "a whole screen triangle per pass, all of it from shared";
    EXPECT_EQ(barriers, 1u);
}

// ---------------------------------------------------------------------------
// Ray tracing — the same image by other arithmetic
// ---------------------------------------------------------------------------

namespace {

// One triangle facing the camera, wound so the intersection test accepts it.
// The winding matters here in a way it does not for the rasteriser: culling a
// back face falls out of testing a < eps rather than |a| < eps.
WorldTriangle facing_triangle()
{
    return WorldTriangle{Float3{0.0f, 0.4f, 0.0f}, Float3{-0.4f, -0.4f, 0.0f},
                         Float3{0.4f, -0.4f, 0.0f}};
}

RayBasis default_basis()
{
    return ray_basis(default_camera(),
                     static_cast<float>(WIDTH) / static_cast<float>(HEIGHT));
}

// The scene both renderers draw, as world triangles and as a flat vertex list.
std::vector<WorldTriangle> shared_scene()
{
    std::vector<WorldTriangle> out;
    for (uint32_t g = 0; g < 4; ++g) {
        const float cx = -1.5f + 1.0f * static_cast<float>(g);
        out.push_back(WorldTriangle{Float3{cx, 0.3f, 0.0f},
                                    Float3{cx - 0.3f, -0.3f, 0.0f},
                                    Float3{cx + 0.3f, -0.3f, 0.0f}});
    }
    return out;
}

std::vector<Float3> as_vertex_list(const std::vector<WorldTriangle>& triangles)
{
    std::vector<Float3> out;
    for (const WorldTriangle& t : triangles) {
        out.push_back(t.v0);
        out.push_back(t.v1);
        out.push_back(t.v2);
    }
    return out;
}

// Through draw_raytrace, so the kernel these tests check is the one
// benchmarks/render_bench times. It allocates a screen buffer this route never
// reads — the price of one upload path for both renderers, and invisible to
// counters that ignore allocation.
std::vector<Float3> render_traced_with(const std::vector<WorldTriangle>& triangles,
                                       const Camera& camera, const Shading& shading,
                                       MyGPURuntime* external)
{
    MyGPURuntime local(1u << 24);
    return draw_raytrace((external != nullptr) ? *external : local,
                         as_vertex_list(triangles), DrawTarget{WIDTH, HEIGHT, camera},
                         shading);
}

std::vector<Float3> render_traced(const std::vector<WorldTriangle>& triangles,
                                  MyGPURuntime* external = nullptr,
                                  const Shading& shading = Shading{})
{
    return render_traced_with(triangles, default_camera(), shading, external);
}

std::vector<Float3> render_traced_from(const std::vector<WorldTriangle>& triangles,
                                       const Camera& camera)
{
    return render_traced_with(triangles, camera, Shading{}, nullptr);
}

// The raster path from an arbitrary camera, which only the angled-view test
// needs — render_triangle keeps its own default.
std::vector<Float3> render_triangle_from(const std::vector<Float3>& world,
                                         const Camera& camera)
{
    MyGPURuntime rt(1u << 24);
    return draw_walk(rt, world, DrawTarget{WIDTH, HEIGHT, camera});
}

}  // namespace

TEST(Pipeline, RayBasisAgreesWithLookAt)
{
    // Both cameras have to see the same thing, or comparing the renderers
    // compares two scenes.
    const Camera cam = default_camera();
    const RayBasis basis = default_basis();

    expect_near(basis.origin, cam.eye, PIXEL_EPS, "origin");
    EXPECT_LT(basis.forward.z, 0.0f) << "the camera looks down -z";
    EXPECT_NEAR(dot(basis.right, basis.forward), 0.0f, PIXEL_EPS) << "right is square on";
    EXPECT_NEAR(dot(basis.up, basis.forward), 0.0f, PIXEL_EPS);
    EXPECT_GT(length(basis.right), length(basis.up))
        << "a wide frame spreads further across than down";
}

TEST(Pipeline, IntersectFindsAFrontHit)
{
    const WorldTriangle t = facing_triangle();
    const Hit centre = intersect(t, Float3{0.0f, 0.0f, 3.0f}, Float3{0.0f, 0.0f, -1.0f});

    ASSERT_TRUE(centre.hit);
    EXPECT_NEAR(centre.t, 3.0f, PIXEL_EPS) << "the triangle sits at z = 0";
    EXPECT_GT(centre.u, 0.0f);
    EXPECT_GT(centre.v, 0.0f);
    EXPECT_LT(centre.u + centre.v, 1.0f) << "inside means the weights sum below one";
}

TEST(Pipeline, IntersectRejectsWhatItShould)
{
    const WorldTriangle t = facing_triangle();

    // Pointing away: the hit is behind the origin, which the eps on t rejects.
    EXPECT_FALSE(intersect(t, Float3{0, 0, 3}, Float3{0, 0, 1}).hit) << "behind";

    // Past the edge.
    EXPECT_FALSE(intersect(t, Float3{0, 0, 3}, Float3{2.0f, 0.0f, -1.0f}).hit) << "wide";

    // From behind the triangle, which a < eps culls.
    EXPECT_FALSE(intersect(t, Float3{0, 0, -3}, Float3{0, 0, 1}).hit) << "back face";
}

TEST(Pipeline, TracePixelPutsTheTriangleWhereTheProjectionDoes)
{
    // The two paths meeting for one pixel, before a whole frame is asked of
    // them. The centre of the frame looks at the origin, which the triangle
    // covers.
    const std::vector<WorldTriangle> scene = {facing_triangle()};
    const Float3 centre =
        trace_pixel(scene, default_basis(), WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT);
    EXPECT_GT(centre.x + centre.y + centre.z, 0.0f) << "the ray has to hit";

    const Float3 corner = trace_pixel(scene, default_basis(), 0, 0, WIDTH, HEIGHT);
    EXPECT_FLOAT_EQ(corner.x + corner.y + corner.z, 0.0f) << "and miss in the corner";
}

TEST(Pipeline, RaytraceStageMatchesTheHostTrace)
{
    const std::vector<WorldTriangle> scene = shared_scene();
    const std::vector<Float3> frame = render_traced(scene);
    ASSERT_EQ(frame.size(), static_cast<size_t>(WIDTH) * HEIGHT);

    const RayBasis basis = default_basis();
    uint32_t covered = 0;
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            const Float3 want = trace_pixel(scene, basis, x, y, WIDTH, HEIGHT);
            const Float3 got = frame[y * WIDTH + x];
            ASSERT_NEAR(got.x, want.x, PIXEL_EPS) << "pixel " << x << "," << y << " r";
            ASSERT_NEAR(got.y, want.y, PIXEL_EPS) << "pixel " << x << "," << y << " g";
            ASSERT_NEAR(got.z, want.z, PIXEL_EPS) << "pixel " << x << "," << y << " b";
            if (want.x + want.y + want.z > 0.0f) {
                ++covered;
            }
        }
    }
    EXPECT_GT(covered, 0u) << "a blank frame would satisfy every check above";
}

TEST(Pipeline, RaytracerAndRasteriserDrawTheSameScene)
{
    // What the whole project is for. Möller-Trumbore and edge functions share
    // no arithmetic, so agreeing on a frame is evidence neither can produce on
    // its own — the host references they are each checked against were written
    // from the same conventions, and a sign wrong in both would pass.
    const std::vector<WorldTriangle> scene = shared_scene();

    const std::vector<Float3> traced = render_traced(scene);
    const std::vector<Float3> rastered = render_triangle(as_vertex_list(scene));
    ASSERT_EQ(traced.size(), rastered.size());

    uint32_t differing = 0;
    for (size_t i = 0; i < traced.size(); ++i) {
        const bool same = std::abs(traced[i].x - rastered[i].x) < PIXEL_EPS &&
                          std::abs(traced[i].y - rastered[i].y) < PIXEL_EPS &&
                          std::abs(traced[i].z - rastered[i].z) < PIXEL_EPS;
        if (!same) {
            ++differing;
        }
    }

    // Edge pixels can land either side of a boundary the two decide with
    // different arithmetic, so a handful is expected; a wrong convention shows
    // as hundreds.
    EXPECT_LT(differing, traced.size() / 100)
        << differing << " of " << traced.size() << " pixels disagree";
}

TEST(Pipeline, RaytraceStageRejectsAnEmptyScene)
{
    MyGPURuntime rt(1u << 20);
    RaytraceStageArgs args;
    args.basis = default_basis();
    args.width = WIDTH;
    args.height = HEIGHT;
    args.triangle_count = 0;

    EXPECT_THROW(run_raytrace_stage(rt, args), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Diffuse lighting — the point at which the two paths stop being comparable
// ---------------------------------------------------------------------------

namespace {

Shading lit()
{
    Shading s;
    s.mode = ShadingMode::Diffuse;
    s.light_position = Float3{0.0f, 0.0f, 5.0f};
    s.base_colour = Float3{1.0f, 1.0f, 1.0f};
    return s;
}

}  // namespace

TEST(Pipeline, DiffuseIsBrightestFacingTheLight)
{
    // The light is straight down +z; a surface whose normal points at it takes
    // the full base colour, and one turned away takes less.
    const Float3 at_light = shade_diffuse(Float3{0, 0, 1}, Float3{0, 0, 0}, lit());
    EXPECT_NEAR(at_light.x, 1.0f, PIXEL_EPS);

    const Float3 tilted =
        shade_diffuse(normalize(Float3{1, 0, 1}), Float3{0, 0, 0}, lit());
    EXPECT_LT(tilted.x, at_light.x);
    EXPECT_GT(tilted.x, 0.0f);
}

TEST(Pipeline, DiffuseNeverLightsFromBehind)
{
    // dot goes negative once the surface turns away, and a negative scale would
    // brighten it again as it turned further.
    const Float3 away = shade_diffuse(Float3{0, 0, -1}, Float3{0, 0, 0}, lit());
    EXPECT_FLOAT_EQ(away.x, 0.0f);
    EXPECT_FLOAT_EQ(away.y, 0.0f);
    EXPECT_FLOAT_EQ(away.z, 0.0f);
}

TEST(Pipeline, DiffuseScalesTheBaseColour)
{
    Shading s = lit();
    s.base_colour = Float3{0.5f, 0.25f, 0.0f};

    const Float3 got = shade_diffuse(Float3{0, 0, 1}, Float3{0, 0, 0}, s);
    EXPECT_NEAR(got.x, 0.5f, PIXEL_EPS);
    EXPECT_NEAR(got.y, 0.25f, PIXEL_EPS);
    EXPECT_NEAR(got.z, 0.0f, PIXEL_EPS);
}

TEST(Pipeline, LitTracePixelIsNotTheBarycentricOne)
{
    // Both hit; only the colouring differs. Checked so that wiring the mode up
    // and then ignoring it would fail rather than pass quietly.
    const std::vector<WorldTriangle> scene = {facing_triangle()};

    const Float3 flat =
        trace_pixel(scene, default_basis(), WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT);
    const Float3 shaded =
        trace_pixel(scene, default_basis(), WIDTH / 2, HEIGHT / 2, WIDTH, HEIGHT, lit());

    EXPECT_GT(flat.x + flat.y + flat.z, 0.0f) << "the unlit one still hits";
    EXPECT_GT(shaded.x + shaded.y + shaded.z, 0.0f) << "and so does the lit one";
    EXPECT_GT(std::abs(shaded.x - flat.x) + std::abs(shaded.y - flat.y) +
                  std::abs(shaded.z - flat.z),
              PIXEL_EPS)
        << "but they must not agree";
}

TEST(Pipeline, LitKernelMatchesTheLitHost)
{
    const std::vector<WorldTriangle> scene = shared_scene();
    const std::vector<Float3> frame = render_traced(scene, nullptr, lit());
    ASSERT_EQ(frame.size(), static_cast<size_t>(WIDTH) * HEIGHT);

    const RayBasis basis = default_basis();
    uint32_t covered = 0;
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            const Float3 want = trace_pixel(scene, basis, x, y, WIDTH, HEIGHT, lit());
            const Float3 got = frame[y * WIDTH + x];
            ASSERT_NEAR(got.x, want.x, PIXEL_EPS) << "pixel " << x << "," << y;
            if (want.x + want.y + want.z > 0.0f) {
                ++covered;
            }
        }
    }
    EXPECT_GT(covered, 0u);
}

// ---------------------------------------------------------------------------
// A camera that moves
//
// Both paths already take an arbitrary Camera — ray_basis and view_projection
// each build their own from one. What was missing is evidence that the two
// build the *same* one anywhere but the axis-aligned default, where several
// conventions happen to agree.
// ---------------------------------------------------------------------------

namespace {

Camera angled_camera()
{
    Camera cam;
    cam.eye = Float3{2.0f, 1.5f, 2.5f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    cam.up = Float3{0.0f, 1.0f, 0.0f};
    cam.fov_y_degrees = 45.0f;
    return cam;
}

}  // namespace

TEST(Pipeline, MovingTheCameraMovesThePicture)
{
    const std::vector<WorldTriangle> scene = shared_scene();

    const std::vector<Float3> straight = render_traced(scene);
    const std::vector<Float3> angled = render_traced_from(scene, angled_camera());

    uint32_t differing = 0;
    uint32_t angled_covered = 0;
    for (size_t i = 0; i < straight.size(); ++i) {
        if (std::abs(straight[i].x - angled[i].x) > PIXEL_EPS) {
            ++differing;
        }
        if (angled[i].x + angled[i].y + angled[i].z > 0.0f) {
            ++angled_covered;
        }
    }
    EXPECT_GT(differing, straight.size() / 20) << "the view has to have changed";
    EXPECT_GT(angled_covered, 0u) << "and the scene must still be in frame";
}

TEST(Pipeline, BothRenderersAgreeFromAnAngle)
{
    // The stronger form of the agreement test. Off the axis, look_at and
    // ray_basis have nothing left to agree on by accident — a handedness or an
    // aspect applied to the wrong axis shows here and not at the default.
    const std::vector<WorldTriangle> scene = shared_scene();

    const std::vector<Float3> traced = render_traced_from(scene, angled_camera());
    const std::vector<Float3> rastered =
        render_triangle_from(as_vertex_list(scene), angled_camera());
    ASSERT_EQ(traced.size(), rastered.size());

    uint32_t differing = 0;
    uint32_t covered = 0;
    for (size_t i = 0; i < traced.size(); ++i) {
        const bool same = std::abs(traced[i].x - rastered[i].x) < PIXEL_EPS &&
                          std::abs(traced[i].y - rastered[i].y) < PIXEL_EPS &&
                          std::abs(traced[i].z - rastered[i].z) < PIXEL_EPS;
        if (!same) {
            ++differing;
        }
        if (traced[i].x + traced[i].y + traced[i].z > 0.0f) {
            ++covered;
        }
    }
    EXPECT_GT(covered, 0u) << "two blank frames agree perfectly and prove nothing";
    EXPECT_LT(differing, traced.size() / 100)
        << differing << " of " << traced.size() << " pixels disagree";
}

TEST(Pipeline, TheLitRasteriserAgreesWithTheLitRayTracer)
{
    // Lighting used to be the point at which the two renderers stopped being
    // comparable: pass 1 projects a vertex and keeps screen x, y, depth and 1/w,
    // so the world position a point light needs was gone by pass 2. Holding the
    // geometry on the device brought it back — pass 2 interpolates the world
    // vertices pass 1 read, with the same perspective correction it gives a
    // colour, and takes the triangle's normal from a buffer beside them.
    //
    // The two arrive at that position by different routes: the tracer solves for
    // the ray-triangle hit, the rasteriser interpolates across the projected
    // triangle. Agreeing is what says both are right.
    const std::vector<WorldTriangle> scene = shared_scene();
    const std::vector<Float3> world = as_vertex_list(scene);
    const DrawTarget target{WIDTH, HEIGHT, angled_camera()};

    Shading lit;
    lit.mode = ShadingMode::Diffuse;

    MyGPURuntime ray_rt(1u << 24);
    const std::vector<Float3> traced = draw_raytrace(ray_rt, world, target, lit);

    MyGPURuntime raster_rt(1u << 24);
    DeviceGeometry geometry = upload(raster_rt, world);
    DeviceFrame frame = allocate_frame(raster_rt, target);
    const std::vector<Float3> rastered =
        draw_walk(raster_rt, geometry, frame, target, false, lit);

    ASSERT_EQ(traced.size(), rastered.size());
    uint32_t differing = 0;
    uint32_t shaded = 0;
    for (size_t i = 0; i < traced.size(); ++i) {
        const bool same = std::abs(traced[i].x - rastered[i].x) < PIXEL_EPS &&
                          std::abs(traced[i].y - rastered[i].y) < PIXEL_EPS &&
                          std::abs(traced[i].z - rastered[i].z) < PIXEL_EPS;
        differing += same ? 0u : 1u;
        // Lit rather than merely covered: a diffuse term of zero would leave a
        // frame that a broken normal would also produce.
        if (rastered[i].x > 0.01f) {
            ++shaded;
        }
    }
    EXPECT_GT(shaded, 0u) << "nothing was lit, so nothing was compared";
    EXPECT_LT(differing, traced.size() / 100)
        << differing << " of " << traced.size() << " pixels disagree";

    // And it is not the barycentric frame under another name.
    const std::vector<Float3> plain = draw_walk(raster_rt, geometry, frame, target);
    uint32_t moved = 0;
    for (size_t i = 0; i < plain.size(); ++i) {
        moved += std::abs(plain[i].x - rastered[i].x) > PIXEL_EPS ? 1u : 0u;
    }
    EXPECT_GT(moved, 0u) << "Diffuse drew what Barycentric draws";

    release(raster_rt, frame);
    release(raster_rt, geometry);
}

TEST(Pipeline, LightingIsTheWalksAlone)
{
    // The tiled pair read their triangles from tile lists that carry screen
    // positions and nothing else. Lighting them means growing those lists by a
    // world position a vertex and a normal a triangle, which is a change to the
    // binning format rather than to a kernel — so they refuse rather than draw
    // something unlit and call it lit.
    const std::vector<Float3> world = as_vertex_list(shared_scene());
    const DrawTarget target{WIDTH, HEIGHT, angled_camera()};

    Shading lit;
    lit.mode = ShadingMode::Diffuse;

    MyGPURuntime rt(1u << 24);
    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, target);
    EXPECT_THROW(draw_tiled(rt, geometry, frame, target, false, lit), std::runtime_error);

    // And geometry uploaded for the ray tracer has no normals to light with.
    DeviceGeometry world_only = upload(rt, world, VertexStage::None);
    EXPECT_THROW(draw_walk(rt, world_only, frame, target, false, lit),
                 std::runtime_error);

    release(rt, world_only);
    release(rt, frame);
    release(rt, geometry);
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

namespace {

// Far enough back that the whole cube lands on the frame, and turned so that
// more than one face shows — a straight-on cube would exercise a single quad.
Camera cube_camera()
{
    Camera cam;
    cam.eye = Float3{1.6f, 1.2f, 2.4f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    return cam;
}

}  // namespace

TEST(Pipeline, MeshDrawsAsTheVertexListItExpandsTo)
{
    // What the Mesh overloads have to promise: indexing changes how the
    // geometry is stored, never what is drawn.
    const Mesh cube = cube_mesh();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime indexed_rt(1u << 24);
    const std::vector<Float3> indexed = draw_walk(indexed_rt, cube, target);

    MyGPURuntime flat_rt(1u << 24);
    const std::vector<Float3> flat = draw_walk(flat_rt, cube.flattened(), target);

    ASSERT_EQ(indexed.size(), flat.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        EXPECT_FLOAT_EQ(indexed[i].x, flat[i].x) << "pixel " << i << " r";
        EXPECT_FLOAT_EQ(indexed[i].y, flat[i].y) << "pixel " << i << " g";
        EXPECT_FLOAT_EQ(indexed[i].z, flat[i].z) << "pixel " << i << " b";
    }

    // Greater, not smaller, and that is the trade rather than a regression.
    // These runtimes hold pass 2 alone — the draw clears the counters between
    // its passes — and pass 2 is where indexing costs: fifteen loads a triangle
    // against twelve, the index having to arrive before the vertex it names.
    //
    // What it buys sits in pass 1, which IndexingTransformsEachVertexOnce
    // measures. The tiled and shared routes take that saving with none of this
    // cost, bin_triangles handing the kernel de-indexed triangles either way.
    EXPECT_GT(indexed_rt.stats().weighted_lane_ops, flat_rt.stats().weighted_lane_ops);
}

TEST(Pipeline, IndexingTransformsEachVertexOnce)
{
    // What indexing is for, and the one figure the draw routes hide: pass 1
    // runs a thread per vertex, so a cube's eight corners are eight transforms
    // where the flattened list spends thirty-six.
    const Mesh cube = cube_mesh();
    const std::vector<Float3> flat = cube.flattened();
    ASSERT_EQ(cube.vertex_count(), 8u);
    ASSERT_EQ(flat.size(), 36u);

    MyGPURuntime indexed_rt(1u << 20);
    run_stage(cube.vertices, cube.vertex_count(), -1.0f, &indexed_rt);

    MyGPURuntime flat_rt(1u << 20);
    run_stage(flat, static_cast<uint32_t>(flat.size()), -1.0f, &flat_rt);

    const uint64_t indexed = indexed_rt.stats().weighted_lane_ops;
    const uint64_t flattened = flat_rt.stats().weighted_lane_ops;

    EXPECT_LT(indexed, flattened);

    // Under a third, not merely less. Eight vertices against thirty-six is a
    // ratio of 0.22, and a bare EXPECT_LT would still pass on a saving too
    // small to have been worth an index buffer.
    EXPECT_LT(indexed * 3, flattened);
}

TEST(Pipeline, PredicatedRayTracerAgreesWithTheBranchItReplaces)
{
    // A cube proves less here than it looks. Folding the exits into a flag also
    // removes what they guarded, and neither hazard that creates shows up on a
    // closed mesh viewed from outside — the first draft of this test passed on
    // one while both were still broken.
    //
    // So two scenes chosen for what they can detect, and the cube kept only to
    // confirm the ordinary case still works.
    struct Case {
        const char* name;
        std::vector<Float3> world;
        Camera camera;
    };
    const Camera head_on{Float3{0.0f, 0.0f, 4.0f}, Float3{0, 0, 0}, Float3{0, 1, 0},
                         60.0f};
    std::vector<Case> cases;

    cases.push_back({"cube", cube_mesh().flattened(), cube_camera()});

    // A degenerate triangle: v0 == v1 leaves e1 zero, so the determinant is
    // exactly zero and an unguarded reciprocal is infinity. That infinity
    // reaches the blend, where multiplying it by a zero weight yields NaN — the
    // whole frame, not the one triangle.
    std::vector<Float3> degenerate = cube_mesh().flattened();
    degenerate.push_back(Float3{0.4f, 0.4f, 0.4f});
    degenerate.push_back(Float3{0.4f, 0.4f, 0.4f});
    degenerate.push_back(Float3{-0.4f, -0.4f, 0.4f});
    cases.push_back({"degenerate triangle", degenerate, cube_camera()});

    // Two triangles over the same pixels with the far one FIRST. The blend needs
    // a finite starting distance: infinity times a zero weight is NaN, every
    // later depth test against NaN fails, and nearest-wins quietly becomes
    // first-hit-wins. Nothing is NaN in the frame when that happens — the pixels
    // simply show the wrong triangle, so a test that only looks for NaN misses
    // it. Ordering the far one first is what makes the two rules disagree.
    cases.push_back({"far triangle first",
                     {Float3{-1.5f, -1.5f, -1.0f}, Float3{1.5f, -1.5f, -1.0f},
                      Float3{0.0f, 1.5f, -1.0f}, Float3{-0.5f, -0.5f, 1.0f},
                      Float3{0.5f, -0.5f, 1.0f}, Float3{0.0f, 0.5f, 1.0f}},
                     head_on});

    for (const Case& c : cases) {
        const DrawTarget target{WIDTH, HEIGHT, c.camera};

        MyGPURuntime branch_rt(1u << 24);
        const std::vector<Float3> branched = draw_raytrace(branch_rt, c.world, target);
        const uint64_t branch_cost = branch_rt.stats().weighted_lane_ops;
        const float branch_divergence = branch_rt.divergence_rate();

        MyGPURuntime blend_rt(1u << 24);
        const std::vector<Float3> blended =
            draw_raytrace(blend_rt, c.world, target, Shading{}, true);

        ASSERT_EQ(blended.size(), branched.size()) << c.name;
        for (size_t i = 0; i < branched.size(); ++i) {
            ASSERT_EQ(blended[i].x, branched[i].x) << c.name << " pixel " << i;
            ASSERT_EQ(blended[i].y, branched[i].y) << c.name << " pixel " << i;
            ASSERT_EQ(blended[i].z, branched[i].z) << c.name << " pixel " << i;
        }

        EXPECT_GT(branch_divergence, 0.0f) << c.name << " had nothing to remove";
        EXPECT_EQ(blend_rt.divergence_rate(), 0.0f) << c.name;
        EXPECT_GT(blend_rt.stats().weighted_lane_ops, branch_cost) << c.name;
    }
}

TEST(Pipeline, PredicatedRayTracerCostsMoreThanTheRasterBlendDoes)
{
    // The comparison the raster measurement could not make on its own. Its blend
    // loses about 2% because most of the loop sits outside the branch; the ray
    // tracer has far more inside one, and loses by more still. Having more of
    // the loop under the branch cuts both ways — every lane now finishes an
    // intersection it would have abandoned at the first of four exits.
    const std::vector<Float3> world = cube_mesh().flattened();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    const auto overhead = [&](bool ray) {
        MyGPURuntime branch_rt(1u << 24);
        MyGPURuntime blend_rt(1u << 24);
        if (ray) {
            draw_raytrace(branch_rt, world, target);
            draw_raytrace(blend_rt, world, target, Shading{}, true);
        } else {
            draw_walk(branch_rt, world, target, false);
            draw_walk(blend_rt, world, target, true);
        }
        const double before = static_cast<double>(branch_rt.stats().weighted_lane_ops);
        return (static_cast<double>(blend_rt.stats().weighted_lane_ops) - before) /
               before;
    };

    EXPECT_GT(overhead(true), overhead(false));
}

TEST(Pipeline, PredicationChangesTheCostAndNotThePixels)
{
    const Mesh cube = cube_mesh();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    // Every route the flag reaches, indexed and flattened both, against the
    // branch it replaces. A blend is only a variant of a kernel if it draws the
    // same thing, and it has to do so exactly: the frames go through EQ, not
    // NEAR, because the arithmetic was written to keep one term whole and
    // multiply the other away rather than to land close.
    struct Case {
        const char* name;
        std::vector<Float3> (*route)(MyGPURuntime&, const Mesh&, const DrawTarget&, bool);
        // Whether removing the coverage branch removes all of the route's
        // divergence. The shared route stages its tile through a cooperative
        // fill, and lanes there disagree about how many triangles they carry —
        // a second source the coverage flag has no bearing on.
        bool sole_source;
    };
    const Case cases[] = {{"walk", draw_walk, true},
                          {"tiled", draw_tiled, true},
                          {"shared", draw_shared, false}};

    for (const Case& c : cases) {
        MyGPURuntime branch_rt(1u << 24);
        const std::vector<Float3> branched = c.route(branch_rt, cube, target, false);
        const uint64_t branch_cost = branch_rt.stats().weighted_lane_ops;
        const float branch_divergence = branch_rt.divergence_rate();

        MyGPURuntime blend_rt(1u << 24);
        const std::vector<Float3> blended = c.route(blend_rt, cube, target, true);

        ASSERT_EQ(blended.size(), branched.size()) << c.name;
        for (size_t i = 0; i < branched.size(); ++i) {
            ASSERT_EQ(blended[i].x, branched[i].x) << c.name << " pixel " << i;
            ASSERT_EQ(blended[i].y, branched[i].y) << c.name << " pixel " << i;
            ASSERT_EQ(blended[i].z, branched[i].z) << c.name << " pixel " << i;
        }

        // What the variant was built to demonstrate: no lane disagrees over the
        // coverage test, so the rate falls by everything that test contributed.
        EXPECT_GT(branch_divergence, 0.0f) << c.name << " had nothing to remove";
        EXPECT_LT(blend_rt.divergence_rate(), branch_divergence) << c.name;
        if (c.sole_source) {
            EXPECT_EQ(blend_rt.divergence_rate(), 0.0f) << c.name;
        }

        // And what it costs. Lanes the triangle never covered now shade anyway,
        // and on this machine that is dearer than the masked issue it saves —
        // for every scene measured, not only this one.
        EXPECT_GT(blend_rt.stats().weighted_lane_ops, branch_cost) << c.name;
    }
}

TEST(Pipeline, PredicationLosesByFarMoreOnceLoadsAreChargedByTheLine)
{
    // The flat model was flattering the blend. Both variants make the same
    // loads, and at 100 a lane those drowned the arithmetic the blend adds; a
    // warp's load is one transaction once lines are counted, and the arithmetic
    // is then most of what is left to compare.
    const Mesh cube = cube_mesh();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    const auto penalty = [&cube, &target](MemoryModel model) {
        const auto cost = [&](bool predicated) {
            MyGPURuntime rt(1u << 24);
            rt.myrt_set_memory_model(model);
            draw_walk(rt, cube, target, predicated);
            return static_cast<double>(rt.stats().weighted_lane_ops);
        };
        const double branch = cost(false);
        return (cost(true) - branch) / branch;
    };

    const double flat = penalty(MemoryModel::Flat);
    const double cached = penalty(MemoryModel::Cached);

    EXPECT_GT(flat, 0.0) << "the blend was not dearer even under the flat charge";
    EXPECT_LT(flat, 0.05) << "the flat penalty is the couple of percent recorded";
    EXPECT_GT(cached, 4 * flat)
        << "charging by the line left the blend's arithmetic no dearer than before";
}

TEST(Pipeline, GeometryHeldOnTheDeviceDrawsWhatUploadingEachTimeDoes)
{
    // The two forms are one implementation, so this is really asking whether the
    // one-shot wrappers still hand their buffers to the same routes. A pixel is
    // the strictest thing to compare and the cheapest to get wrong.
    const Mesh cube = cube_mesh();
    const std::vector<Float3> flat = cube.flattened();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime rt(1u << 24);
    DeviceGeometry indexed = upload(rt, cube);
    DeviceGeometry flattened = upload(rt, flat);
    DeviceGeometry world_only = upload(rt, flat, VertexStage::None);
    DeviceFrame frame = allocate_frame(rt, target);

    MyGPURuntime once(1u << 24);
    const auto same = [](const std::vector<Float3>& a, const std::vector<Float3>& b,
                         const char* what) {
        ASSERT_EQ(a.size(), b.size()) << what;
        for (size_t i = 0; i < a.size(); ++i) {
            ASSERT_EQ(a[i].x, b[i].x) << what << " pixel " << i;
            ASSERT_EQ(a[i].y, b[i].y) << what << " pixel " << i;
            ASSERT_EQ(a[i].z, b[i].z) << what << " pixel " << i;
        }
    };

    same(draw_walk(rt, indexed, frame, target), draw_walk(once, cube, target),
         "walk, indexed");
    same(draw_walk(rt, flattened, frame, target), draw_walk(once, flat, target),
         "walk, flattened");
    same(draw_tiled(rt, indexed, frame, target), draw_tiled(once, cube, target), "tiled");
    same(draw_shared(rt, flattened, frame, target), draw_shared(once, flat, target),
         "shared");
    same(draw_raytrace(rt, world_only, frame, target), draw_raytrace(once, flat, target),
         "raytrace");

    // The route with no vertex stage cannot take an index buffer, and says so
    // rather than resolving one: an index list exists to feed the transform this
    // route does not have.
    EXPECT_THROW(draw_raytrace(rt, indexed, frame, target), std::runtime_error);

    release(rt, frame);
    release(rt, world_only);
    release(rt, flattened);
    release(rt, indexed);
}

TEST(Pipeline, ASecondDrawOfResidentGeometryFindsItAlreadyThere)
{
    // What uploading once is for. Every miss this project has measured was
    // compulsory, because a draw abandoned its buffers and the next one asked
    // about a fresh address; holding the geometry, the second draw reads the
    // lines the first one fetched.
    const Mesh sphere = load_obj(std::string(GPURT_ASSETS_DIR) + "/sphere.obj");
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime rt(1u << 26);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    DeviceGeometry geometry = upload(rt, sphere);
    DeviceFrame frame = allocate_frame(rt, target);

    // Read after the draw rather than differenced across it: the sync between
    // the passes clears the counters, so what stands afterwards is pass 2 alone.
    const auto misses = [&rt, &geometry, &frame, &target] {
        draw_walk(rt, geometry, frame, target);
        return rt.stats().cache_misses;
    };

    const uint64_t first = misses();
    const uint64_t second = misses();
    EXPECT_GT(first, 0u);
    EXPECT_EQ(second, 0u) << "the second draw refetched what the first had left";

    // And the other half of the claim: uploading again replaces the bytes under
    // those lines, so the draw after it has to fetch them again. Only the
    // buffers uploaded — pass 1 rewrites the screen buffer through the kernel,
    // which the cache does see.
    DeviceGeometry again = upload(rt, sphere);
    release(rt, geometry);
    geometry = again;
    EXPECT_GT(misses(), second) << "an upload was mistaken for the data already there";

    release(rt, frame);
    release(rt, geometry);
}

TEST(Pipeline, ReleasedGeometryIsHandedBackToTheAllocator)
{
    // myrt_free's first use outside the tests, and the reason a benchmark can
    // draw a scene a hundred times: an arena this small holds two copies of the
    // sphere at once and nothing like a hundred.
    const Mesh sphere = load_obj(std::string(GPURT_ASSETS_DIR) + "/sphere.obj");
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime rt(1u << 18);
    const size_t free_at_rest = rt.myrt_device_free_bytes();

    for (int i = 0; i < 100; ++i) {
        DeviceGeometry geometry = upload(rt, sphere);
        DeviceFrame frame = allocate_frame(rt, target);
        ASSERT_LT(rt.myrt_device_free_bytes(), free_at_rest);
        release(rt, frame);
        release(rt, geometry);
    }

    // Byte for byte, not merely enough to go round again: a free list that gave
    // back slightly less each time would pass a hundred rounds and fail a
    // thousand.
    EXPECT_EQ(rt.myrt_device_free_bytes(), free_at_rest);
}

TEST(Pipeline, EveryRouteDrawsTheSameCube)
{
    const Mesh cube = cube_mesh();
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime walk_rt(1u << 24);
    const std::vector<Float3> walked = draw_walk(walk_rt, cube, target);

    MyGPURuntime tiled_rt(1u << 24);
    const std::vector<Float3> tiled = draw_tiled(tiled_rt, cube, target);

    MyGPURuntime shared_rt(1u << 24);
    const std::vector<Float3> shared = draw_shared(shared_rt, cube, target);

    // A cube seen from an angle has faces at different depths, so this fails if
    // nearest-wins is wrong in any of the three.
    ASSERT_EQ(tiled.size(), walked.size());
    for (size_t i = 0; i < walked.size(); ++i) {
        ASSERT_NEAR(tiled[i].x, walked[i].x, PIXEL_EPS) << "tiled pixel " << i;
        ASSERT_NEAR(shared[i].x, walked[i].x, PIXEL_EPS) << "shared pixel " << i;
        ASSERT_NEAR(tiled[i].y, walked[i].y, PIXEL_EPS) << "tiled pixel " << i;
        ASSERT_NEAR(shared[i].y, walked[i].y, PIXEL_EPS) << "shared pixel " << i;
    }

    // Not a blank frame, which every comparison above would also pass.
    bool lit = false;
    for (const Float3& p : walked) {
        lit = lit || p.x > 0.0f || p.y > 0.0f || p.z > 0.0f;
    }
    EXPECT_TRUE(lit) << "the cube did not reach the frame";
}

TEST(Pipeline, ReorderingForAVertexCacheBuysNothingUnderTheFlatModel)
{
    // Ordering a mesh for a vertex cache is worth a factor of three or four on
    // fixed-function hardware. Pass 1 here materialises each unique vertex once
    // whatever order the triangles arrive in, so the issued work barely notices
    // — and that is what materialising cost a buffer and a pass for.
    //
    // A flat charge per lane also cannot tell a vertex just read from one that was
    // not, which is a second reason for the same reading. The test below separates
    // them by putting the question to MemoryModel::Cached.
    const Mesh sphere = load_obj(std::string(GPURT_ASSETS_DIR) + "/sphere.obj");
    const Mesh mixed = shuffled(sphere, 12345);
    const Mesh fixed = optimised_for_cache(mixed, 32);

    // The counterfactual moves a long way.
    EXPECT_LT(simulated_cache_misses(fixed, 32) * 3, simulated_cache_misses(mixed, 32));

    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    MyGPURuntime mixed_rt(1u << 26);
    const std::vector<Float3> mixed_frame = draw_tiled(mixed_rt, mixed, target);

    MyGPURuntime fixed_rt(1u << 26);
    const std::vector<Float3> fixed_frame = draw_tiled(fixed_rt, fixed, target);

    // Within a tenth of a percent, not to the byte. The residue is the next
    // test: nearest-wins is a data-dependent branch, and which triangle reaches
    // a pixel first decides how often it fires.
    const double mixed_ops = static_cast<double>(mixed_rt.stats().weighted_lane_ops);
    const double fixed_ops = static_cast<double>(fixed_rt.stats().weighted_lane_ops);
    EXPECT_LT(std::abs(fixed_ops - mixed_ops) / mixed_ops, 0.001);

    // The picture does not move at all: only the order changed.
    ASSERT_EQ(fixed_frame.size(), mixed_frame.size());
    for (size_t i = 0; i < mixed_frame.size(); ++i) {
        EXPECT_FLOAT_EQ(fixed_frame[i].x, mixed_frame[i].x) << "pixel " << i;
        EXPECT_FLOAT_EQ(fixed_frame[i].y, mixed_frame[i].y) << "pixel " << i;
        EXPECT_FLOAT_EQ(fixed_frame[i].z, mixed_frame[i].z) << "pixel " << i;
    }
}

TEST(Pipeline, ReorderingPaysOnceTheCacheIsSmallerThanTheMesh)
{
    // What the flat model was hiding. The indexed walk carries the index buffer
    // into pass 2 and reads a vertex through it, so a vertex two triangles apart in
    // the list may still be resident — and whether it is depends on the order the
    // triangles arrive in, which is what Forsyth's heuristic chooses.
    //
    // A small frame: the reuse being measured is between the triangles a block
    // walks, and every block walks all of them however many pixels there are.
    const Mesh sphere = load_obj(std::string(GPURT_ASSETS_DIR) + "/sphere.obj");
    const Mesh mixed = shuffled(sphere, 12345);
    const Mesh fixed = optimised_for_cache(mixed, 32);
    const DrawTarget target{16, 16, cube_camera()};

    // Eight lines hold 64 of the sphere's 182 vertices. Latency modelled as well,
    // because an L1 hit against an L2 hit is 8 against 30 in issue capacity and 30
    // against 200 in cycles — the same reordering shows up far larger in time.
    const auto walk = [&target](const Mesh& mesh, size_t l1_lines) {
        MyGPURuntime rt(1u << 26);
        rt.myrt_set_memory_model(MemoryModel::Cached);
        rt.myrt_set_latency_model(LatencyModel::Modelled);
        rt.myrt_set_cache_lines(l1_lines, 512);
        draw_walk(rt, mesh, target);
        return rt.stats();
    };

    const SchedulerStats mixed_small = walk(mixed, 8);
    const SchedulerStats fixed_small = walk(fixed, 8);

    // The mechanism, and it is not fewer fetches: every line is touched by someone
    // either way, so the misses are compulsory and identical. What moves is which
    // level answered the rest.
    EXPECT_EQ(fixed_small.cache_misses, mixed_small.cache_misses);
    EXPECT_LT(fixed_small.l2_hits * 3, mixed_small.l2_hits);

    EXPECT_LT(fixed_small.weighted_lane_ops, mixed_small.weighted_lane_ops);
    EXPECT_LT(fixed_small.cycles * 10, mixed_small.cycles * 9)
        << "reordering saved less than a tenth of the cycles";

    // And the bound on the claim: at the hardware size the mesh fits, nothing is
    // evicted, and the reorder has nothing to win back.
    const SchedulerStats mixed_large = walk(mixed, L1_LINES);
    const SchedulerStats fixed_large = walk(fixed, L1_LINES);
    EXPECT_EQ(fixed_large.l2_hits, mixed_large.l2_hits);
    EXPECT_LT(std::abs(static_cast<double>(fixed_large.cycles) -
                       static_cast<double>(mixed_large.cycles)) /
                  static_cast<double>(mixed_large.cycles),
              0.001);
}

TEST(Pipeline, TriangleOrderReachesTheCountersOnlyThroughDepth)
{
    // Why the tolerance above is not zero, and it is not the vertex cache.
    //
    // A flat grid covers each pixel with exactly one triangle, so nearest-wins
    // succeeds once however the triangles are ordered and every counter comes
    // out identical. A sphere covers each pixel twice, front and back, and
    // whether the near one arrives first decides whether the branch fires once
    // or twice — which lanes of a warp then disagree about.
    const DrawTarget target{WIDTH, HEIGHT, cube_camera()};

    const Mesh grid = load_obj(std::string(GPURT_ASSETS_DIR) + "/grid.obj");
    MyGPURuntime flat_a(1u << 26);
    MyGPURuntime flat_b(1u << 26);
    draw_tiled(flat_a, grid, target);
    draw_tiled(flat_b, shuffled(grid, 4242), target);

    EXPECT_EQ(flat_a.stats().weighted_lane_ops, flat_b.stats().weighted_lane_ops);
    EXPECT_EQ(flat_a.stats().warp_steps, flat_b.stats().warp_steps);

    const Mesh sphere = load_obj(std::string(GPURT_ASSETS_DIR) + "/sphere.obj");
    MyGPURuntime round_a(1u << 26);
    MyGPURuntime round_b(1u << 26);
    draw_tiled(round_a, sphere, target);
    draw_tiled(round_b, shuffled(sphere, 4242), target);

    // Ordinary work is unchanged; what the shuffle costs is agreement.
    EXPECT_GT(round_b.stats().warp_steps, round_a.stats().warp_steps);
    EXPECT_GT(round_b.divergence_rate(), round_a.divergence_rate());
}
