#include <cstdint>
#include <stdexcept>
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

// Where the host says each vertex lands, which is what a rendered frame is
// checked against.
std::vector<Float3> project_all(const std::vector<Float3>& world)
{
    const Float4x4 vp = default_vp();
    std::vector<Float3> screen;
    for (const Float3& v : world) {
        screen.push_back(project_vertex(vp, v, WIDTH, HEIGHT));
    }
    return screen;
}

// Both passes end to end, returning the framebuffer a pixel at a time.
//
// Pass 1 is followed by myrt_sync, which clears the statistics: with three
// vertices in a 32-lane warp its range guard masks 29 of them, and leaving that
// in would swamp any reading of what pass 2 alone costs. A caller that passes
// its own runtime therefore sees the raster pass's divergence and nothing else.
std::vector<Float3> render_triangle(const std::vector<Float3>& world,
                                    MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 22);
    MyGPURuntime& rt = (external != nullptr) ? *external : local;

    const size_t world_bytes = world.size() * WORLD_VERTEX_BYTES;
    const size_t screen_bytes = world.size() * SCREEN_VERTEX_BYTES;
    const size_t frame_bytes = static_cast<size_t>(WIDTH) * HEIGHT * PIXEL_BYTES;

    void* world_dev = rt.myrt_malloc(world_bytes);
    void* screen_dev = rt.myrt_malloc(screen_bytes);
    void* frame_dev = rt.myrt_malloc(frame_bytes);

    std::vector<float> flat;
    for (const Float3& v : world) {
        flat.push_back(v.x);
        flat.push_back(v.y);
        flat.push_back(v.z);
    }
    rt.myrt_memcpy(world_dev, flat.data(), world_bytes, Direction::HostToDevice);

    VertexStageArgs vertex_args;
    vertex_args.view_projection = default_vp();
    vertex_args.world_offset = rt.myrt_device_offset(world_dev);
    vertex_args.screen_offset = rt.myrt_device_offset(screen_dev);
    vertex_args.vertex_count = static_cast<uint32_t>(world.size());
    vertex_args.width = WIDTH;
    vertex_args.height = HEIGHT;
    run_vertex_stage(rt, vertex_args);
    rt.myrt_sync();

    RasterStageArgs raster_args;
    raster_args.screen_offset = rt.myrt_device_offset(screen_dev);
    raster_args.framebuffer_offset = rt.myrt_device_offset(frame_dev);
    raster_args.width = WIDTH;
    raster_args.height = HEIGHT;
    raster_args.triangle_count = static_cast<uint32_t>(world.size() / 3);
    run_raster_stage(rt, raster_args);

    std::vector<float> out(static_cast<size_t>(WIDTH) * HEIGHT * PIXEL_FLOATS, 0.0f);
    rt.myrt_memcpy(out.data(), frame_dev, frame_bytes, Direction::DeviceToHost);

    std::vector<Float3> frame;
    for (size_t i = 0; i < out.size(); i += PIXEL_FLOATS) {
        frame.push_back(Float3{out[i + 0], out[i + 1], out[i + 2]});
    }
    return frame;
}

// The tiled route to the same frame: bin on the host, then one block per tile.
// Pass 1 is followed by myrt_sync for the same reason as above, so a caller's
// runtime sees only what the raster pass cost.
std::vector<Float3> render_triangle_tiled(const std::vector<Float3>& world,
                                          MyGPURuntime* external = nullptr)
{
    MyGPURuntime local(1u << 24);
    MyGPURuntime& rt = (external != nullptr) ? *external : local;

    const size_t world_bytes = world.size() * WORLD_VERTEX_BYTES;
    const size_t screen_bytes = world.size() * SCREEN_VERTEX_BYTES;
    const size_t frame_bytes = static_cast<size_t>(WIDTH) * HEIGHT * PIXEL_BYTES;

    void* world_dev = rt.myrt_malloc(world_bytes);
    void* screen_dev = rt.myrt_malloc(screen_bytes);
    void* frame_dev = rt.myrt_malloc(frame_bytes);

    std::vector<float> flat;
    for (const Float3& v : world) {
        flat.push_back(v.x);
        flat.push_back(v.y);
        flat.push_back(v.z);
    }
    rt.myrt_memcpy(world_dev, flat.data(), world_bytes, Direction::HostToDevice);

    VertexStageArgs vertex_args;
    vertex_args.view_projection = default_vp();
    vertex_args.world_offset = rt.myrt_device_offset(world_dev);
    vertex_args.screen_offset = rt.myrt_device_offset(screen_dev);
    vertex_args.vertex_count = static_cast<uint32_t>(world.size());
    vertex_args.width = WIDTH;
    vertex_args.height = HEIGHT;
    run_vertex_stage(rt, vertex_args);
    rt.myrt_sync();

    // Binning reads the projected vertices, so it happens after pass 1 — on the
    // host, which means reading them back first. Real hardware bins in a
    // geometry stage without the round trip.
    std::vector<float> screen(world.size() * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(screen.data(), screen_dev, screen_bytes, Direction::DeviceToHost);

    std::vector<ScreenTriangle> triangles;
    for (size_t i = 0; i + 8 < screen.size(); i += 9) {
        triangles.push_back(
            ScreenTriangle{Float3{screen[i + 0], screen[i + 1], screen[i + 2]},
                           Float3{screen[i + 3], screen[i + 4], screen[i + 5]},
                           Float3{screen[i + 6], screen[i + 7], screen[i + 8]}});
    }

    const TileBinning binning = bin_triangles(triangles, WIDTH, HEIGHT);

    void* verts_dev = rt.myrt_malloc(binning.vertices.size() * sizeof(float));
    void* table_dev = rt.myrt_malloc(binning.table.size() * sizeof(float));
    rt.myrt_memcpy(verts_dev, binning.vertices.data(),
                   binning.vertices.size() * sizeof(float), Direction::HostToDevice);
    rt.myrt_memcpy(table_dev, binning.table.data(), binning.table.size() * sizeof(float),
                   Direction::HostToDevice);

    TiledRasterStageArgs raster_args;
    raster_args.tile_vertices_offset = rt.myrt_device_offset(verts_dev);
    raster_args.tile_table_offset = rt.myrt_device_offset(table_dev);
    raster_args.framebuffer_offset = rt.myrt_device_offset(frame_dev);
    raster_args.width = WIDTH;
    raster_args.height = HEIGHT;
    raster_args.tiles_x = binning.tiles_x;
    run_tiled_raster_stage(rt, raster_args);

    std::vector<float> out(static_cast<size_t>(WIDTH) * HEIGHT * PIXEL_FLOATS, 0.0f);
    rt.myrt_memcpy(out.data(), frame_dev, frame_bytes, Direction::DeviceToHost);

    std::vector<Float3> frame;
    for (size_t i = 0; i < out.size(); i += PIXEL_FLOATS) {
        frame.push_back(Float3{out[i + 0], out[i + 1], out[i + 2]});
    }
    return frame;
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

    const std::vector<Float3> screen = project_all(world);
    uint32_t covered = 0;
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            const Float3 want = shade_pixel(screen[0], screen[1], screen[2], x, y);
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

    const std::vector<Float3> screen = project_all(world);
    const std::vector<ScreenTriangle> triangles = {
        ScreenTriangle{screen[0], screen[1], screen[2]},
        ScreenTriangle{screen[3], screen[4], screen[5]},
    };

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

TEST(Pipeline, BinningStoresNineFloatsPerEntry)
{
    const TileBinning binning = bin_triangles(spread_triangles(), WIDTH, HEIGHT);

    uint32_t entries = 0;
    for (uint32_t t = 0; t < binning.tile_count(); ++t) {
        entries += static_cast<uint32_t>(binning.table[t * 2 + 1]);
    }
    EXPECT_EQ(binning.vertices.size(), entries * 9u);
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
