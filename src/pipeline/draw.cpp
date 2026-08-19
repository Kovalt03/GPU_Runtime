#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "ir_builder.hpp"
#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/raster_tiled.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/vertex.hpp"

namespace {

// Releases whatever it was handed when the scope ends, so that the convenience
// overloads leak nothing even when a route refuses — draw_shared throws on a
// tile it cannot stage, and that is a supported answer rather than a failure.
template <typename Handle>
struct Owned {
    MyGPURuntime& rt;
    Handle& handle;

    ~Owned()
    {
        release(rt, handle);
    }
};

// One unit normal a triangle, computed where the world positions are already in
// hand and uploaded beside them.
//
// cross(v1 - v0, v2 - v0) normalised, which is the expression the ray tracer
// emits per hit — the two renderers have to agree about which side of a triangle
// faces the light, and the cheapest way to guarantee that is one formula.
//
// A degenerate triangle has no normal and gets a zero one rather than a NaN:
// normalize throws on a zero-length vector, and a scene is allowed to contain a
// sliver that no light will ever reach.
void upload_face_normals(MyGPURuntime& rt, DeviceGeometry& geometry,
                         const std::vector<Float3>& vertices,
                         const std::vector<uint32_t>& indices)
{
    const bool indexed = !indices.empty();
    const size_t triangles = indexed ? indices.size() / 3 : vertices.size() / 3;
    if (triangles == 0) {
        return;
    }

    std::vector<float> normals;
    normals.reserve(triangles * FACE_NORMAL_FLOATS);
    for (size_t t = 0; t < triangles; ++t) {
        const Float3 v0 = indexed ? vertices[indices[t * 3 + 0]] : vertices[t * 3 + 0];
        const Float3 v1 = indexed ? vertices[indices[t * 3 + 1]] : vertices[t * 3 + 1];
        const Float3 v2 = indexed ? vertices[indices[t * 3 + 2]] : vertices[t * 3 + 2];

        const Float3 face = cross(v1 - v0, v2 - v0);
        const Float3 unit = length(face) > 0.0f ? normalize(face) : Float3{};
        normals.push_back(unit.x);
        normals.push_back(unit.y);
        normals.push_back(unit.z);
    }

    const size_t bytes = normals.size() * sizeof(float);
    geometry.normals = rt.myrt_malloc(bytes);
    rt.myrt_memcpy(geometry.normals, normals.data(), bytes, Direction::HostToDevice);
}

// The half both uploads share: vertices on the device, and somewhere for pass 1
// to put them.
//
// Normals are not here because what counts as a triangle differs — a vertex list
// is triangles three at a time, a mesh is whatever its indices say — and doing
// it in the shared half meant allocating one set, overwriting the handle with
// the other, and never freeing the first. The leak test found it.
DeviceGeometry upload_positions(MyGPURuntime& rt, const std::vector<Float3>& world,
                                VertexStage stage)
{
    DeviceGeometry geometry;
    geometry.vertex_count = static_cast<uint32_t>(world.size());
    geometry.triangle_count = static_cast<uint32_t>(world.size() / 3);

    const size_t world_bytes =
        static_cast<size_t>(geometry.vertex_count) * WORLD_VERTEX_BYTES;
    geometry.world = rt.myrt_malloc(world_bytes);
    if (stage == VertexStage::Projects) {
        geometry.screen = rt.myrt_malloc(static_cast<size_t>(geometry.vertex_count) *
                                         SCREEN_VERTEX_BYTES);
    }

    std::vector<float> flat;
    flat.reserve(world.size() * WORLD_VERTEX_FLOATS);
    for (const Float3& v : world) {
        flat.push_back(v.x);
        flat.push_back(v.y);
        flat.push_back(v.z);
    }
    rt.myrt_memcpy(geometry.world, flat.data(), world_bytes, Direction::HostToDevice);
    return geometry;
}

// vertex_count rather than the vertices themselves, because that is all pass 1
// depends on: build_vertex_program transforms slot i into slot i and never asks
// whether i is shared.
void run_pass_one(MyGPURuntime& rt, const DeviceGeometry& geometry,
                  const DrawTarget& target)
{
    VertexStageArgs args;
    args.view_projection = target.camera.view_projection(target.aspect());
    args.world_offset = rt.myrt_device_offset(geometry.world);
    args.screen_offset = rt.myrt_device_offset(geometry.screen);
    args.vertex_count = geometry.vertex_count;
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
std::vector<ScreenTriangle> read_back_flattened(MyGPURuntime& rt,
                                                const DeviceGeometry& geometry)
{
    std::vector<float> screen(
        static_cast<size_t>(geometry.vertex_count) * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(screen.data(), geometry.screen, screen.size() * sizeof(float),
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
std::vector<ScreenTriangle> read_back_indexed(MyGPURuntime& rt,
                                              const DeviceGeometry& geometry)
{
    std::vector<float> screen(
        static_cast<size_t>(geometry.vertex_count) * SCREEN_VERTEX_FLOATS, 0.0f);
    rt.myrt_memcpy(screen.data(), geometry.screen, screen.size() * sizeof(float),
                   Direction::DeviceToHost);

    const auto at = [&screen](uint32_t i) {
        const size_t v = static_cast<size_t>(i) * SCREEN_VERTEX_FLOATS;
        return std::pair<Float3, float>{
            Float3{screen[v + 0], screen[v + 1], screen[v + 2]}, screen[v + 3]};
    };

    std::vector<ScreenTriangle> triangles;
    for (size_t t = 0; t < geometry.triangle_count; ++t) {
        const auto [v0, iw0] = at(geometry.indices[t * 3 + 0]);
        const auto [v1, iw1] = at(geometry.indices[t * 3 + 1]);
        const auto [v2, iw2] = at(geometry.indices[t * 3 + 2]);
        triangles.push_back(ScreenTriangle{v0, v1, v2, iw0, iw1, iw2});
    }
    return triangles;
}

// Which of the two the handle wants, so that no route has to ask.
std::vector<ScreenTriangle> read_back_triangles(MyGPURuntime& rt,
                                                const DeviceGeometry& geometry)
{
    return geometry.indexed() ? read_back_indexed(rt, geometry)
                              : read_back_flattened(rt, geometry);
}

std::vector<Float3> download(MyGPURuntime& rt, const DeviceFrame& frame_buffer)
{
    const size_t bytes =
        static_cast<size_t>(frame_buffer.width) * frame_buffer.height * PIXEL_BYTES;
    std::vector<float> out(bytes / sizeof(float), 0.0f);
    rt.myrt_memcpy(out.data(), frame_buffer.pixels, bytes, Direction::DeviceToHost);

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
//
// Allocated per draw and not held in the handle, because a tile list is a
// function of the camera as well as of the geometry: the same buffers seen from
// somewhere else bin differently.
TiledRasterStageArgs bin_and_upload(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                    const DeviceFrame& frame, const DrawTarget& target)
{
    const TileBinning binning =
        bin_triangles(read_back_triangles(rt, geometry), target.width, target.height);

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
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.width = target.width;
    args.height = target.height;
    args.tiles_x = binning.tiles_x;
    args.max_tile_triangles = fullest;
    return args;
}

}  // namespace

// --- what stays on the device ------------------------------------------------

DeviceGeometry upload(MyGPURuntime& rt, const std::vector<Float3>& world,
                      VertexStage stage)
{
    DeviceGeometry geometry = upload_positions(rt, world, stage);
    if (stage == VertexStage::Projects) {
        // Three vertices a triangle, in the order they arrive.
        upload_face_normals(rt, geometry, world, {});
    }
    return geometry;
}

DeviceGeometry upload(MyGPURuntime& rt, const Mesh& mesh)
{
    // mesh.vertices rather than mesh.flattened(): that is what sizes both the
    // world buffer and pass 1's output by the unique count, and a cube reserves
    // eight screen slots where the flattened list needed thirty-six.
    DeviceGeometry geometry = upload_positions(rt, mesh.vertices, VertexStage::Projects);
    geometry.triangle_count = mesh.triangle_count();
    geometry.indices = mesh.indices;

    // Indices reach the device as floats: the ISA has no integer registers, and
    // the kernel multiplies an index by a vertex stride to get an address, so it
    // wants a float there anyway. Whole numbers are exact to 2^24, which is more
    // vertices than a scene here will hold.
    std::vector<float> as_floats;
    as_floats.reserve(mesh.indices.size());
    for (const uint32_t i : mesh.indices) {
        as_floats.push_back(static_cast<float>(i));
    }

    const size_t index_bytes = as_floats.size() * sizeof(float);
    geometry.index = rt.myrt_malloc(index_bytes);
    rt.myrt_memcpy(geometry.index, as_floats.data(), index_bytes,
                   Direction::HostToDevice);

    // From the indices: the unique vertices in the order they happen to arrive
    // are not triangles, and normalling them would describe a shape nobody drew.
    upload_face_normals(rt, geometry, mesh.vertices, mesh.indices);
    return geometry;
}

DeviceFrame allocate_frame(MyGPURuntime& rt, const DrawTarget& target)
{
    DeviceFrame frame;
    frame.width = target.width;
    frame.height = target.height;
    const size_t pixels = static_cast<size_t>(target.width) * target.height;
    frame.pixels = rt.myrt_malloc(pixels * PIXEL_BYTES);
    frame.depth = rt.myrt_malloc(pixels * DEPTH_BYTES);
    return frame;
}

std::vector<Float3> read_back(MyGPURuntime& rt, const DeviceFrame& frame)
{
    return download(rt, frame);
}

void release(MyGPURuntime& rt, DeviceGeometry& geometry)
{
    // myrt_free takes a null pointer the way C's free does, so the routes that
    // reserve no screen buffer need no special case here.
    rt.myrt_free(geometry.world);
    rt.myrt_free(geometry.screen);
    rt.myrt_free(geometry.index);
    rt.myrt_free(geometry.normals);
    geometry = DeviceGeometry{};
}

void release(MyGPURuntime& rt, DeviceFrame& frame)
{
    rt.myrt_free(frame.pixels);
    rt.myrt_free(frame.depth);
    frame = DeviceFrame{};
}

// --- the routes over what is already there -----------------------------------

namespace {

// The clear, as a kernel and its arguments. One thread a pixel, four stores
// each — a kernel rather than a transfer from the host because a frame cleared
// every frame is device work, and because the depth and the colour are two
// buffers, so a host clear would be two transfers of what the device is about to
// overwrite.
struct ClearArgs {
    size_t framebuffer_offset = 0;
    size_t depth_offset = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    Float3 colour;
    float depth = 2.0f;
};

Program build_clear_program(void** raw)
{
    const ClearArgs& a = *static_cast<const ClearArgs*>(raw[0]);
    IRBuilder k;
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    k.if_(k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
                k.lt(py, k.constant(static_cast<float>(a.height)))),
          [&] {
              const Reg<Scalar> pixel =
                  k.add(k.mul(py, k.constant(static_cast<float>(a.width))), px);
              const Reg<Scalar> at =
                  k.mul(pixel, k.constant(static_cast<float>(PIXEL_BYTES)));
              const float base = static_cast<float>(a.framebuffer_offset);
              k.store(at, k.constant(a.colour.x), base + 0.0f);
              k.store(at, k.constant(a.colour.y), base + 4.0f);
              k.store(at, k.constant(a.colour.z), base + 8.0f);

              const Reg<Scalar> depth_at =
                  k.add(k.constant(static_cast<float>(a.depth_offset)),
                        k.mul(pixel, k.constant(static_cast<float>(DEPTH_BYTES))));
              k.store(depth_at, k.constant(a.depth), 0.0f);
          });
    return k.build();
}

ClearArgs clear_args(MyGPURuntime& rt, const DeviceFrame& frame, const DrawTarget& target,
                     Float3 colour, float depth)
{
    ClearArgs args;
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.depth_offset = rt.myrt_device_offset(frame.depth);
    args.width = target.width;
    args.height = target.height;
    args.colour = colour;
    args.depth = depth;
    return args;
}

dim3 clear_grid(const DrawTarget& target)
{
    return dim3{(target.width + WARP_SIZE - 1) / WARP_SIZE, target.height, 1};
}

}  // namespace

void clear_frame(MyGPURuntime& rt, const DeviceFrame& frame, const DrawTarget& target,
                 Float3 colour, float depth)
{
    ClearArgs args = clear_args(rt, frame, target, colour, depth);
    void* raw[] = {&args};
    rt.myrt_launch(build_clear_program, clear_grid(target), dim3{WARP_SIZE, 1, 1}, raw);
    rt.myrt_sync(false);
}

void queue_clear(MyGPURuntime& rt, const DeviceFrame& frame, const DrawTarget& target,
                 Float3 colour, float depth, StreamId stream)
{
    // args must outlive the launch, and this one returns immediately — so the
    // kernel is built here, where the arguments are still alive. myrt_launch_async
    // calls the builder before it returns, which is what makes that safe.
    ClearArgs args = clear_args(rt, frame, target, colour, depth);
    void* raw[] = {&args};
    const Program prog = build_clear_program(raw);
    rt.myrt_launch_async([prog](void**) { return prog; },
                         LaunchConfig{clear_grid(target), dim3{WARP_SIZE, 1, 1}}, nullptr,
                         stream);
}

std::vector<Float3> draw_depth_tested(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                      const DeviceFrame& frame, const DrawTarget& target,
                                      const Shading& shading)
{
    run_pass_one(rt, geometry, target);

    RasterStageArgs args;
    args.screen_offset = rt.myrt_device_offset(geometry.screen);
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = geometry.triangle_count;
    args.indexed = geometry.indexed();
    args.index_offset = geometry.indexed() ? rt.myrt_device_offset(geometry.index) : 0;
    args.shading = shading;
    args.world_offset = rt.myrt_device_offset(geometry.world);
    args.normal_offset =
        geometry.normals != nullptr ? rt.myrt_device_offset(geometry.normals) : 0;
    args.depth = DepthUse::Test;
    args.depth_offset = rt.myrt_device_offset(frame.depth);
    run_raster_stage(rt, args);
    return download(rt, frame);
}

std::vector<Float3> draw_walk(MyGPURuntime& rt, const DeviceGeometry& geometry,
                              const DeviceFrame& frame, const DrawTarget& target,
                              bool predicated, const Shading& shading)
{
    if (shading.mode == ShadingMode::Diffuse && geometry.normals == nullptr) {
        throw std::runtime_error(
            "draw_walk: lighting needs the face normals, and this geometry was "
            "uploaded without a vertex stage");
    }

    run_pass_one(rt, geometry, target);

    RasterStageArgs args;
    args.screen_offset = rt.myrt_device_offset(geometry.screen);
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.shading = shading;
    args.world_offset = rt.myrt_device_offset(geometry.world);
    args.normal_offset =
        geometry.normals == nullptr ? 0 : rt.myrt_device_offset(geometry.normals);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = geometry.triangle_count;
    args.predicated = predicated;
    if (geometry.indexed()) {
        // The whole of what indexing costs pass 2 is this pair: three dependent
        // loads a triangle, in exchange for the transforms pass 1 did not make.
        args.indexed = true;
        args.index_offset = rt.myrt_device_offset(geometry.index);
    }
    run_raster_stage(rt, args);
    return download(rt, frame);
}

std::vector<Float3> draw_early_z(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                 const DeviceFrame& frame, const DrawTarget& target,
                                 const Shading& shading)
{
    if (shading.mode == ShadingMode::Diffuse && geometry.normals == nullptr) {
        throw std::runtime_error(
            "draw_early_z: lighting needs the face normals, and this geometry was "
            "uploaded without a vertex stage");
    }

    run_pass_one(rt, geometry, target);

    RasterStageArgs args;
    args.screen_offset = rt.myrt_device_offset(geometry.screen);
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.depth_offset = rt.myrt_device_offset(frame.depth);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = geometry.triangle_count;
    args.shading = shading;
    args.world_offset = rt.myrt_device_offset(geometry.world);
    args.normal_offset =
        geometry.normals == nullptr ? 0 : rt.myrt_device_offset(geometry.normals);
    if (geometry.indexed()) {
        args.indexed = true;
        args.index_offset = rt.myrt_device_offset(geometry.index);
    }

    // No sync between them: what a caller reads is both passes, which is the
    // figure the trade is about. The prepass's stores are what the second launch
    // loads, and myrt_launch returning is what orders them.
    args.depth = DepthUse::Prepass;
    run_raster_stage(rt, args);

    args.depth = DepthUse::EarlyZ;
    run_raster_stage(rt, args);
    return download(rt, frame);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const DeviceGeometry& geometry,
                               const DeviceFrame& frame, const DrawTarget& target,
                               bool predicated, const Shading& shading)
{
    // The kernel is the same program whichever form the geometry arrived in:
    // bin_triangles copies each triangle into every tile it reaches, so what
    // reaches the device is already de-indexed.
    run_pass_one(rt, geometry, target);
    TiledRasterStageArgs args = bin_and_upload(rt, geometry, frame, target);
    args.predicated = predicated;
    args.shading = shading;
    run_tiled_raster_stage(rt, args);
    return download(rt, frame);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                const DeviceFrame& frame, const DrawTarget& target,
                                bool predicated, const Shading& shading,
                                bool async_staging)
{
    run_pass_one(rt, geometry, target);
    TiledRasterStageArgs args = bin_and_upload(rt, geometry, frame, target);
    args.predicated = predicated;
    args.shading = shading;
    args.staging =
        async_staging ? TileStaging::AsyncDoubleBuffered : TileStaging::Synchronous;
    run_shared_raster_stage(rt, args);
    return download(rt, frame);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const DeviceGeometry& geometry,
                                  const DeviceFrame& frame, const DrawTarget& target,
                                  const Shading& shading, bool predicated)
{
    if (geometry.indexed()) {
        throw std::runtime_error(
            "draw_raytrace: the geometry is indexed, and this route has no vertex "
            "stage for an index buffer to feed — upload it flattened");
    }

    RaytraceStageArgs args;
    args.basis = ray_basis(target.camera, target.aspect());
    args.shading = shading;
    args.triangles_offset = rt.myrt_device_offset(geometry.world);
    args.framebuffer_offset = rt.myrt_device_offset(frame.pixels);
    args.width = target.width;
    args.height = target.height;
    args.triangle_count = geometry.triangle_count;
    args.predicated = predicated;
    run_raytrace_stage(rt, args);
    return download(rt, frame);
}

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

// --- upload, draw, release ---------------------------------------------------
// The one-shot forms, which is what a test or a benchmark measuring a single
// frame wants. Each is the persistent pair with the buffers' lifetime narrowed
// to the call, so both forms run one implementation and the figures they produce
// describe the same code.

std::vector<Float3> draw_walk(MyGPURuntime& rt, const std::vector<Float3>& world,
                              const DrawTarget& target, bool predicated)
{
    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_walk(rt, geometry, frame, target, predicated);
}

std::vector<Float3> draw_predicated(MyGPURuntime& rt, const std::vector<Float3>& world,
                                    const DrawTarget& target)
{
    return draw_walk(rt, world, target, true);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const std::vector<Float3>& world,
                               const DrawTarget& target, bool predicated)
{
    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_tiled(rt, geometry, frame, target, predicated);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const std::vector<Float3>& world,
                                const DrawTarget& target, bool predicated)
{
    DeviceGeometry geometry = upload(rt, world);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_shared(rt, geometry, frame, target, predicated);
}

std::vector<Float3> draw_raytrace(MyGPURuntime& rt, const std::vector<Float3>& world,
                                  const DrawTarget& target, const Shading& shading,
                                  bool predicated)
{
    // VertexStage::None: nothing here projects, so pass 1's output buffer would
    // be reserved and never written.
    DeviceGeometry geometry = upload(rt, world, VertexStage::None);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_raytrace(rt, geometry, frame, target, shading, predicated);
}

std::vector<Float3> draw_walk(MyGPURuntime& rt, const Mesh& mesh,
                              const DrawTarget& target, bool predicated)
{
    // The whole of what indexing buys is in upload(): a cube runs eight threads
    // in pass 1 where the flattened list ran thirty-six, and a transform is the
    // most expensive instruction in the set.
    DeviceGeometry geometry = upload(rt, mesh);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_walk(rt, geometry, frame, target, predicated);
}

std::vector<Float3> draw_predicated(MyGPURuntime& rt, const Mesh& mesh,
                                    const DrawTarget& target)
{
    return draw_walk(rt, mesh, target, true);
}

std::vector<Float3> draw_tiled(MyGPURuntime& rt, const Mesh& mesh,
                               const DrawTarget& target, bool predicated)
{
    DeviceGeometry geometry = upload(rt, mesh);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_tiled(rt, geometry, frame, target, predicated);
}

std::vector<Float3> draw_shared(MyGPURuntime& rt, const Mesh& mesh,
                                const DrawTarget& target, bool predicated)
{
    DeviceGeometry geometry = upload(rt, mesh);
    DeviceFrame frame = allocate_frame(rt, target);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};
    const Owned<DeviceFrame> own_frame{rt, frame};
    return draw_shared(rt, geometry, frame, target, predicated);
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
                                 const DrawTarget& target, Uniforms uniforms)
{
    // No width or height in the plan, so no framebuffer: this runs pass 1 and
    // never writes a pixel. Saying so is the point of BufferPlan — this path
    // used to reach for a raster upload and reserve three megabytes of frame at
    // 512, which the runtime it had been given could not hold.
    BufferPlan plan;
    plan.world_vertices = static_cast<uint32_t>(vertices.size());
    plan.screen_vertices = plan.world_vertices;

    // Its own runtime, so the reading is a total rather than a difference, and
    // no frame is allocated at all — the plan is here to size the arena.
    MyGPURuntime rt(plan.device_bytes() + (1u << 16));
    DeviceGeometry geometry = upload(rt, vertices);
    const Owned<DeviceGeometry> own_geometry{rt, geometry};

    VertexStageArgs args;
    args.view_projection = target.camera.view_projection(target.aspect());
    args.world_offset = rt.myrt_device_offset(geometry.world);
    args.screen_offset = rt.myrt_device_offset(geometry.screen);
    args.vertex_count = geometry.vertex_count;
    args.width = target.width;
    args.height = target.height;

    if (uniforms == Uniforms::Window) {
        // The matrix goes to the device instead of into the instruction stream.
        // Sixteen floats of their own, so that the offset is not zero — which is
        // how the kernel is told to read rather than bake.
        constexpr uint32_t SIDE = 4;
        void* window = rt.myrt_malloc(SIDE * SIDE * sizeof(float));
        const Float4x4 mvp = target.camera.view_projection(target.aspect());
        std::array<float, SIDE * SIDE> flat{};
        for (uint32_t row = 0; row < SIDE; ++row) {
            for (uint32_t col = 0; col < SIDE; ++col) {
                flat[row * SIDE + col] = mvp.at(row, col);
            }
        }
        rt.myrt_memcpy(window, flat.data(), flat.size() * sizeof(float),
                       Direction::HostToDevice);
        args.uniform_offset = rt.myrt_device_offset(window);
    }

    run_vertex_stage(rt, args);

    return rt.stats();
}
