// A camera going round something, which is the shortest animation that shows a
// shape is a shape.
//
// Two of them, because they light the pipeline differently:
//
//   cube    a colour a face, carried from the vertex stage as varyings. Nothing
//           is lit; what turning it shows is that the six faces are six
//           surfaces and the depth test knows which is in front.
//   sphere  one colour lit by one direction, and the camera moving is the only
//           thing that changes. What turning it shows is the shading.
//
// They take different routes, and the reason is the walk's shape rather than a
// preference: it is one thread a pixel against every triangle, so 960 triangles
// at 240x240 is 64 seconds a frame. The cube's twelve cost nothing at any size;
// the sphere goes to the ray tracer, where a tree makes a pixel meet the few
// triangles it might rather than all of them. Lighting needs no vertex stage
// there — the tracer takes the normal from the edges it already loaded.
//
// Neither needs a skeleton, which is the point of having them beside
// skinned_render: an animation that goes wrong here has nothing to do with
// binding or forward kinematics.
//
//   ./build/apps/orbit                      result/images/orbit_cube.gif
//   ./build/apps/orbit --shape sphere       result/images/orbit_sphere.gif
//   ./build/apps/orbit --size 320 --frames 60

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gif.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/raytrace.hpp"
#include "pipeline/vertex.hpp"
#include "ppm.hpp"

namespace {

// Six faces, six colours, and a vertex takes the colour of the face it is on.
// The cube arrives as twelve triangles in face order, so the face is the
// triangle index halved — worked out here rather than on the device because the
// ISA has no integer division and this is a host loop over thirty-six vertices.
std::vector<float> face_colours(size_t vertices)
{
    const Float3 palette[6] = {
        Float3{0.90f, 0.25f, 0.25f}, Float3{0.25f, 0.75f, 0.35f},
        Float3{0.25f, 0.45f, 0.90f}, Float3{0.95f, 0.80f, 0.20f},
        Float3{0.75f, 0.30f, 0.85f}, Float3{0.20f, 0.80f, 0.85f},
    };

    std::vector<float> attributes;
    attributes.reserve(vertices * 3);
    for (size_t v = 0; v < vertices; ++v) {
        const Float3& colour = palette[(v / 6) % 6];
        attributes.push_back(colour.x);
        attributes.push_back(colour.y);
        attributes.push_back(colour.z);
    }
    return attributes;
}

// A box, wound outwards so the back-face test keeps the sides facing the camera
// and drops the ones behind them.
void add_box(std::vector<Float3>& tris, std::vector<float>& material, Float3 lo,
             Float3 hi, float which)
{
    const auto quad = [&](Float3 a, Float3 b, Float3 c, Float3 d) {
        tris.push_back(a);
        tris.push_back(b);
        tris.push_back(c);
        tris.push_back(a);
        tris.push_back(c);
        tris.push_back(d);
        material.push_back(which);
        material.push_back(which);
    };
    quad({lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, lo.y, hi.z});
    quad({hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, lo.y, lo.z});
    quad({lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z}, {lo.x, lo.y, hi.z});
    quad({hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}, {hi.x, lo.y, lo.z});
    quad({lo.x, hi.y, hi.z}, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z});
    quad({lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {lo.x, lo.y, hi.z});
}

// Four floats a material: a colour, then how much of the light leaves along the
// mirror direction. A floor at 0.15 shows what stands on it faintly; a mirror at
// 0.8 shows the room.
enum Material : uint32_t { FLOOR = 0, BLOCK = 1, BLOCK_TWO = 2, MIRROR = 3 };
inline constexpr float MATERIAL_TABLE[] = {
    0.82f, 0.80f, 0.78f, 0.15f,  // floor, a little of the room in it
    0.58f, 0.16f, 0.52f, 0.00f,  // one block
    0.90f, 0.34f, 0.30f, 0.00f,  // the other
    0.88f, 0.92f, 0.95f, 0.80f,  // the panel
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const uint32_t size = args.flag("size", 240);
    const uint32_t frames = args.flag("frames", 48);
    const std::string shape = args.text_flag("shape", "cube");
    const bool thrown = shape == "throw";
    const bool mirror = shape == "mirror";
    const bool cube = shape == "cube";
    const uint32_t bounces = args.flag("bounces", 4);

    const std::string assets = std::string(GPURT_ASSETS_DIR);
    const std::vector<Float3> world =
        load_obj(assets + "/" + (cube ? "cube.obj" : "sphere.obj")).flattened();

    // The bounding sphere, fitted to the field of view. Square frames, so the
    // two half-angles are the same and there is no tighter one to pick.
    Float3 lo = world[0];
    Float3 hi = world[0];
    for (const Float3& v : world) {
        lo = Float3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Float3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
    const Float3 centre{(lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f};
    const float radius = 0.5f * std::sqrt(dot(hi - lo, hi - lo));

    Camera framing;
    const float distance =
        radius / std::sin(radians(framing.fov_y_degrees) * 0.5f) * 1.15f;

    const std::vector<float> attributes =
        cube ? face_colours(world.size()) : std::vector<float>{};

    // One light, fixed in the world, so that turning the camera turns the shape
    // under it rather than carrying the highlight round with the view.
    Shading shading;
    if (cube) {
        shading.mode = ShadingMode::Custom;
        shading.shade = [](IRBuilder& k, const Fragment& f) {
            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(f.out.component(c), f.varyings[c]);
            }
        };
    } else {
        shading.mode = ShadingMode::Diffuse;
        shading.base_colour = Float3{0.35f, 0.65f, 0.95f};
        shading.light_position = Float3{
            centre.x + radius * 3.0f, centre.y + radius * 3.0f, centre.z + radius * 2.0f};
    }

    std::printf("\n[SCENE] %s, %zu triangles, %u frames at %ux%u\n\n", shape.c_str(),
                world.size() / 3, frames, size, size);

    // Where the sphere is at each step of the throw, and the box holding the whole
    // flight — so the camera is placed once and sees all of it.
    // Forward at a constant rate, and falling — but the ground sends it back up
    // with some of the speed gone, so the arcs shrink. Solved step by step
    // rather than in closed form: a bounce has no formula that carries across
    // it, which is why one arc is a parabola and a sequence of them is not.
    const float ground = centre.y - radius * 3.2f;
    const float restitution = 0.68f;
    const auto flight = [&](uint32_t count) {
        std::vector<Float3> path;
        Float3 at{centre.x - radius * 7.0f, centre.y + radius * 3.0f, centre.z};
        float rise = radius * 0.30f;
        const float forward = radius * 0.30f;
        const float pull = radius * 0.075f;
        for (uint32_t i = 0; i < count; ++i) {
            path.push_back(at);
            at = Float3{at.x + forward, at.y + rise, at.z};
            rise -= pull;
            if (at.y < ground) {
                at.y = ground + (ground - at.y) * restitution;
                rise = -rise * restitution;
            }
        }
        return path;
    };
    const std::vector<Float3> path = flight(frames);
    const auto thrown_at = [&](uint32_t frame) { return path[frame]; };

    Float3 flight_lo = thrown_at(0);
    Float3 flight_hi = thrown_at(0);
    for (uint32_t f = 0; f < frames; ++f) {
        const Float3 p = thrown_at(f);
        flight_lo = Float3{std::min(flight_lo.x, p.x), std::min(flight_lo.y, p.y),
                           std::min(flight_lo.z, p.z)};
        flight_hi = Float3{std::max(flight_hi.x, p.x), std::max(flight_hi.y, p.y),
                           std::max(flight_hi.z, p.z)};
    }
    const Float3 flight_centre{(flight_lo.x + flight_hi.x) * 0.5f,
                               (flight_lo.y + flight_hi.y) * 0.5f,
                               (flight_lo.z + flight_hi.z) * 0.5f};
    const float flight_radius =
        0.5f * std::sqrt(dot(flight_hi - flight_lo, flight_hi - flight_lo)) + radius;
    const float flight_distance =
        flight_radius / std::sin(radians(framing.fov_y_degrees) * 0.5f) * 1.15f;

    // The mirror room, built once. Two panels facing each other, a block between
    // them and a floor wide enough to run out of the frame.
    std::vector<Float3> room;
    std::vector<float> room_material;
    if (mirror) {
        const float r = radius;

        // A floor wide enough to run out of the frame, two blocks on it, and one
        // panel standing behind them. The panel is what the scene is for: what
        // it shows is a second ray traced from where the first stopped, and a
        // rasteriser has nothing to put there.
        add_box(room, room_material, {-r * 9.0f, -r * 0.3f, -r * 9.0f},
                {r * 9.0f, 0.0f, r * 9.0f}, static_cast<float>(FLOOR));
        add_box(room, room_material, {-r * 2.3f, 0.0f, -r * 0.6f},
                {-r * 0.5f, r * 1.8f, r * 1.2f}, static_cast<float>(BLOCK));
        add_box(room, room_material, {r * 0.6f, 0.0f, -r * 0.2f},
                {r * 2.6f, r * 2.0f, r * 1.8f}, static_cast<float>(BLOCK_TWO));
        add_box(room, room_material, {-r * 3.4f, 0.0f, -r * 4.6f},
                {r * 3.4f, r * 3.6f, -r * 4.3f}, static_cast<float>(MIRROR));
    }

    const float room_radius = radius * 5.5f;

    std::vector<std::vector<Float3>> animation;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const float turn =
            6.2831853f * static_cast<float>(frame) / static_cast<float>(frames);

        Camera camera = framing;
        const float room_distance =
            room_radius / std::sin(radians(framing.fov_y_degrees) * 0.5f) * 1.15f;
        camera.target = thrown ? flight_centre
                               : (mirror ? Float3{centre.x, centre.y + radius * 1.2f,
                                                  centre.z - radius * 1.6f}
                                         : centre);
        camera.eye = thrown ? Float3{flight_centre.x, flight_centre.y,
                                     flight_centre.z + flight_distance}
                            : (mirror ? Float3{centre.x + std::sin(turn) * radius * 1.3f,
                                               centre.y + radius * 3.4f,
                                               centre.z + radius * 9.5f -
                                                   std::cos(turn) * radius * 1.2f}
                                      : Float3{centre.x + std::sin(turn) * distance,
                                               centre.y + radius * 0.6f,
                                               centre.z + std::cos(turn) * distance});

        const float framed =
            thrown ? flight_distance : (mirror ? room_distance : distance);
        const float held = thrown ? flight_radius : (mirror ? room_radius : radius);
        camera.near_z = std::max(framed - held, framed * 0.01f);
        camera.far_z = framed + held * 2.0f;
        const DrawTarget target{size, size, camera};

        MyGPURuntime rt(1u << 27);

        if (mirror) {
            MyGPURuntime& r = rt;
            DeviceGeometry traced = upload_accelerated(r, room);
            DeviceFrame frame_buffer = allocate_frame(r, target);

            void* material = r.myrt_malloc(room_material.size() * sizeof(float));
            void* table = r.myrt_malloc(sizeof(MATERIAL_TABLE));
            r.myrt_memcpy(material, room_material.data(),
                          room_material.size() * sizeof(float), Direction::HostToDevice);
            r.myrt_memcpy(table, MATERIAL_TABLE, sizeof(MATERIAL_TABLE),
                          Direction::HostToDevice);

            RaytraceStageArgs walk;
            walk.basis = ray_basis(camera, target.aspect());
            walk.triangles_offset = r.myrt_device_offset(traced.world);
            walk.framebuffer_offset = r.myrt_device_offset(frame_buffer.pixels);
            walk.width = size;
            walk.height = size;
            walk.triangle_count = traced.triangle_count;
            walk.traversal = Traversal::Bvh;
            walk.stack_depth = traced.bvh_depth;
            walk.bvh_offset = r.myrt_device_offset(traced.bvh);
            walk.shade_when = ShadeWhen::Deferred;
            walk.bounces = bounces;
            walk.material_offset = r.myrt_device_offset(material);
            walk.material_table_offset = r.myrt_device_offset(table);
            walk.shading.light_position = Float3{0.55f, 1.0f, 0.45f};
            run_raytrace_stage(r, walk);

            animation.push_back(read_back(r, frame_buffer));
            if (frame % 12 == 0) {
                std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                            r.divergence_rate() * 100.0);
            }
            release(r, frame_buffer);
            release(r, traced);
            continue;
        }

        if (!cube) {
            // The throw moves the geometry and leaves everything else alone,
            // which is the cheaper of the two ways to animate: no matrix reaches
            // the device and no stage is asked to apply one.
            std::vector<Float3> placed = world;
            if (thrown) {
                const Float3 to = thrown_at(frame);
                for (Float3& v : placed) {
                    v = Float3{v.x - centre.x + to.x, v.y - centre.y + to.y,
                               v.z - centre.z + to.z};
                }
            }
            DeviceGeometry traced = upload_accelerated(rt, placed);
            DeviceFrame frame_buffer = allocate_frame(rt, target);
            animation.push_back(draw_raytrace(rt, traced, frame_buffer, target, shading));
            if (frame % 12 == 0) {
                std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                            rt.divergence_rate() * 100.0);
            }
            release(rt, frame_buffer);
            release(rt, traced);
            continue;
        }

        DeviceGeometry geometry = upload(rt, world);
        DeviceFrame buffer = allocate_frame(rt, target);
        clear_frame(rt, buffer, target);

        VertexStageArgs pass1;
        pass1.view_projection = target.camera.view_projection(target.aspect());
        pass1.world_offset = rt.myrt_device_offset(geometry.world);
        pass1.vertex_count = geometry.vertex_count;
        pass1.width = size;
        pass1.height = size;

        // Three varyings widen the screen vertex past what upload reserved.
        rt.myrt_free(geometry.screen);
        geometry.screen_bytes =
            static_cast<size_t>(geometry.vertex_count) * screen_vertex_bytes(3);
        geometry.screen = rt.myrt_malloc(geometry.screen_bytes);

        void* device_attributes = rt.myrt_malloc(attributes.size() * sizeof(float));
        rt.myrt_memcpy(device_attributes, attributes.data(),
                       attributes.size() * sizeof(float), Direction::HostToDevice);

        pass1.screen_offset = rt.myrt_device_offset(geometry.screen);
        pass1.screen_bytes = geometry.screen_bytes;
        pass1.varying_count = 3;
        pass1.attribute_offset = rt.myrt_device_offset(device_attributes);
        run_vertex_stage(rt, pass1);
        rt.myrt_sync(false);

        RasterStageArgs pass2;
        pass2.screen_offset = pass1.screen_offset;
        pass2.framebuffer_offset = rt.myrt_device_offset(buffer.pixels);
        pass2.depth_offset = rt.myrt_device_offset(buffer.depth);
        pass2.depth = DepthUse::Test;
        pass2.width = size;
        pass2.height = size;
        pass2.triangle_count = geometry.triangle_count;
        pass2.varying_count = 3;
        pass2.shading = shading;
        run_raster_stage(rt, pass2);

        animation.push_back(read_back(rt, buffer));
        if (frame % 12 == 0) {
            std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                        rt.divergence_rate() * 100.0);
        }
    }

    const std::string name = cube     ? "orbit_cube"
                             : thrown ? "orbit_throw"
                             : mirror ? "orbit_mirror"
                                      : "orbit_sphere";
    const std::string gif = args.images_dir() + name + ".gif";
    write_gif(gif, animation, size, size, 4);

    std::vector<float> flat;
    for (const Float3& pixel : animation.front()) {
        flat.push_back(pixel.x);
        flat.push_back(pixel.y);
        flat.push_back(pixel.z);
    }
    const std::string still = args.images_dir() + name + ".ppm";
    write_ppm(still, flat, size, size);

    std::printf("\nwrote %s and %s\n", gif.c_str(), still.c_str());
    return 0;
}
