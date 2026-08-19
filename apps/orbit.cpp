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

// A box, wound so every face's normal points out of it.
//
// The intersection test rejects a back face, so a face wound the other way is
// simply not there — and a box with five of them inside out looks like a box
// seen through itself rather than like anything failing. cross(b - a, c - a)
// against the face's own centre is what says which way each one is, and it is
// worth checking rather than reasoning about.
//
// `which` names the first material; `per_face` walks the six from there, which
// is how a cube gets a colour a face without six calls.
void add_box(std::vector<Float3>& tris, std::vector<float>& material, Float3 lo,
             Float3 hi, float which, bool per_face = false)
{
    uint32_t face = 0;
    const auto quad = [&](Float3 a, Float3 b, Float3 c, Float3 d) {
        tris.push_back(a);
        tris.push_back(b);
        tris.push_back(c);
        tris.push_back(a);
        tris.push_back(c);
        tris.push_back(d);
        const float mine = which + (per_face ? static_cast<float>(face) : 0.0f);
        material.push_back(mine);
        material.push_back(mine);
        ++face;
    };
    quad({lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z});
    quad({hi.x, lo.y, lo.z}, {lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z});
    quad({lo.x, lo.y, lo.z}, {lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {lo.x, hi.y, lo.z});
    quad({hi.x, lo.y, hi.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z});
    quad({lo.x, hi.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z});
    quad({lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {lo.x, lo.y, hi.z});
}

// One face, wound so its normal points back down `-z` at the camera.
//
// A pane rather than a box for the glass, because a box would bend the ray once
// and not twice: the intersection test rejects a back face, so a ray already
// inside a closed volume never meets the far wall from within. One surface is
// what this route can actually model, so it is what the scene puts there.
void add_pane(std::vector<Float3>& tris, std::vector<float>& material, float which,
              Float3 lo, Float3 hi)
{
    tris.push_back(Float3{hi.x, lo.y, lo.z});
    tris.push_back(Float3{lo.x, lo.y, lo.z});
    tris.push_back(Float3{lo.x, hi.y, hi.z});
    tris.push_back(Float3{hi.x, lo.y, lo.z});
    tris.push_back(Float3{lo.x, hi.y, hi.z});
    tris.push_back(Float3{hi.x, hi.y, hi.z});
    material.push_back(which);
    material.push_back(which);
}

// Six faces of a cube, then the ground, a mirror, and a pane of glass.
// MATERIAL_FLOATS each: a colour, then what the surface does not keep —
// how much leaves along the mirror direction and how much through it, and the
// index the second of those bends by.
//
// The arrangement is the one Vulkan-Samples uses for its reflection sample —
// two cubes with a different colour a face, a ground that reflects a tenth, and
// two mirrors facing each other at nine tenths. The cubes are what a reflection
// has to show, the faces are how one tells which reflection it is looking at,
// and the facing pair is what makes a reflection contain another. The glass is
// this project's, and is here because refraction is the one direction a mirror
// scene never produces: every ray in it leaves on the side it arrived.
enum Material : uint32_t {
    FACE_ZERO = 0,  // six consecutive, one a face
    GROUND = 6,
    MIRROR = 7,
    GLASS = 8,
    BALL = 9,
};

// clang-format off
inline constexpr float MATERIAL_TABLE[] = {
    // colour            mirror  through  index
    0.90f, 0.15f, 0.15f,  0.00f,  0.00f,  1.00f,  // +z  red
    0.15f, 0.80f, 0.20f,  0.00f,  0.00f,  1.00f,  // -z  green
    0.20f, 0.35f, 0.90f,  0.00f,  0.00f,  1.00f,  // -x  blue
    0.90f, 0.85f, 0.20f,  0.00f,  0.00f,  1.00f,  // +x  yellow
    0.20f, 0.85f, 0.85f,  0.00f,  0.00f,  1.00f,  // +y  cyan
    0.85f, 0.25f, 0.85f,  0.00f,  0.00f,  1.00f,  // -y  magenta
    0.70f, 0.70f, 0.70f,  0.10f,  0.00f,  1.00f,  // ground
    0.30f, 0.90f, 1.00f,  0.90f,  0.00f,  1.00f,  // mirror
    0.80f, 0.92f, 1.00f,  0.10f,  0.85f,  1.52f,  // glass, near enough to crown
    0.30f, 0.60f, 0.95f,  0.05f,  0.00f,  1.00f,  // the thrown ball
};
// clang-format on
static_assert(sizeof(MATERIAL_TABLE) / sizeof(float) == (BALL + 1) * MATERIAL_FLOATS,
              "a material is MATERIAL_FLOATS wide and the kernel indexes by that");

// A walk with a material a triangle. The only route that has a shadow, a
// reflection or a refraction to give: all three read the table, and a shader
// that names one colour has nowhere to put them.
std::vector<Float3> trace_materials(MyGPURuntime& rt, const std::vector<Float3>& world,
                                    const std::vector<float>& material,
                                    const DrawTarget& target, Float3 light,
                                    uint32_t bounces, bool shadows)
{
    DeviceGeometry traced = upload_accelerated(rt, world);
    DeviceFrame buffer = allocate_frame(rt, target);

    // The tree moved the triangles, so the materials move with them. One float
    // a triangle, permuted by the order the build reports.
    std::vector<float> placed(material.size());
    for (size_t i = 0; i < traced.triangle_order.size(); ++i) {
        placed[i] = material[traced.triangle_order[i]];
    }

    void* indices = rt.myrt_malloc(placed.size() * sizeof(float));
    void* table = rt.myrt_malloc(sizeof(MATERIAL_TABLE));
    rt.myrt_memcpy(indices, placed.data(), placed.size() * sizeof(float),
                   Direction::HostToDevice);
    rt.myrt_memcpy(table, MATERIAL_TABLE, sizeof(MATERIAL_TABLE),
                   Direction::HostToDevice);

    RaytraceStageArgs walk;
    walk.basis = ray_basis(target.camera, target.aspect());
    walk.triangles_offset = rt.myrt_device_offset(traced.world);
    walk.framebuffer_offset = rt.myrt_device_offset(buffer.pixels);
    walk.width = target.width;
    walk.height = target.height;
    walk.triangle_count = traced.triangle_count;
    walk.traversal = Traversal::Bvh;
    walk.stack_depth = traced.bvh_depth;
    walk.bvh_offset = rt.myrt_device_offset(traced.bvh);
    walk.shade_when = ShadeWhen::Deferred;
    walk.bounces = bounces;
    walk.material_offset = rt.myrt_device_offset(indices);
    walk.material_table_offset = rt.myrt_device_offset(table);
    walk.shading.light_position = light;
    walk.shadows = shadows;
    run_raytrace_stage(rt, walk);

    std::vector<Float3> pixels = read_back(rt, buffer);
    release(rt, buffer);
    release(rt, traced);
    return pixels;
}

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
        // Released at the top of the first arc, so the whole flight is bounces
        // rather than one long fall — that fall would set the vertical extent
        // the camera has to hold, and shrink the sphere inside it.
        Float3 at{centre.x - radius * 5.0f, ground + radius * 4.0f, centre.z};
        float rise = 0.0f;
        const float forward = radius * 0.16f;
        const float pull = radius * 0.09f;
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
    // Fitted to the box rather than to a sphere around it. The flight is wide
    // and shallow, so the sphere that holds it is mostly empty and backs the
    // camera off far enough to shrink the ball to a dot.
    const float flight_span =
        0.5f * std::max(flight_hi.x - flight_lo.x, flight_hi.y - flight_lo.y) + radius;
    const float flight_distance =
        flight_span / std::tan(radians(framing.fov_y_degrees) * 0.5f) * 1.08f;

    // The mirror room, built once. Two panels facing each other, two cubes
    // between them, and a floor wide enough to run out of the frame.
    std::vector<Float3> room;
    std::vector<float> room_material;
    if (mirror) {
        // World units of its own, not the loaded mesh's: this scene uses no mesh
        // and tying it to one would make the camera and the room agree only by
        // accident. The proportions are the sample's — two cubes a unit either
        // side of the origin, a ground fifteen across a unit below them, and two
        // mirrors seven out facing each other.
        add_box(room, room_material, {-15.0f, -1.2f, -15.0f}, {15.0f, -1.0f, 15.0f},
                static_cast<float>(GROUND));
        add_box(room, room_material, {-1.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, 0.5f},
                static_cast<float>(FACE_ZERO), true);
        add_box(room, room_material, {0.5f, -0.5f, -0.5f}, {1.5f, 0.5f, 0.5f},
                static_cast<float>(FACE_ZERO), true);
        add_box(room, room_material, {-5.0f, -1.0f, -7.1f}, {5.0f, 4.0f, -6.9f},
                static_cast<float>(MIRROR));
        add_box(room, room_material, {-5.0f, -1.0f, 6.9f}, {5.0f, 4.0f, 7.1f},
                static_cast<float>(MIRROR));

        // Behind the cubes, so the cubes hide its near half and what reaches
        // the camera is the part standing against the far mirror. Refraction
        // reads as a displacement of whatever is behind it, and the mirror is
        // the one thing in this room with enough structure to show one.
        add_pane(room, room_material, static_cast<float>(GLASS), {-2.20f, -1.0f, 2.20f},
                 {2.20f, 1.20f, 2.20f});
    }

    std::vector<std::vector<Float3>> animation;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const float turn =
            6.2831853f * static_cast<float>(frame) / static_cast<float>(frames);

        Camera camera = framing;
        // The mirror room is in world units of its own, so its camera is too —
        // deriving it from the loaded mesh would have the two agree by accident.
        camera.target = thrown   ? flight_centre
                        : mirror ? Float3{0.0f, 0.2f, 0.0f}
                                 : centre;
        camera.eye =
            thrown ? Float3{flight_centre.x, flight_centre.y + flight_radius * 0.30f,
                            flight_centre.z + flight_distance}
            : mirror
                ? Float3{std::sin(turn) * 1.8f, 0.9f, -2.6f - std::cos(turn) * 0.7f}
                : Float3{centre.x + std::sin(turn) * distance, centre.y + radius * 0.6f,
                         centre.z + std::cos(turn) * distance};

        if (mirror) {
            // Named rather than derived from a framing distance, because this
            // camera stands among the objects instead of outside them: a near
            // plane placed where the subject would be if the camera had backed
            // off puts the cubes behind it and leaves a frame of sky.
            camera.near_z = 0.1f;
            camera.far_z = 60.0f;
        } else {
            const float framed = thrown ? flight_distance : distance;
            const float held = thrown ? flight_radius : radius;
            camera.near_z = std::max(framed - held, framed * 0.01f);
            camera.far_z = framed + held * 2.0f;
        }
        const DrawTarget target{size, size, camera};

        MyGPURuntime rt(1u << 27);

        // The scheduler's guard is a flat number of cycles, which a large
        // enough grid reaches for an honest reason — a thousand pixels a side
        // with shadow rays is twelve traversals over a million pixels. Scaled
        // by the pixels asked for, which lands on the default at the sizes
        // these scenes are usually run at and still bounds a kernel that does
        // not end.
        rt.myrt_cycle_budget(static_cast<uint64_t>(size) * size * 4096ull);

        // How the light arrives. Named a scene at a time: the mirror room is
        // in world units of its own, and the throw wants the ball's shadow to
        // land where the camera can see it.
        const Float3 light =
            mirror ? Float3{0.55f, 1.0f, 0.45f} : Float3{0.35f, 1.0f, 0.5f};
        const bool shadows = args.flag("shadows", 1) != 0;

        if (mirror) {
            animation.push_back(trace_materials(rt, room, room_material, target, light,
                                                bounces, shadows));
            if (frame % 12 == 0) {
                std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                            rt.divergence_rate() * 100.0);
            }
            continue;
        }

        if (thrown) {
            // The throw moves the geometry and leaves everything else alone,
            // which is the cheaper of the two ways to animate: no matrix reaches
            // the device and no stage is asked to apply one.
            std::vector<Float3> placed = world;
            const Float3 to = thrown_at(frame);
            for (Float3& v : placed) {
                v = Float3{v.x - centre.x + to.x, v.y - centre.y + to.y,
                           v.z - centre.z + to.z};
            }
            std::vector<float> which(placed.size() / 3, static_cast<float>(BALL));

            // The floor the arcs are solved against, drawn so the shrinking
            // reads as a bounce rather than as a drift, and so the ball has
            // somewhere to cast onto. It does not move, so it goes in after the
            // sphere is translated.
            const float y = ground - radius;
            const float x0 = flight_lo.x - radius * 2.0f;
            const float x1 = flight_hi.x + radius * 2.0f;
            const float z0 = centre.z - radius * 8.0f;
            const float z1 = centre.z + radius * 8.0f;
            placed.push_back(Float3{x0, y, z0});
            placed.push_back(Float3{x1, y, z1});
            placed.push_back(Float3{x1, y, z0});
            placed.push_back(Float3{x0, y, z0});
            placed.push_back(Float3{x0, y, z1});
            placed.push_back(Float3{x1, y, z1});
            which.push_back(static_cast<float>(GROUND));
            which.push_back(static_cast<float>(GROUND));

            animation.push_back(
                trace_materials(rt, placed, which, target, light, bounces, shadows));
            if (frame % 12 == 0) {
                std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                            rt.divergence_rate() * 100.0);
            }
            continue;
        }

        if (!cube) {
            DeviceGeometry traced = upload_accelerated(rt, world);
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
    // One frame is a picture and several are an animation, so a run writes one
    // or the other. It also lets the two be taken at different sizes without
    // the second overwriting the first: a still can afford an hour a frame and
    // an animation cannot.
    if (frames == 1) {
        std::vector<float> flat;
        for (const Float3& pixel : animation.front()) {
            flat.push_back(pixel.x);
            flat.push_back(pixel.y);
            flat.push_back(pixel.z);
        }
        const std::string still = args.images_dir() + name + ".ppm";
        write_ppm(still, flat, size, size);
        std::printf("\nwrote %s\n", still.c_str());
    } else {
        const std::string gif = args.images_dir() + name + ".gif";
        write_gif(gif, animation, size, size, 4);
        std::printf("\nwrote %s\n", gif.c_str());
    }
    return 0;
}
