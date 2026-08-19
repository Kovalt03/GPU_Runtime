// A mesh bound to a skeleton, drawn once a frame, written out as an animation.
//
// The last thing in the pipeline that a single image cannot show. Everything
// before it could be checked by looking at one PPM; a pose can only be checked
// against the poses either side of it.
//
// The skeleton and the motion come from a Biovision Hierarchy file, the mesh
// from an OBJ, and the binding from neither — assets/ holds no rigged mesh, so
// every vertex is bound to the nearest bone of the rest pose. That is what an
// auto-rigger does and it is stated rather than hidden: the weights are a rule's
// and not an artist's.
//
//   ./build/kernels/skinned_render                       result/images/skinned.gif
//   ./build/kernels/skinned_render assets/arm.bvh assets/sphere.obj
//   ./build/kernels/skinned_render --width 128 --height 96
//   ./build/kernels/skinned_render --route raytrace
//
// A rig that already says which bone a vertex follows can be pointed at instead,
// one OBJ a bone sharing a vertex pool:
//
//   ./build/kernels/skinned_render --rig <dir> --motion <dir>/walk.bvh
//
// A rig of any size wants --route raytrace. The walk is one thread a pixel
// against every triangle, so it grows with both and a character mesh is tens of
// thousands of triangles: 124 seconds a frame against the tree's 3, measured on
// 25,342. The tree costs something the walk does not, though — the ray tracer
// has no vertex stage, so a skinned mesh has to be posed before it arrives, and
// the skinning then happens on the host rather than on the device.
//
// Nothing from such a directory is committed here. The rigs this was developed
// against are licensed for use and not for redistribution, which is the ordinary
// case for character assets and the reason the default is a mesh this repository
// owns.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gif.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/vertex.hpp"
#include "ppm.hpp"
#include "skeleton.hpp"
#include "skin.hpp"

namespace {

// The mesh, and which bone each of its vertices follows.
struct Rigged {
    std::vector<Float3> vertices;
    std::vector<uint32_t> bone;
    std::string how;
};

// One OBJ a bone, all sharing a vertex pool and partitioning the triangles —
// what a rig exported per bone looks like. The partition is over triangles, not
// vertices, so the list is flattened: a vertex on a seam is duplicated and each
// copy takes the bone of the triangle it belongs to, which is what the rig
// actually says.
Rigged load_rig(const std::string& dir, const Skeleton& skeleton)
{
    Rigged rigged;
    rigged.how = "per-bone meshes";

    std::vector<Float3> pool;
    for (uint32_t j = 0; j < skeleton.joint_count(); ++j) {
        const Joint& joint = skeleton.joints[j];
        if (joint.end_site) {
            continue;
        }

        // Joint names carry the exporter's prefix and the files rarely do.
        const size_t colon = joint.name.find(':');
        const std::string plain =
            colon == std::string::npos ? joint.name : joint.name.substr(colon + 1);

        Mesh part;
        try {
            part = load_obj(dir + "/" + plain + ".obj");
        } catch (const std::runtime_error&) {
            continue;
        }
        if (pool.empty()) {
            pool = part.vertices;
        }

        for (const uint32_t index : part.indices) {
            if (index >= pool.size()) {
                throw std::runtime_error("skinned_render: " + plain +
                                         ".obj indexes past the shared vertex pool");
            }
            rigged.vertices.push_back(pool[index]);
            rigged.bone.push_back(j);
        }
    }

    if (rigged.vertices.empty()) {
        throw std::runtime_error("skinned_render: no bone meshes found in " + dir);
    }
    return rigged;
}

std::vector<float> flatten(const std::vector<Float4x4>& matrices)
{
    std::vector<float> flat;
    flat.reserve(matrices.size() * 16);
    for (const Float4x4& m : matrices) {
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) {
                flat.push_back(m.at(r, c));
            }
        }
    }
    return flat;
}

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const uint32_t width = args.flag("width", 96);
    const uint32_t height = args.flag("height", 96);
    const std::string rig = args.text_flag("rig", "");
    const bool trace = args.text_flag("route", "walk") == "raytrace";

    const std::string assets = std::string(GPURT_ASSETS_DIR);
    const std::string motion_path =
        args.text_flag("motion", args.text(0, assets + "/arm.bvh"));
    const Motion motion = load_bvh_motion(motion_path);

    Rigged rigged;
    if (!rig.empty()) {
        rigged = load_rig(rig, motion.skeleton);
    } else {
        rigged.vertices = load_obj(args.text(1, assets + "/grid.obj")).flattened();
        rigged.how = "bound to the nearest bone";
    }

    // Into the skeleton's space before anything is bound: a mesh scaled but not
    // moved binds to whichever two or three joints it happens to sit beside.
    const Float4x4 fit = fit_to_rest(rigged.vertices, motion);
    for (Float3& v : rigged.vertices) {
        const Float4 moved = transform(fit, v, 1.0f);
        v = Float3{moved.x, moved.y, moved.z};
    }

    const Skin skin = rigged.bone.empty() ? bind_nearest(rigged.vertices, motion)
                                          : bind_given(rigged.bone, motion);

    // Framed on the rest pose, so the camera does not have to be told about a
    // motion it has not seen.
    Float3 lo = rigged.vertices[0];
    Float3 hi = rigged.vertices[0];
    for (const Float3& v : rigged.vertices) {
        lo = Float3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Float3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
    const Float3 centre{(lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f};
    const Float3 span = hi - lo;
    const float reach = std::max(span.x, std::max(span.y, span.z));

    Camera camera;
    camera.target = centre;
    camera.eye = Float3{centre.x, centre.y, centre.z + reach * 2.2f};
    camera.far_z = reach * 10.0f;
    const DrawTarget target{width, height, camera};

    std::printf("\n[ANIM] %u joints, %zu vertices, %s\n", motion.skeleton.joint_count(),
                rigged.vertices.size(), rigged.how.c_str());
    std::printf("  %u frames at %.4fs, %ux%u\n\n", motion.frame_count(),
                motion.frame_time, width, height);

    std::vector<std::vector<Float3>> animation;
    for (uint32_t frame = 0; frame < motion.frame_count(); ++frame) {
        const std::vector<float> palette = flatten(skin_palette(motion, skin, frame));
        std::vector<float> bones;
        bones.reserve(skin.bone.size());
        for (const uint32_t b : skin.bone) {
            bones.push_back(static_cast<float>(b));
        }

        MyGPURuntime rt(1u << 28);

        if (trace) {
            // Posed on the host, because this route has no vertex stage to pose
            // it with. What it buys is that a pixel meets the triangles a tree
            // says it might rather than all of them.
            std::vector<Float3> posed;
            posed.reserve(rigged.vertices.size());
            for (size_t v = 0; v < rigged.vertices.size(); ++v) {
                const Float4x4& m =
                    *reinterpret_cast<const Float4x4*>(&palette[skin.bone[v] * 16]);
                const Float4 at = transform(m, rigged.vertices[v], 1.0f);
                posed.push_back(Float3{at.x, at.y, at.z});
            }

            DeviceGeometry traced = upload_accelerated(rt, posed);
            DeviceFrame frame_buffer = allocate_frame(rt, target);
            animation.push_back(draw_raytrace(rt, traced, frame_buffer, target));
            std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                        rt.divergence_rate() * 100.0);
            release(rt, frame_buffer);
            release(rt, traced);
            continue;
        }

        DeviceGeometry geometry = upload(rt, rigged.vertices);
        DeviceFrame buffer = allocate_frame(rt, target);
        void* device_bones = rt.myrt_malloc(bones.size() * sizeof(float));
        void* device_palette = rt.myrt_malloc(palette.size() * sizeof(float));
        rt.myrt_memcpy(device_bones, bones.data(), bones.size() * sizeof(float),
                       Direction::HostToDevice);
        rt.myrt_memcpy(device_palette, palette.data(), palette.size() * sizeof(float),
                       Direction::HostToDevice);

        // Cleared before anything is drawn: DepthUse::Test reads the buffer
        // rather than owning it, so an uncleared one starts at zero and refuses
        // every fragment. Nothing fails — the frame simply comes out empty.
        clear_frame(rt, buffer, target);

        VertexStageArgs pass1;
        pass1.view_projection = target.camera.view_projection(target.aspect());
        pass1.world_offset = rt.myrt_device_offset(geometry.world);
        pass1.screen_offset = rt.myrt_device_offset(geometry.screen);
        pass1.vertex_count = geometry.vertex_count;
        pass1.width = width;
        pass1.height = height;
        pass1.shade = skinning_shader(rt.myrt_device_offset(device_bones),
                                      rt.myrt_device_offset(device_palette));
        run_vertex_stage(rt, pass1);
        rt.myrt_sync(false);

        RasterStageArgs pass2;
        pass2.screen_offset = pass1.screen_offset;
        pass2.framebuffer_offset = rt.myrt_device_offset(buffer.pixels);
        pass2.depth_offset = rt.myrt_device_offset(buffer.depth);
        pass2.depth = DepthUse::Test;
        pass2.width = width;
        pass2.height = height;
        pass2.triangle_count = geometry.triangle_count;
        run_raster_stage(rt, pass2);

        animation.push_back(read_back(rt, buffer));
        std::printf("  frame %2u  divergence %5.2f%%\n", frame,
                    rt.divergence_rate() * 100.0);
    }

    const std::string gif = args.images_dir() + "skinned.gif";
    write_gif(gif, animation, width, height,
              static_cast<uint32_t>(motion.frame_time * 100.0f + 0.5f));

    // The first frame as a PPM too: a still is what a diff can read, and every
    // other kernel here writes one.
    std::vector<float> flat;
    for (const Float3& pixel : animation.front()) {
        flat.push_back(pixel.x);
        flat.push_back(pixel.y);
        flat.push_back(pixel.z);
    }
    const std::string still = args.images_dir() + "skinned.ppm";
    write_ppm(still, flat, width, height);

    std::printf("\nwrote %s and %s\n", gif.c_str(), still.c_str());
    return 0;
}
