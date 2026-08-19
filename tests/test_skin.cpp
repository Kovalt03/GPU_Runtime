#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/vertex.hpp"
#include "skeleton.hpp"
#include "skin.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

std::string asset(const std::string& name)
{
    return std::string(GPURT_ASSETS_DIR) + "/" + name;
}

// A mesh fitted to a skeleton, which is the state everything below starts from.
struct Bound {
    Motion motion;
    std::vector<Float3> vertices;
    Skin skin;
};

Bound bind(const std::string& bvh, const std::string& obj)
{
    Bound bound;
    bound.motion = load_bvh_motion(asset(bvh));
    bound.vertices = load_obj(asset(obj)).flattened();

    const Float4x4 fit = fit_to_rest(bound.vertices, bound.motion);
    for (Float3& v : bound.vertices) {
        const Float4 moved = transform(fit, v, 1.0f);
        v = Float3{moved.x, moved.y, moved.z};
    }
    bound.skin = bind_nearest(bound.vertices, bound.motion);
    return bound;
}

// Where the host says a vertex goes, which the device has to agree with.
Float3 skinned(const Bound& bound, const std::vector<Float4x4>& palette, size_t vertex)
{
    const Float4 moved =
        transform(palette[bound.skin.bone[vertex]], bound.vertices[vertex], 1.0f);
    return Float3{moved.x, moved.y, moved.z};
}

}  // namespace

TEST(Skin, TheBindPoseLeavesEveryVertexWhereItWas)
{
    // The palette at the frame the skin was bound in is the identity, joint by
    // joint: the pose composed with its own inverse. Nothing should move, and
    // this is what says the inverse-rest half is right — a wrong one still
    // animates, just from the wrong starting place.
    const Bound bound = bind("arm.bvh", "grid.obj");
    const std::vector<Float4x4> palette = skin_palette(bound.motion, bound.skin, 0);

    for (size_t v = 0; v < bound.vertices.size(); ++v) {
        const Float3 at = skinned(bound, palette, v);
        EXPECT_NEAR(at.x, bound.vertices[v].x, 1e-3f) << "vertex " << v;
        EXPECT_NEAR(at.y, bound.vertices[v].y, 1e-3f) << "vertex " << v;
        EXPECT_NEAR(at.z, bound.vertices[v].z, 1e-3f) << "vertex " << v;
    }
}

TEST(Skin, FittingIsWhatSpreadsAMeshAcrossTheJoints)
{
    // Scaling a mesh without moving it leaves it wherever it was modelled, and
    // every vertex then binds to whichever two or three joints happen to be
    // nearest. The frames still animate, which is why the thing to check is bone
    // coverage rather than whether anything moved.
    const Motion motion = load_bvh_motion(asset("arm.bvh"));
    const std::vector<Float3> mesh = load_obj(asset("grid.obj")).flattened();

    const auto joints_used = [&](const std::vector<Float3>& vertices) {
        const Skin skin = bind_nearest(vertices, motion);
        std::vector<bool> seen(motion.skeleton.joint_count(), false);
        for (const uint32_t bone : skin.bone) {
            seen[bone] = true;
        }
        uint32_t used = 0;
        for (const bool touched : seen) {
            used += touched ? 1u : 0u;
        }
        return used;
    };

    std::vector<Float3> fitted = mesh;
    const Float4x4 fit = fit_to_rest(fitted, motion);
    for (Float3& v : fitted) {
        const Float4 moved = transform(fit, v, 1.0f);
        v = Float3{moved.x, moved.y, moved.z};
    }

    EXPECT_GT(joints_used(fitted), joints_used(mesh));
    EXPECT_GE(joints_used(fitted), 3u) << "the fit did not spread the mesh";
}

TEST(Skin, TheDeviceMovesAVertexWhereTheHostSaysItGoes)
{
    // The whole chain: a bone index read from a buffer, a matrix read from the
    // palette with the lane-varying wide load, and one transform. Held against
    // the same arithmetic on the host, which is written from the palette rather
    // than from the shader.
    const Bound bound = bind("arm.bvh", "grid.obj");
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};

    for (uint32_t frame = 0; frame < bound.motion.frame_count(); ++frame) {
        const std::vector<Float4x4> palette =
            skin_palette(bound.motion, bound.skin, frame);

        std::vector<float> flat_palette;
        for (const Float4x4& m : palette) {
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 4; ++c) {
                    flat_palette.push_back(m.at(r, c));
                }
            }
        }
        std::vector<float> bones;
        for (const uint32_t b : bound.skin.bone) {
            bones.push_back(static_cast<float>(b));
        }

        MyGPURuntime rt(1u << 24);
        void* world = rt.myrt_malloc(bound.vertices.size() * sizeof(Float3));
        void* screen = rt.myrt_malloc(bound.vertices.size() * SCREEN_VERTEX_BYTES);
        void* device_bones = rt.myrt_malloc(bones.size() * sizeof(float));
        void* device_palette = rt.myrt_malloc(flat_palette.size() * sizeof(float));
        rt.myrt_memcpy(world, bound.vertices.data(),
                       bound.vertices.size() * sizeof(Float3), Direction::HostToDevice);
        rt.myrt_memcpy(device_bones, bones.data(), bones.size() * sizeof(float),
                       Direction::HostToDevice);
        rt.myrt_memcpy(device_palette, flat_palette.data(),
                       flat_palette.size() * sizeof(float), Direction::HostToDevice);

        VertexStageArgs pass1;
        pass1.view_projection = target.camera.view_projection(target.aspect());
        pass1.world_offset = rt.myrt_device_offset(world);
        pass1.screen_offset = rt.myrt_device_offset(screen);
        pass1.vertex_count = static_cast<uint32_t>(bound.vertices.size());
        pass1.width = WIDTH;
        pass1.height = HEIGHT;
        pass1.shade = skinning_shader(rt.myrt_device_offset(device_bones),
                                      rt.myrt_device_offset(device_palette));
        run_vertex_stage(rt, pass1);
        rt.myrt_sync(false);

        // Pass 1 writes screen coordinates, so the check is against what the
        // host's own projection makes of the same skinned position.
        std::vector<float> wrote(bound.vertices.size() * SCREEN_VERTEX_FLOATS, 0.0f);
        rt.myrt_memcpy(wrote.data(), screen, wrote.size() * sizeof(float),
                       Direction::DeviceToHost);

        for (size_t v = 0; v < bound.vertices.size(); ++v) {
            const Float3 at = skinned(bound, palette, v);
            const Float4 clip = transform(pass1.view_projection, at, 1.0f);
            const float sx = (clip.x / clip.w + 1.0f) * (WIDTH * 0.5f);
            const float sy = (1.0f - clip.y / clip.w) * (HEIGHT * 0.5f);

            EXPECT_NEAR(wrote[v * SCREEN_VERTEX_FLOATS + 0], sx, 1e-2f)
                << "frame " << frame << ", vertex " << v;
            EXPECT_NEAR(wrote[v * SCREEN_VERTEX_FLOATS + 1], sy, 1e-2f)
                << "frame " << frame << ", vertex " << v;
        }
    }
}

TEST(Skin, SkinningIsAWideLoadAndAMatrixVectorProduct)
{
    // Five instructions a vertex, and the shape of them is the point: the bone
    // index is one scalar load, the matrix is one V_LD_GLOBAL_MAT4_F32, and the
    // move is one V_MATVEC_MAT4_F32. Sixteen scalar loads would do the same work
    // and cost sixteen line lookups where this costs one.
    const Bound bound = bind("arm.bvh", "grid.obj");
    const std::vector<Float4x4> palette = skin_palette(bound.motion, bound.skin, 1);

    IRBuilder k;
    Vertex vertex;
    vertex.out = k.vec3();
    vertex.position = k.vec3();
    vertex.index = k.constant(0.0f);
    vertex.instance = k.constant(0.0f);
    skinning_shader(64, 4096)(k, vertex);
    const Program program = k.build();

    uint32_t wide_loads = 0;
    uint32_t transforms = 0;
    uint32_t scalar_loads = 0;
    for (const Instruction& instruction : program) {
        wide_loads += instruction.op == Opcode::V_LD_GLOBAL_MAT4_F32 ? 1u : 0u;
        transforms += instruction.op == Opcode::V_MATVEC_MAT4_F32 ? 1u : 0u;
        scalar_loads += instruction.op == Opcode::V_LD_GLOBAL_F32 ? 1u : 0u;
    }
    EXPECT_EQ(wide_loads, 1u);
    EXPECT_EQ(transforms, 1u);
    EXPECT_EQ(scalar_loads, 1u) << "the bone index is one load, not sixteen";
    EXPECT_EQ(palette.size(), bound.motion.skeleton.joint_count());
}

TEST(Skin, ABindingItCannotHonourIsRefused)
{
    const Motion motion = load_bvh_motion(asset("arm.bvh"));
    EXPECT_THROW(bind_nearest({}, motion), std::runtime_error);
    EXPECT_THROW(bind_given({}, motion), std::runtime_error);
    EXPECT_THROW(fit_to_rest({}, motion), std::runtime_error);

    // A vertex bound to a joint the skeleton does not have.
    EXPECT_THROW(bind_given({0u, motion.skeleton.joint_count()}, motion),
                 std::runtime_error);
}
