#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "ir_builder.hpp"
#include "skin.hpp"
#include "thread.hpp"  // MAT4_REGISTERS

namespace {

// Distance from a point to the segment between two joints, squared.
float distance_to_segment(Float3 p, Float3 a, Float3 b)
{
    const Float3 ab = b - a;
    const float length = dot(ab, ab);

    // A joint whose child sits on top of it degenerates to a point, which the
    // hands of a real skeleton do constantly.
    const float t =
        length < 1e-12f ? 0.0f : std::min(1.0f, std::max(0.0f, dot(p - a, ab) / length));
    const Float3 nearest = a + ab * t;
    return dot(p - nearest, p - nearest);
}

}  // namespace

uint32_t Skin::vertex_count() const
{
    return static_cast<uint32_t>(bone.size());
}

Skin bind_given(const std::vector<uint32_t>& bone, const Motion& motion)
{
    if (bone.empty()) {
        throw std::runtime_error("bind_given: a mesh with no vertices");
    }

    Skin skin;
    skin.bone = bone;

    // The rest pose is frame zero, and the inverse of it is what takes a vertex
    // into its bone's space. Computed once: it is the bind pose, and a bind pose
    // that changed per frame would not be one.
    const std::vector<Float4x4> rest = motion.pose(0);
    skin.inverse_rest.reserve(rest.size());
    for (const Float4x4& joint : rest) {
        skin.inverse_rest.push_back(inverse(joint));
    }

    for (const uint32_t index : bone) {
        if (index >= rest.size()) {
            throw std::runtime_error("bind_given: vertex bound to joint " +
                                     std::to_string(index) + " of " +
                                     std::to_string(rest.size()));
        }
    }
    return skin;
}

Float4x4 fit_to_rest(const std::vector<Float3>& vertices, const Motion& motion)
{
    if (vertices.empty()) {
        throw std::runtime_error("fit_to_rest: a mesh with no vertices");
    }

    Float3 lo = vertices[0];
    Float3 hi = vertices[0];
    for (const Float3& v : vertices) {
        lo = Float3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Float3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }

    const std::vector<Float4x4> rest = motion.pose(0);
    Float3 slo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
               std::numeric_limits<float>::max()};
    Float3 shi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
               -std::numeric_limits<float>::max()};
    for (const Float4x4& joint : rest) {
        const Float3 p{joint.at(0, 3), joint.at(1, 3), joint.at(2, 3)};
        slo = Float3{std::min(slo.x, p.x), std::min(slo.y, p.y), std::min(slo.z, p.z)};
        shi = Float3{std::max(shi.x, p.x), std::max(shi.y, p.y), std::max(shi.z, p.z)};
    }

    // Uniform, and on the tallest axis: a skeleton is long in one direction and
    // a per-axis fit would stretch a mesh to match a stick.
    const Float3 mesh_span = hi - lo;
    const Float3 rest_span = shi - slo;
    const float longest = std::max(rest_span.x, std::max(rest_span.y, rest_span.z));
    const float mesh_longest = std::max(mesh_span.x, std::max(mesh_span.y, mesh_span.z));
    const float scale = mesh_longest < 1e-9f ? 1.0f : longest / mesh_longest;

    Float4x4 fit = Float4x4::identity();
    fit.at(0, 0) = scale;
    fit.at(1, 1) = scale;
    fit.at(2, 2) = scale;
    fit.at(0, 3) = (slo.x + shi.x) * 0.5f - (lo.x + hi.x) * 0.5f * scale;
    fit.at(1, 3) = (slo.y + shi.y) * 0.5f - (lo.y + hi.y) * 0.5f * scale;
    fit.at(2, 3) = (slo.z + shi.z) * 0.5f - (lo.z + hi.z) * 0.5f * scale;
    return fit;
}

Skin bind_nearest(const std::vector<Float3>& vertices, const Motion& motion)
{
    if (vertices.empty()) {
        throw std::runtime_error("bind_nearest: a mesh with no vertices");
    }

    const std::vector<Float4x4> rest = motion.pose(0);
    const auto at = [&](uint32_t joint) {
        return Float3{rest[joint].at(0, 3), rest[joint].at(1, 3), rest[joint].at(2, 3)};
    };

    std::vector<uint32_t> bone(vertices.size(), 0);
    for (size_t v = 0; v < vertices.size(); ++v) {
        const Float3 p = vertices[v];
        float best = std::numeric_limits<float>::max();

        for (uint32_t j = 0; j < rest.size(); ++j) {
            // A joint's segment runs to its first child. A joint with none — the
            // end sites, and any leaf — is a point, and the clamp above handles
            // it without a special case here.
            Float3 tip = at(j);
            for (uint32_t c = j + 1; c < rest.size(); ++c) {
                if (motion.skeleton.joints[c].parent == static_cast<int>(j)) {
                    tip = at(c);
                    break;
                }
            }

            const float d = distance_to_segment(p, at(j), tip);
            if (d < best) {
                best = d;
                bone[v] = j;
            }
        }
    }
    return bind_given(bone, motion);
}

std::vector<Float4x4> skin_palette(const Motion& motion, const Skin& skin, uint32_t frame)
{
    const std::vector<Float4x4> now = motion.pose(frame);
    if (now.size() != skin.inverse_rest.size()) {
        throw std::runtime_error("skin_palette: the skin was bound to another skeleton");
    }

    std::vector<Float4x4> palette;
    palette.reserve(now.size());
    for (size_t j = 0; j < now.size(); ++j) {
        palette.push_back(now[j] * skin.inverse_rest[j]);
    }
    return palette;
}

VertexFn skinning_shader(size_t bone_offset, size_t palette_offset)
{
    return [bone_offset, palette_offset](IRBuilder& k, const Vertex& v) {
        // Which joint this vertex follows, from a buffer of its own.
        const Reg<Scalar> bone =
            k.load(k.mul(v.index, k.constant(4.0f)), static_cast<float>(bone_offset));

        // Its matrix, in one instruction. Sixteen scalar loads would cost the
        // same under a flat charge and sixteen times as many line lookups under
        // a coalesced one — and the constant window is no help, the address
        // differing by lane.
        const Reg<Mat4> matrix = k.load_mat4(
            k.mul(bone, k.constant(static_cast<float>(MAT4_REGISTERS * sizeof(float)))),
            static_cast<float>(palette_offset));

        const Reg<Vec4> at = k.vec4();
        for (uint32_t c = 0; c < 3; ++c) {
            k.copy_into(at.component(c), v.position.component(c));
        }
        k.set(at.component(3), 1.0f);

        const Reg<Vec4> moved = k.transform(matrix, at);
        for (uint32_t c = 0; c < 3; ++c) {
            k.copy_into(v.out.component(c), moved.component(c));
        }
    };
}
