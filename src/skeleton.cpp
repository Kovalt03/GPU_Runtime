#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "skeleton.hpp"

namespace {

Channel channel_from(const std::string& name)
{
    if (name == "Xposition") {
        return Channel::Xposition;
    }
    if (name == "Yposition") {
        return Channel::Yposition;
    }
    if (name == "Zposition") {
        return Channel::Zposition;
    }
    if (name == "Xrotation") {
        return Channel::Xrotation;
    }
    if (name == "Yrotation") {
        return Channel::Yrotation;
    }
    if (name == "Zrotation") {
        return Channel::Zrotation;
    }
    throw std::runtime_error("load_bvh_motion: unknown channel " + name);
}

Float4x4 rotation_about(Channel axis, float degrees)
{
    const float r = radians(degrees);
    const float c = std::cos(r);
    const float s = std::sin(r);

    Float4x4 m = Float4x4::identity();
    switch (axis) {
    case Channel::Xrotation:
        m.at(1, 1) = c;
        m.at(1, 2) = -s;
        m.at(2, 1) = s;
        m.at(2, 2) = c;
        break;
    case Channel::Yrotation:
        m.at(0, 0) = c;
        m.at(0, 2) = s;
        m.at(2, 0) = -s;
        m.at(2, 2) = c;
        break;
    case Channel::Zrotation:
        m.at(0, 0) = c;
        m.at(0, 1) = -s;
        m.at(1, 0) = s;
        m.at(1, 1) = c;
        break;
    default: break;
    }
    return m;
}

// One joint's subtree. Recursive because the file is, and a skeleton is fifty
// joints deep at most — the recursion is the parser's shape rather than a cost.
void read_joint(std::istream& in, Skeleton& skeleton, int parent, bool end_site)
{
    const auto expect = [&](const std::string& want) {
        std::string got;
        if (!(in >> got) || got != want) {
            throw std::runtime_error("load_bvh_motion: expected " + want + ", found " +
                                     got);
        }
    };

    Joint joint;
    joint.parent = parent;
    joint.end_site = end_site;
    if (end_site) {
        joint.name = parent >= 0 ? skeleton.joints[parent].name + "_End" : "_End";
    } else if (!(in >> joint.name)) {
        throw std::runtime_error("load_bvh_motion: a joint with no name");
    }

    expect("{");
    expect("OFFSET");
    if (!(in >> joint.offset.x >> joint.offset.y >> joint.offset.z)) {
        throw std::runtime_error("load_bvh_motion: an OFFSET short of three numbers");
    }

    const int self = static_cast<int>(skeleton.joints.size());
    joint.channel_base = skeleton.channel_count();
    if (!end_site) {
        expect("CHANNELS");
        uint32_t count = 0;
        if (!(in >> count) || count > MAX_CHANNELS) {
            throw std::runtime_error("load_bvh_motion: " + std::to_string(count) +
                                     " channels on one joint");
        }
        joint.channel_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            std::string name;
            in >> name;
            joint.channels[i] = channel_from(name);
        }
    }
    skeleton.joints.push_back(joint);

    // Children after the push, so that a parent always precedes its children and
    // one forward pass can compose the world matrices.
    std::string token;
    while (in >> token) {
        if (token == "}") {
            return;
        }
        if (token == "JOINT") {
            read_joint(in, skeleton, self, false);
        } else if (token == "End") {
            std::string site;
            in >> site;  // "Site"
            read_joint(in, skeleton, self, true);
        } else {
            throw std::runtime_error("load_bvh_motion: unexpected " + token +
                                     " inside a joint");
        }
    }
    throw std::runtime_error("load_bvh_motion: a joint that never closed");
}

}  // namespace

uint32_t Skeleton::joint_count() const
{
    return static_cast<uint32_t>(joints.size());
}

uint32_t Skeleton::channel_count() const
{
    uint32_t total = 0;
    for (const Joint& joint : joints) {
        total += joint.channel_count;
    }
    return total;
}

int Skeleton::find(const std::string& name) const
{
    for (uint32_t i = 0; i < joints.size(); ++i) {
        if (joints[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

uint32_t Motion::frame_count() const
{
    const uint32_t per_frame = skeleton.channel_count();
    return per_frame == 0 ? 0 : static_cast<uint32_t>(channels.size() / per_frame);
}

std::vector<Float4x4> Motion::pose(uint32_t frame) const
{
    if (frame >= frame_count()) {
        throw std::runtime_error("Motion::pose: frame " + std::to_string(frame) + " of " +
                                 std::to_string(frame_count()));
    }

    const uint32_t per_frame = skeleton.channel_count();
    const float* values = &channels[static_cast<size_t>(frame) * per_frame];

    std::vector<Float4x4> world(skeleton.joints.size());
    for (uint32_t i = 0; i < skeleton.joints.size(); ++i) {
        const Joint& joint = skeleton.joints[i];

        // Offset first, then the position channels if there are any: the file
        // gives a translation relative to where the offset already put it.
        Float4x4 local = Float4x4::identity();
        local.at(0, 3) = joint.offset.x;
        local.at(1, 3) = joint.offset.y;
        local.at(2, 3) = joint.offset.z;

        for (uint32_t c = 0; c < joint.channel_count; ++c) {
            const float v = values[joint.channel_base + c];
            switch (joint.channels[c]) {
            case Channel::Xposition: local.at(0, 3) += v; break;
            case Channel::Yposition: local.at(1, 3) += v; break;
            case Channel::Zposition: local.at(2, 3) += v; break;

            // Composed in the order the file lists them, which is what the
            // format means by a channel order — assuming ZXY would put a limb
            // somewhere plausible on a file that said otherwise.
            case Channel::Xrotation:
            case Channel::Yrotation:
            case Channel::Zrotation:
                local = local * rotation_about(joint.channels[c], v);
                break;
            }
        }

        world[i] =
            joint.parent < 0 ? local : world[static_cast<uint32_t>(joint.parent)] * local;
    }
    return world;
}

Motion load_bvh_motion(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_bvh_motion: cannot open " + path);
    }

    std::string token;
    if (!(in >> token) || token != "HIERARCHY") {
        throw std::runtime_error("load_bvh_motion: " + path +
                                 " does not start with "
                                 "HIERARCHY");
    }
    if (!(in >> token) || token != "ROOT") {
        throw std::runtime_error("load_bvh_motion: no ROOT joint in " + path);
    }

    Motion motion;
    read_joint(in, motion.skeleton, -1, false);

    if (!(in >> token) || token != "MOTION") {
        throw std::runtime_error("load_bvh_motion: no MOTION section in " + path);
    }

    uint32_t frames = 0;
    in >> token;  // "Frames:"
    if (token != "Frames:" || !(in >> frames)) {
        throw std::runtime_error("load_bvh_motion: no frame count in " + path);
    }
    in >> token >> token;  // "Frame" "Time:"
    if (!(in >> motion.frame_time)) {
        throw std::runtime_error("load_bvh_motion: no frame time in " + path);
    }

    const size_t per_frame = motion.skeleton.channel_count();
    motion.channels.resize(static_cast<size_t>(frames) * per_frame);
    for (size_t i = 0; i < motion.channels.size(); ++i) {
        if (!(in >> motion.channels[i])) {
            throw std::runtime_error("load_bvh_motion: " + path + " promised " +
                                     std::to_string(frames) + " frames of " +
                                     std::to_string(per_frame) +
                                     " and ran out at value " + std::to_string(i));
        }
    }
    return motion;
}
