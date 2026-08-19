#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "math3d.hpp"

// A skeleton and the motion that moves it, read from a Biovision Hierarchy file.
//
// **Not the BVH in bvh.hpp.** That one is a bounding volume hierarchy, the tree
// a ray walks; this is Biovision Hierarchy, a text format holding a joint tree
// and a channel value per joint per frame. The two abbreviations collide in
// graphics generally and would collide in this repository doubly, so the header
// is named after what it holds rather than after the file it reads.
//
// Host code with no ISA in it, beside mesh.hpp for the same reason: a skeleton
// is something the geometry arrives with.

// What a frame supplies for one joint, in the order the file names them.
enum class Channel : uint8_t {
    Xposition,
    Yposition,
    Zposition,
    Xrotation,
    Yrotation,
    Zrotation,
};

// Rotation is applied in the order the channels are listed, which for every file
// seen so far is Z then X then Y — and is a property of the file rather than a
// convention, so it is read rather than assumed.
inline constexpr uint32_t MAX_CHANNELS = 6;

struct Joint {
    std::string name;

    // Index into Skeleton::joints, or -1 for the root. Parents always come
    // first, so a single forward pass composes every world matrix.
    int parent = -1;

    // Where this joint sits in its parent's space, at rest.
    Float3 offset;

    // Which channels a frame supplies for it, and where they start in a frame's
    // run of values.
    Channel channels[MAX_CHANNELS];
    uint32_t channel_count = 0;
    uint32_t channel_base = 0;

    // An End Site has an offset and no channels. Kept because it is where a
    // limb actually ends, which is what a mesh bound to the last real joint
    // needs in order to know how long that joint is.
    bool end_site = false;
};

struct Skeleton {
    std::vector<Joint> joints;

    uint32_t joint_count() const;
    uint32_t channel_count() const;

    // Index of a joint by name, or -1. Linear: a skeleton is fifty joints and
    // this is called once per bone when a mesh is bound, not per frame.
    int find(const std::string& name) const;
};

struct Motion {
    Skeleton skeleton;
    float frame_time = 0.0f;

    // frames x skeleton.channel_count(), row-major by frame.
    std::vector<float> channels;

    uint32_t frame_count() const;

    // One world matrix a joint, for a frame. The forward kinematics: a joint's
    // world transform is its parent's, then its offset, then its rotation.
    //
    // Returned rather than written into a caller's buffer because it is fifty
    // matrices once a frame — the cost that matters is on the device, where a
    // vertex reads one of these.
    std::vector<Float4x4> pose(uint32_t frame) const;
};

// Reads a Biovision Hierarchy file.
//
// Throws std::runtime_error on a file it cannot open or a structure it does not
// recognise. The format is small enough to parse exactly, so anything
// unexpected is an error rather than something to skip: a silently dropped joint
// would move a limb somewhere plausible.
Motion load_bvh_motion(const std::string& path);
