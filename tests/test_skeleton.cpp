#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "skeleton.hpp"

namespace {

std::string arm()
{
    return std::string(GPURT_ASSETS_DIR) + "/arm.bvh";
}

Float3 world_position(const Float4x4& m)
{
    return Float3{m.at(0, 3), m.at(1, 3), m.at(2, 3)};
}

}  // namespace

TEST(Skeleton, TheHierarchyIsReadInParentFirstOrder)
{
    // Not a convenience: a joint's world transform is its parent's composed with
    // its own, and pose() makes one forward pass. A child that came first would
    // read a matrix nothing had written.
    const Motion motion = load_bvh_motion(arm());

    // Four joints and the end site the last one needs to have a length.
    EXPECT_EQ(motion.skeleton.joint_count(), 5u);
    EXPECT_EQ(motion.frame_count(), 4u);
    EXPECT_FLOAT_EQ(motion.frame_time, 0.033333f);

    for (uint32_t i = 0; i < motion.skeleton.joint_count(); ++i) {
        EXPECT_LT(motion.skeleton.joints[i].parent, static_cast<int>(i))
            << "joint " << i << " comes before its parent";
    }

    EXPECT_EQ(motion.skeleton.find("Shoulder"), 0);
    EXPECT_EQ(motion.skeleton.joints[0].parent, -1);
    EXPECT_GE(motion.skeleton.find("Hand"), 0);
    EXPECT_EQ(motion.skeleton.find("Elbow"), -1);
}

TEST(Skeleton, AnEndSiteIsWhereTheLimbActuallyStops)
{
    // It has an offset and no channels, and it is kept rather than dropped: a
    // mesh bound to the last real joint needs to know how long that joint is,
    // and the end site is the only thing that says so.
    const Motion motion = load_bvh_motion(arm());

    const Joint& tip = motion.skeleton.joints[motion.skeleton.joint_count() - 1];
    EXPECT_TRUE(tip.end_site);
    EXPECT_EQ(tip.channel_count, 0u);
    EXPECT_FLOAT_EQ(tip.offset.y, -0.5f);

    // Channels are counted over the joints that have them, which is what a
    // frame's run of values is sized by.
    EXPECT_EQ(motion.skeleton.channel_count(), 6u + 3u + 3u + 3u);
    EXPECT_EQ(motion.channels.size(), 4u * motion.skeleton.channel_count());
}

TEST(Skeleton, TheRestPoseHangsTheOffsetsEndToEnd)
{
    // Frame zero has every rotation at zero, so the arm is the offsets summed.
    // That is the check that offsets compose at all, before any rotation is
    // asked to compose with them.
    const Motion motion = load_bvh_motion(arm());
    const std::vector<Float4x4> rest = motion.pose(0);

    EXPECT_NEAR(world_position(rest[0]).y, 0.0f, 1e-5f);
    EXPECT_NEAR(world_position(rest[1]).y, -1.0f, 1e-5f);
    EXPECT_NEAR(world_position(rest[2]).y, -3.0f, 1e-5f);
    EXPECT_NEAR(world_position(rest[3]).y, -4.5f, 1e-5f);
    EXPECT_NEAR(world_position(rest[4]).y, -5.0f, 1e-5f);

    for (const Float4x4& joint : rest) {
        EXPECT_NEAR(world_position(joint).x, 0.0f, 1e-5f);
        EXPECT_NEAR(world_position(joint).z, 0.0f, 1e-5f);
    }
}

TEST(Skeleton, ARotationCarriesEverythingBelowIt)
{
    // What forward kinematics is for, and the reason a pose cannot be worked out
    // a joint at a time: bending the elbow moves the hand, and the hand's own
    // channels say nothing about it.
    const Motion motion = load_bvh_motion(arm());

    const int fore = motion.skeleton.find("ForeArm");
    const int hand = motion.skeleton.find("Hand");
    ASSERT_GE(fore, 0);
    ASSERT_GE(hand, 0);

    const std::vector<Float4x4> straight = motion.pose(0);
    const std::vector<Float4x4> bent = motion.pose(2);  // 60 degrees at ForeArm

    // The joint that turns has not moved — a rotation is about its own origin.
    EXPECT_NEAR(world_position(bent[fore]).y, world_position(straight[fore]).y, 1e-5f);
    EXPECT_NEAR(world_position(bent[fore]).x, world_position(straight[fore]).x, 1e-5f);

    // What hangs below it has. The offset is 1.5 along -y, and a positive turn
    // about z sends it to +x — which is the handedness the file's channel order
    // decides, not something to assume.
    EXPECT_GT(std::fabs(world_position(bent[hand]).x), 1.0f)
        << "the hand did not follow the forearm";
    EXPECT_NEAR(world_position(bent[hand]).x, 1.5f * std::sin(radians(60.0f)), 1e-4f);
    EXPECT_NEAR(world_position(bent[hand]).y,
                world_position(straight[fore]).y - 1.5f * std::cos(radians(60.0f)),
                1e-4f);

    // And a turn further up carries the turn below it as well.
    const std::vector<Float4x4> both = motion.pose(3);
    EXPECT_GT(std::fabs(world_position(both[hand]).x),
              std::fabs(world_position(bent[hand]).x));
}

TEST(Skeleton, AFileItCannotReadIsRefused)
{
    EXPECT_THROW(load_bvh_motion("nowhere.bvh"), std::runtime_error);

    // The acceleration structure's own name for a file, which is not this
    // format at all — the two abbreviations collide and the error should say so
    // rather than parsing halfway.
    EXPECT_THROW(load_bvh_motion(std::string(GPURT_ASSETS_DIR) + "/cube.obj"),
                 std::runtime_error);
}
