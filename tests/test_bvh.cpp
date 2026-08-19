#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "bvh.hpp"
#include "mesh.hpp"

namespace {

// A scene big enough that a tree has something to decide, laid out from a fixed
// seed so a figure taken from it can be reproduced.
std::vector<Float3> scattered_triangles(uint32_t count, uint32_t seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> place(-4.0f, 4.0f);
    std::uniform_real_distribution<float> size(0.05f, 0.4f);

    std::vector<Float3> world;
    world.reserve(count * 3);
    for (uint32_t i = 0; i < count; ++i) {
        const Float3 centre{place(rng), place(rng), place(rng)};
        const float s = size(rng);
        world.push_back(centre + Float3{-s, -s, 0.0f});
        world.push_back(centre + Float3{s, -s, 0.0f});
        world.push_back(centre + Float3{0.0f, s, s});
    }
    return world;
}

// A dense cluster with a few large triangles thrown far out. The shape a mesh
// actually has, and the one a median split handles worst: half the centroids sit
// inside a fraction of the root's box.
std::vector<Float3> clustered(uint32_t count, uint32_t seed = 7)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> tight(0.0f, 0.35f);
    std::uniform_real_distribution<float> away(-20.0f, 20.0f);
    std::uniform_real_distribution<float> large(1.0f, 3.0f);

    std::vector<Float3> world;
    world.reserve(count * 3);
    for (uint32_t i = 0; i < count; ++i) {
        const bool outlier = i % 25 == 0;
        const Float3 centre = outlier ? Float3{away(rng), away(rng), away(rng)}
                                      : Float3{tight(rng), tight(rng), tight(rng)};
        const float s = outlier ? large(rng) : 0.03f;
        world.push_back(centre + Float3{-s, -s, 0.0f});
        world.push_back(centre + Float3{s, -s, 0.0f});
        world.push_back(centre + Float3{0.0f, s, s});
    }
    return world;
}

bool contains(Float3 lo, Float3 hi, Float3 p)
{
    const float eps = 1e-4f;
    return p.x >= lo.x - eps && p.x <= hi.x + eps && p.y >= lo.y - eps &&
           p.y <= hi.y + eps && p.z >= lo.z - eps && p.z <= hi.z + eps;
}

}  // namespace

TEST(Bvh, EveryTriangleIsInExactlyOneLeaf)
{
    // The permutation is the part a traversal trusts without checking: a leaf is
    // a range, so a triangle that landed in two ranges would be intersected
    // twice and one that landed in none would be invisible.
    const std::vector<Float3> world = scattered_triangles(200);
    const Bvh bvh = build_bvh(world);

    std::vector<uint32_t> times_covered(bvh.triangle_count(), 0);
    for (uint32_t n = 0; n < bvh.node_count(); ++n) {
        const float* node = &bvh.nodes[n * BVH_NODE_FLOATS];
        const uint32_t count = static_cast<uint32_t>(node[7]);
        if (count == 0) {
            continue;
        }
        const uint32_t first = static_cast<uint32_t>(node[6]);
        for (uint32_t i = 0; i < count; ++i) {
            ASSERT_LT(first + i, bvh.triangle_count());
            ++times_covered[first + i];
        }
    }

    for (uint32_t i = 0; i < times_covered.size(); ++i) {
        EXPECT_EQ(times_covered[i], 1u)
            << "triangle " << i << " is in " << times_covered[i] << " leaves";
    }
    EXPECT_EQ(bvh.triangles.size(), world.size());
}

TEST(Bvh, ANodeContainsTheTrianglesUnderIt)
{
    // What a traversal actually relies on: a ray that misses a node's box may
    // skip everything below it. Bounds that were too small would drop geometry
    // in a way no frame comparison localises.
    const std::vector<Float3> world = scattered_triangles(150);
    const Bvh bvh = build_bvh(world);

    for (uint32_t n = 0; n < bvh.node_count(); ++n) {
        const float* node = &bvh.nodes[n * BVH_NODE_FLOATS];
        const Float3 lo{node[0], node[1], node[2]};
        const Float3 hi{node[3], node[4], node[5]};
        ASSERT_LE(lo.x, hi.x);
        ASSERT_LE(lo.y, hi.y);
        ASSERT_LE(lo.z, hi.z);

        const uint32_t count = static_cast<uint32_t>(node[7]);
        if (count > 0) {
            const uint32_t first = static_cast<uint32_t>(node[6]);
            for (uint32_t t = first; t < first + count; ++t) {
                for (uint32_t v = 0; v < 3; ++v) {
                    EXPECT_TRUE(contains(lo, hi, bvh.triangles[t * 3 + v]))
                        << "node " << n << " does not hold its triangle " << t;
                }
            }
            continue;
        }

        // An interior node has to hold both children whole, since a ray that
        // misses it never tests them.
        const uint32_t left = static_cast<uint32_t>(node[6]);
        for (uint32_t child : {left, left + 1}) {
            const float* c = &bvh.nodes[child * BVH_NODE_FLOATS];
            EXPECT_TRUE(contains(lo, hi, Float3{c[0], c[1], c[2]}));
            EXPECT_TRUE(contains(lo, hi, Float3{c[3], c[4], c[5]}));
        }
    }
}

TEST(Bvh, TheRootHoldsTheWholeScene)
{
    const std::vector<Float3> world = scattered_triangles(64);
    const Bvh bvh = build_bvh(world);

    Float3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max()};
    Float3 hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
              -std::numeric_limits<float>::max()};
    for (Float3 v : world) {
        lo = Float3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = Float3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }

    EXPECT_NEAR(bvh.bounds_min().x, lo.x, 1e-5f);
    EXPECT_NEAR(bvh.bounds_min().y, lo.y, 1e-5f);
    EXPECT_NEAR(bvh.bounds_min().z, lo.z, 1e-5f);
    EXPECT_NEAR(bvh.bounds_max().x, hi.x, 1e-5f);
    EXPECT_NEAR(bvh.bounds_max().y, hi.y, 1e-5f);
    EXPECT_NEAR(bvh.bounds_max().z, hi.z, 1e-5f);
}

TEST(Bvh, TheDepthReportedIsTheDepthThatExists)
{
    // The traversal stack is sized by this number, so it is not a statistic —
    // an underestimate overruns a thread's slice of shared memory.
    const std::vector<Float3> world = scattered_triangles(300);
    const Bvh bvh = build_bvh(world);

    uint32_t deepest = 0;
    const auto walk = [&](auto&& self, uint32_t node, uint32_t depth) -> void {
        deepest = std::max(deepest, depth + 1);
        const float* n = &bvh.nodes[node * BVH_NODE_FLOATS];
        if (n[7] > 0.0f) {
            return;
        }
        const uint32_t left = static_cast<uint32_t>(n[6]);
        self(self, left, depth + 1);
        self(self, left + 1, depth + 1);
    };
    walk(walk, 0, 0);

    EXPECT_EQ(bvh.max_depth, deepest);
}

TEST(Bvh, ALeafHoldsNoMoreThanItWasAskedTo)
{
    // Except where the heuristic refused to split, which is the one case a leaf
    // may be larger — and only under SAH, which is the mode that can refuse.
    for (uint32_t leaf_size : {1u, 4u, 8u}) {
        const std::vector<Float3> world = scattered_triangles(120);
        const Bvh bvh = build_bvh(world, BvhSplit::Median, leaf_size);
        for (uint32_t n = 0; n < bvh.node_count(); ++n) {
            const float* node = &bvh.nodes[n * BVH_NODE_FLOATS];
            EXPECT_LE(static_cast<uint32_t>(node[7]), leaf_size)
                << "leaf size " << leaf_size << ", node " << n;
        }
    }
}

TEST(Bvh, SmallerLeavesMakeADeeperTree)
{
    const std::vector<Float3> world = scattered_triangles(400);
    const Bvh fine = build_bvh(world, BvhSplit::Median, 1);
    const Bvh coarse = build_bvh(world, BvhSplit::Median, 8);

    EXPECT_GT(fine.node_count(), coarse.node_count());
    EXPECT_GT(fine.max_depth, coarse.max_depth);
}

TEST(Bvh, SAHBuildsAShallowerTreeThanAMedianSplit)
{
    // The gain that holds whatever the scene looks like, and the one that costs
    // something elsewhere: depth sizes the traversal stack, the stack lives in
    // shared memory, and shared memory is what caps how many blocks an SM holds.
    for (const std::vector<Float3>& world :
         {scattered_triangles(500), clustered(500), clustered(2000)}) {
        const Bvh median = build_bvh(world, BvhSplit::Median);
        const Bvh sah = build_bvh(world, BvhSplit::SAH);
        EXPECT_LT(sah.max_depth, median.max_depth)
            << "median " << median.max_depth << ", SAH " << sah.max_depth;
    }
}

TEST(Bvh, SAHVisitsFewerNodesWhereTheSceneIsUneven)
{
    // What the heuristic is for, stated as the condition it holds under.
    //
    // Triangles of one size spread evenly leave nothing to find: the midpoint of
    // the widest axis is already close to the split minimising area x count, and
    // SAH lands within a fraction of a percent of it. A dense cluster with a few
    // large triangles scattered far off is the case that separates them, and it
    // is what a real mesh looks like.
    const Float3 origin{0.0f, 0.0f, 12.0f};
    const auto visited = [&](const Bvh& bvh) {
        uint64_t total = 0;
        for (int y = -8; y <= 8; ++y) {
            for (int x = -8; x <= 8; ++x) {
                total += nodes_visited(bvh, origin,
                                       Float3{static_cast<float>(x) * 0.05f,
                                              static_cast<float>(y) * 0.05f, -1.0f});
            }
        }
        return total;
    };

    const std::vector<Float3> uneven = clustered(500);
    EXPECT_LT(visited(build_bvh(uneven, BvhSplit::SAH)),
              visited(build_bvh(uneven, BvhSplit::Median)));

    // And on the even scene it neither gains nor loses much, which is the half
    // of the result a test that only checked the win would hide.
    const std::vector<Float3> even = scattered_triangles(500);
    const uint64_t sah = visited(build_bvh(even, BvhSplit::SAH));
    const uint64_t median = visited(build_bvh(even, BvhSplit::Median));
    const uint64_t gap = sah > median ? sah - median : median - sah;
    EXPECT_LT(gap * 100, median) << "expected the two within 1% on an even scene";
}

TEST(Bvh, GeometryItCannotBuildOverIsRefused)
{
    EXPECT_THROW(build_bvh({}), std::runtime_error);
    EXPECT_THROW(build_bvh({Float3{}, Float3{}}), std::runtime_error);
    EXPECT_THROW(build_bvh(scattered_triangles(4), BvhSplit::SAH, 0), std::runtime_error);
}

TEST(Bvh, AMeshBuildsTheSameTreeAsItsFlattenedForm)
{
    const Mesh cube = cube_mesh();
    const Bvh bvh = build_bvh(cube.flattened());
    EXPECT_EQ(bvh.triangle_count(), cube.triangle_count());
    EXPECT_GT(bvh.node_count(), 1u);
}
