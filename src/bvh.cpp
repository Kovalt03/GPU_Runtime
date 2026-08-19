#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "bvh.hpp"

namespace {

struct Bounds {
    Float3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max()};
    Float3 hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
              -std::numeric_limits<float>::max()};

    void add(Float3 p)
    {
        lo = Float3{std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = Float3{std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }

    bool empty() const
    {
        return lo.x > hi.x;
    }

    // An empty box is skipped rather than merged. Merging one would put
    // +/-FLT_MAX into the result and hand the heuristic an infinite area, which
    // is exactly what a sweep over bins runs into — most of them are empty.
    void add(const Bounds& b)
    {
        if (b.empty()) {
            return;
        }
        add(b.lo);
        add(b.hi);
    }

    // Half the surface area, which is what the heuristic compares. The factor of
    // two divides out of every comparison, so it is left off.
    float area() const
    {
        if (empty()) {
            return 0.0f;
        }
        const Float3 d = hi - lo;
        return d.x * d.y + d.y * d.z + d.z * d.x;
    }

    float component(uint32_t axis, bool high) const
    {
        const Float3& v = high ? hi : lo;
        return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
    }
};

float component(Float3 v, uint32_t axis)
{
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

// One triangle's worth of what a split decision needs. Built once, so the
// recursion sorts sixteen bytes rather than three vertices.
struct Ref {
    Bounds bounds;
    Float3 centroid;
    uint32_t triangle = 0;
};

// SAH is evaluated at bin boundaries rather than at every centroid: sorting each
// axis at every level is the textbook build and it is O(n log^2 n), while
// binning is one pass and lands within a percent of the same tree. Twelve is
// where the returns flatten in the literature.
constexpr uint32_t SAH_BINS = 12;

struct Split {
    uint32_t axis = 0;
    float position = 0.0f;
    float cost = std::numeric_limits<float>::max();
};

// Where the widest axis is cut in half, by centroid. No cost is estimated, so
// there is nothing to compare against and the split is always taken.
Split median_split(const std::vector<Ref>& refs, size_t begin, size_t end,
                   const Bounds& centroids)
{
    const Float3 extent = centroids.hi - centroids.lo;
    Split s;
    s.axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2)
                                 : (extent.y > extent.z ? 1 : 2);
    s.position =
        (centroids.component(s.axis, false) + centroids.component(s.axis, true)) * 0.5f;
    s.cost = 0.0f;
    (void)refs;
    (void)begin;
    (void)end;
    return s;
}

// The cheapest bin boundary on any axis. Leaves cost untouched when no axis has
// any spread, which is the caller's signal to halve by count instead — every
// centroid at one point admits no boundary at all.
//
// The heuristic chooses where to split and not whether to: leaf_size is a cap,
// so a node above it divides whatever the areas say. Letting the cost refuse a
// split is the textbook form, and it would make leaf_size a suggestion — a cube
// stays one leaf of twelve under it, because six triangles either side of any
// plane still fill most of the parent's box.
Split sah_split(const std::vector<Ref>& refs, size_t begin, size_t end,
                const Bounds& centroids)
{
    Split best;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const float lo = centroids.component(axis, false);
        const float hi = centroids.component(axis, true);
        if (hi - lo < 1e-9f) {
            continue;  // every centroid at the same place on this axis
        }
        const float scale = static_cast<float>(SAH_BINS) / (hi - lo);

        std::array<Bounds, SAH_BINS> bin_bounds{};
        std::array<uint32_t, SAH_BINS> bin_count{};
        bin_count.fill(0);
        for (size_t i = begin; i < end; ++i) {
            const float offset = (component(refs[i].centroid, axis) - lo) * scale;
            const uint32_t b =
                std::min(SAH_BINS - 1, static_cast<uint32_t>(std::max(0.0f, offset)));
            bin_bounds[b].add(refs[i].bounds);
            ++bin_count[b];
        }

        // Sweeping once from each end turns the eleven boundaries into two
        // linear passes rather than eleven quadratic ones.
        std::array<float, SAH_BINS - 1> left_area{};
        std::array<uint32_t, SAH_BINS - 1> left_count{};
        Bounds running;
        uint32_t count = 0;
        for (uint32_t b = 0; b + 1 < SAH_BINS; ++b) {
            running.add(bin_bounds[b]);
            count += bin_count[b];
            left_area[b] = running.area();
            left_count[b] = count;
        }

        Bounds right_running;
        uint32_t right_count = 0;
        for (uint32_t b = SAH_BINS - 1; b > 0; --b) {
            right_running.add(bin_bounds[b]);
            right_count += bin_count[b];
            const uint32_t boundary = b - 1;
            if (left_count[boundary] == 0 || right_count == 0) {
                continue;
            }
            const float cost =
                left_area[boundary] * static_cast<float>(left_count[boundary]) +
                right_running.area() * static_cast<float>(right_count);
            if (cost < best.cost) {
                best.axis = axis;
                best.position = lo + (hi - lo) * (static_cast<float>(b) /
                                                  static_cast<float>(SAH_BINS));
                best.cost = cost;
            }
        }
    }
    return best;
}

class Builder {
public:
    Builder(std::vector<Ref>& refs, BvhSplit split, uint32_t leaf_size)
        : refs_(refs), split_(split), leaf_size_(leaf_size)
    {
        nodes.resize(BVH_NODE_FLOATS);  // the root, which nothing points at
    }

    // Fills a node that has already been allocated, and allocates its children
    // as a pair before recursing into either.
    //
    // The pair is what makes an interior node one index rather than two, which
    // the layout depends on. Allocating a child at the point the recursion
    // reaches it does not do that: the right child would land after the whole of
    // the left subtree, so left + 1 would name a grandchild.
    void build(uint32_t self, size_t begin, size_t end, uint32_t depth)
    {
        max_depth = std::max(max_depth, depth + 1);

        Bounds bounds;
        Bounds centroids;
        for (size_t i = begin; i < end; ++i) {
            bounds.add(refs_[i].bounds);
            centroids.add(refs_[i].centroid);
        }
        write_bounds(self, bounds);

        const size_t count = end - begin;
        if (count <= leaf_size_) {
            nodes[self * BVH_NODE_FLOATS + 6] = static_cast<float>(begin);
            nodes[self * BVH_NODE_FLOATS + 7] = static_cast<float>(count);
            return;
        }

        const Split s = split_ == BvhSplit::SAH
                            ? sah_split(refs_, begin, end, centroids)
                            : median_split(refs_, begin, end, centroids);

        size_t mid = begin + count / 2;
        if (s.cost < std::numeric_limits<float>::max()) {
            const auto middle = std::partition(
                refs_.begin() + static_cast<std::ptrdiff_t>(begin),
                refs_.begin() + static_cast<std::ptrdiff_t>(end),
                [&](const Ref& r) { return component(r.centroid, s.axis) < s.position; });
            const size_t split_at = static_cast<size_t>(middle - refs_.begin());

            // Every centroid landing on one side leaves the recursion where it
            // started, so the halves are taken by count instead. Coincident
            // centroids are the case: a fan of triangles sharing a vertex.
            if (split_at != begin && split_at != end) {
                mid = split_at;
            }
        }

        const uint32_t left = static_cast<uint32_t>(nodes.size()) / BVH_NODE_FLOATS;
        nodes.resize(nodes.size() + 2 * BVH_NODE_FLOATS);
        nodes[self * BVH_NODE_FLOATS + 6] = static_cast<float>(left);
        nodes[self * BVH_NODE_FLOATS + 7] = 0.0f;

        build(left, begin, mid, depth + 1);
        build(left + 1, mid, end, depth + 1);
    }

    std::vector<float> nodes;
    uint32_t max_depth = 0;

private:
    void write_bounds(uint32_t node, const Bounds& b)
    {
        float* n = &nodes[node * BVH_NODE_FLOATS];
        n[0] = b.lo.x;
        n[1] = b.lo.y;
        n[2] = b.lo.z;
        n[3] = b.hi.x;
        n[4] = b.hi.y;
        n[5] = b.hi.z;
    }

    std::vector<Ref>& refs_;
    BvhSplit split_;
    uint32_t leaf_size_;
};

}  // namespace

uint32_t Bvh::node_count() const
{
    return static_cast<uint32_t>(nodes.size() / BVH_NODE_FLOATS);
}

uint32_t Bvh::triangle_count() const
{
    return static_cast<uint32_t>(triangles.size() / 3);
}

Float3 Bvh::bounds_min() const
{
    return Float3{nodes[0], nodes[1], nodes[2]};
}

Float3 Bvh::bounds_max() const
{
    return Float3{nodes[3], nodes[4], nodes[5]};
}

Bvh build_bvh_over(const std::vector<Box>& boxes, BvhSplit split, uint32_t leaf_size)
{
    if (boxes.empty()) {
        throw std::runtime_error("build_bvh_over: a tree over nothing has no root");
    }
    if (leaf_size == 0) {
        throw std::runtime_error("build_bvh_over: a leaf that holds nothing");
    }

    std::vector<Ref> refs(boxes.size());
    for (uint32_t i = 0; i < boxes.size(); ++i) {
        Ref& r = refs[i];
        r.triangle = i;
        r.bounds.add(boxes[i].lo);
        r.bounds.add(boxes[i].hi);
        r.centroid = (r.bounds.lo + r.bounds.hi) * 0.5f;
    }

    Builder builder(refs, split, leaf_size);
    builder.build(0, 0, refs.size(), 0);

    Bvh bvh;
    bvh.nodes = std::move(builder.nodes);
    bvh.max_depth = builder.max_depth;
    bvh.order.reserve(refs.size());
    for (const Ref& r : refs) {
        bvh.order.push_back(r.triangle);
    }
    return bvh;
}

Bvh build_bvh(const std::vector<Float3>& world, BvhSplit split, uint32_t leaf_size)
{
    if (world.empty() || world.size() % 3 != 0) {
        throw std::runtime_error("build_bvh: " + std::to_string(world.size()) +
                                 " vertices is not a non-empty multiple of three");
    }

    std::vector<Box> boxes(world.size() / 3);
    for (size_t i = 0; i < boxes.size(); ++i) {
        Bounds b;
        for (uint32_t v = 0; v < 3; ++v) {
            b.add(world[i * 3 + v]);
        }
        boxes[i] = Box{b.lo, b.hi};
    }

    Bvh bvh = build_bvh_over(boxes, split, leaf_size);

    // The permutation applied, so a leaf is a run of vertices rather than a run
    // of indices into one.
    bvh.triangles.reserve(world.size());
    for (const uint32_t triangle : bvh.order) {
        for (uint32_t v = 0; v < 3; ++v) {
            bvh.triangles.push_back(world[triangle * 3 + v]);
        }
    }
    return bvh;
}

uint32_t Tlas::instance_count() const
{
    return static_cast<uint32_t>(instances.size() / TLAS_INSTANCE_FLOATS);
}

Tlas build_tlas(const Bvh& blas, const std::vector<Float4x4>& models, BvhSplit split,
                uint32_t leaf_size)
{
    if (models.empty()) {
        throw std::runtime_error("build_tlas: no instances to place");
    }
    if (blas.nodes.empty()) {
        throw std::runtime_error("build_tlas: the lower level has no root");
    }

    const Float3 lo = blas.bounds_min();
    const Float3 hi = blas.bounds_max();

    // Eight corners transformed, and the box around where they land. Loose under
    // a rotation, and the standard answer — tightening it would mean deriving
    // bounds from geometry this level never reads.
    std::vector<Box> boxes;
    boxes.reserve(models.size());
    for (const Float4x4& model : models) {
        Bounds world;
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const Float3 p{(corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y,
                           (corner & 4) ? hi.z : lo.z};
            const Float4 t = transform(model, p, 1.0f);
            world.add(Float3{t.x, t.y, t.z});
        }
        boxes.push_back(Box{world.lo, world.hi});
    }

    Tlas tlas;
    tlas.tree = build_bvh_over(boxes, split, leaf_size);

    // In the tree's order, since a leaf names a range of positions rather than a
    // list of indices — the same trade build_bvh makes for triangles.
    tlas.instances.reserve(tlas.tree.order.size() * TLAS_INSTANCE_FLOATS);
    for (const uint32_t index : tlas.tree.order) {
        const Float4x4 world_to_object = inverse(models[index]);
        for (uint32_t row = 0; row < 4; ++row) {
            for (uint32_t col = 0; col < 4; ++col) {
                tlas.instances.push_back(world_to_object.at(row, col));
            }
        }
    }
    return tlas;
}

uint32_t nodes_visited(const Bvh& bvh, Float3 origin, Float3 direction)
{
    const Float3 inv{1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z};
    const std::array<float, 3> o{origin.x, origin.y, origin.z};
    const std::array<float, 3> d{inv.x, inv.y, inv.z};

    // The same test the kernel runs, so a count taken here describes the tree
    // the device walks rather than an idealised one.
    const auto hits = [&](uint32_t node) {
        const float* n = &bvh.nodes[node * BVH_NODE_FLOATS];
        float near = 0.0f;
        float far = std::numeric_limits<float>::max();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const float t0 = (n[axis] - o[axis]) * d[axis];
            const float t1 = (n[axis + 3] - o[axis]) * d[axis];
            near = std::max(near, std::min(t0, t1));
            far = std::min(far, std::max(t0, t1));
        }
        return near <= far;
    };

    uint32_t visited = 0;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const uint32_t node = stack.back();
        stack.pop_back();
        ++visited;
        if (!hits(node)) {
            continue;
        }
        const float* n = &bvh.nodes[node * BVH_NODE_FLOATS];
        if (n[7] > 0.0f) {
            continue;  // a leaf: its triangles are the intersection tests
        }
        const uint32_t left = static_cast<uint32_t>(n[6]);
        stack.push_back(left);
        stack.push_back(left + 1);
    }
    return visited;
}
