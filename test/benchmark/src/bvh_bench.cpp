// The last algorithm in this repository that was still linear.
//
// The ray tracer tested every triangle for every pixel, so its cost grew with
// the scene however little of it a ray could possibly meet. A bounding volume
// hierarchy is the answer everything from OptiX to an RT core is built around,
// and what it removes is exactly that.
//
// Three questions, and the second is the one worth the run:
//
//   1. what the tree removes, against walking the whole list
//   2. what SIMT gives back — a warp's lanes take different paths down a tree,
//      so the work saved and the slots saved are not the same number
//   3. whether entering the nearer child first pays for finding out which it is
//
//   ./build/benchmarks/bvh_bench                     result/bvh.{md,csv}
//   ./build/benchmarks/bvh_bench --leaf 8            triangles a leaf may hold
//   ./build/benchmarks/bvh_bench --machine <file>    latencies come from it
//   ./build/benchmarks/bvh_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "bvh.hpp"
#include "gpu_spec.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// Small triangles scattered through the view, from a fixed seed so a figure here
// can be reproduced. Scattered rather than stacked because that is where a tree
// has something to skip: a scene where every triangle covers the frame gives a
// hierarchy nothing to reject.
std::vector<Float3> scattered(uint32_t count, uint32_t seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> place(-1.2f, 1.2f);
    std::uniform_real_distribution<float> size(0.05f, 0.25f);

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

struct Reading {
    uint64_t issued = 0;    // slots the scheduler spent
    uint64_t lane_ops = 0;  // lane-instructions that actually ran
    double divergence = 0.0;
    uint32_t depth = 0;
    uint32_t nodes = 0;
};

Reading trace(const std::vector<Float3>& world, bool accelerated, BvhSplit split,
              uint32_t leaf, TraversalOrder order)
{
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    MyGPURuntime rt(1u << 27);
    rt.myrt_set_spec(MACHINE);

    Reading r;
    if (!accelerated) {
        draw_raytrace(rt, world, target);
    } else {
        DeviceGeometry geometry = upload_accelerated(rt, world, split, leaf, order);
        DeviceFrame frame = allocate_frame(rt, target);
        draw_raytrace(rt, geometry, frame, target);
        r.depth = geometry.bvh_depth;
        release(rt, frame);
        release(rt, geometry);
    }
    r.issued = rt.stats().warp_steps;
    r.lane_ops = rt.stats().active_lane_ops;
    r.divergence = rt.divergence_rate() * 100.0;
    return r;
}

std::string with_commas(uint64_t v)
{
    std::string digits = std::to_string(v);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(digits[i]);
    }
    return out;
}

double ratio(uint64_t from, uint64_t to)
{
    return static_cast<double>(from) / static_cast<double>(to);
}

struct Row {
    uint32_t triangles = 0;
    Reading linear;
    Reading unordered;
    Reading nearest;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "bvh";
    MACHINE = machine_from(args);
    const uint32_t leaf = static_cast<uint32_t>(args.flag("leaf", BVH_DEFAULT_LEAF));

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] a tree against every triangle for every pixel\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf("  %ux%u, SAH, %u triangles a leaf\n\n", WIDTH, HEIGHT, leaf);

    std::vector<Row> rows;
    std::printf("  %7s | %12s %6s | %12s %6s %7s | %7s %8s\n", "tris", "linear", "div",
                "bvh", "div", "depth", "issued", "lane work");
    for (const uint32_t count : {16u, 64u, 256u, 1024u, 4096u}) {
        const std::vector<Float3> world = scattered(count);
        Row row;
        row.triangles = count;
        row.linear = trace(world, false, BvhSplit::SAH, leaf, TraversalOrder::Unordered);
        row.unordered =
            trace(world, true, BvhSplit::SAH, leaf, TraversalOrder::Unordered);
        row.nearest = trace(world, true, BvhSplit::SAH, leaf, TraversalOrder::Nearest);
        rows.push_back(row);

        std::printf("  %7u | %12s %5.1f%% | %12s %5.1f%% %7u | %6.2fx %7.2fx\n", count,
                    with_commas(row.linear.issued).c_str(), row.linear.divergence,
                    with_commas(row.unordered.issued).c_str(), row.unordered.divergence,
                    row.unordered.depth, ratio(row.linear.issued, row.unordered.issued),
                    ratio(row.linear.lane_ops, row.unordered.lane_ops));
    }

    std::printf(
        "\n  The two ratios are the result. Lane work is what the tree "
        "removes;\n  issued slots are what a warp keeps after its lanes "
        "take different\n  paths down it.\n");

    std::printf("\n  %7s | %12s | %12s | %s\n", "tris", "unordered", "nearest",
                "ordering");
    for (const Row& r : rows) {
        std::printf("  %7u | %12s | %12s | %6.2fx\n", r.triangles,
                    with_commas(r.unordered.issued).c_str(),
                    with_commas(r.nearest.issued).c_str(),
                    ratio(r.unordered.issued, r.nearest.issued));
    }
    std::printf(
        "\n  Ordering costs two slab tests at every interior node and "
        "saves whole\n  subtrees. Which wins is a property of the scene, "
        "not of the technique.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "bvh_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "triangles,route,leaf,issued,lane_ops,divergence_percent,depth\n";
    for (const Row& r : rows) {
        const auto line = [&](const char* name, const Reading& x) {
            csv << r.triangles << ',' << name << ',' << leaf << ',' << x.issued << ','
                << x.lane_ops << ',' << x.divergence << ',' << x.depth << '\n';
        };
        line("linear", r.linear);
        line("bvh-unordered", r.unordered);
        line("bvh-nearest", r.nearest);
    }

    std::ofstream md(prefix + ".md");
    char buf[400];
    md << "<!-- generated by benchmarks/bvh_bench; do not edit by hand -->\n\n"
       << "## A tree against every triangle for every pixel\n\n"
       << WIDTH << "x" << HEIGHT << ", SAH, " << leaf
       << " triangles a leaf. Small triangles scattered through the view, from a "
          "fixed\nseed.\n\n"
       << "| Triangles | Linear | div | BVH | div | Depth | Issued | Lane work |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf),
                      "| %u | %s | %.1f%% | %s | %.1f%% | %u | **%.2fx** | %.2fx |\n",
                      r.triangles, with_commas(r.linear.issued).c_str(),
                      r.linear.divergence, with_commas(r.unordered.issued).c_str(),
                      r.unordered.divergence, r.unordered.depth,
                      ratio(r.linear.issued, r.unordered.issued),
                      ratio(r.linear.lane_ops, r.unordered.lane_ops));
        md << buf;
    }
    md << "\nLane work is what the tree removes; issued slots are what a warp keeps "
          "after\nits lanes take different paths down it. The gap between the two "
          "columns is\nthe SIMT tax on tree traversal.\n\n"
       << "### Entering the nearer child first\n\n"
       << "| Triangles | Unordered | Nearest | Ordering |\n"
       << "|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %u | %s | %s | **%.2fx** |\n", r.triangles,
                      with_commas(r.unordered.issued).c_str(),
                      with_commas(r.nearest.issued).c_str(),
                      ratio(r.unordered.issued, r.nearest.issued));
        md << buf;
    }
    md << "\nTwo slab tests at every interior node, against whole subtrees the "
          "running\nbest can cut off. Which wins is a property of the scene rather "
          "than of the\ntechnique.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
