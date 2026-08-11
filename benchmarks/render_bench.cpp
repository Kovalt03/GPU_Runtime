// The four routes to one frame, over scenes chosen to make them disagree:
// three rasteriser variants and the ray tracer.
//
// Both renderers run from one DrawTarget, so the camera cannot differ between
// them — which is the whole basis of the comparison, and something the previous
// figures could not promise. Those came from scenes built by hand in a session
// and never committed, so when 1/w joined the screen vertex and moved every
// load count, nothing could be re-run to catch it. The scenes are code here,
// and the tables printed are meant to be pasted straight into RESULTS.md.
//
//   ./build/benchmarks/render_bench                 stdout, plus output/*.md, *.csv
//   ./build/benchmarks/render_bench --out some/path writes some/path.md and .csv
//
// Every column is a count of what the scheduler issued, so two runs on any
// machine agree. Nothing here is timed.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// Matched to the tests, so a figure here and an assertion there describe the
// same picture.
DrawTarget target()
{
    Camera cam;
    cam.eye = Float3{0.0f, 0.0f, 3.0f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    return DrawTarget{WIDTH, HEIGHT, cam};
}

void push_triangle(std::vector<Float3>& out, Float3 centre, float half)
{
    out.push_back(Float3{centre.x, centre.y + half, centre.z});
    out.push_back(Float3{centre.x - half, centre.y - half, centre.z});
    out.push_back(Float3{centre.x + half, centre.y - half, centre.z});
}

// An n x n grid of small triangles across the frame — the shape binning is for,
// since most tiles end up holding none at all.
//
// The 4 x 4 case reproduces the scene the tests use, which is the one row of
// the old tables that could be checked against anything.
std::vector<Float3> spread(uint32_t n)
{
    std::vector<Float3> world;
    for (uint32_t gy = 0; gy < n; ++gy) {
        for (uint32_t gx = 0; gx < n; ++gx) {
            const float t = (n == 1) ? 0.0f : static_cast<float>(gx) / (n - 1);
            const float u = (n == 1) ? 0.0f : static_cast<float>(gy) / (n - 1);
            push_triangle(world, Float3{-2.4f + 4.8f * t, -1.2f + 2.4f * u, 0.0f}, 0.25f);
        }
    }
    return world;
}

// Triangles piled at the centre, each a little further away.
//
// half is what decides whether binning has anything to remove: 0.5 reaches four
// of the eight tiles, 4.0 reaches all of them and leaves the binning nothing —
// which is the case it loses.
std::vector<Float3> stacked(uint32_t count, float half)
{
    std::vector<Float3> world;
    for (uint32_t i = 0; i < count; ++i) {
        push_triangle(world, Float3{0.0f, 0.0f, -0.01f * static_cast<float>(i)}, half);
    }
    return world;
}

struct Reading {
    uint64_t weighted = 0;
    double divergence = 0.0;
    bool ran = true;
    std::string refused;
};

// draw_raytrace carries a defaulted argument the other three do not, so it
// reaches measure through a wrapper of the shared shape rather than by widening
// the signature for one caller.
std::vector<Float3> trace(MyGPURuntime& rt, const std::vector<Float3>& world,
                          const DrawTarget& t)
{
    return draw_raytrace(rt, world, t);
}

Reading measure(std::vector<Float3> (*route)(MyGPURuntime&, const std::vector<Float3>&,
                                             const DrawTarget&),
                const std::vector<Float3>& world)
{
    MyGPURuntime rt(1u << 26);
    Reading r;
    try {
        route(rt, world, target());
    } catch (const std::runtime_error& e) {
        // A tile too full to stage is a result, not a crash: it is the limit
        // the shared-memory route has and the other two do not.
        r.ran = false;
        r.refused = e.what();
        return r;
    }
    r.weighted = rt.stats().weighted_lane_ops;
    r.divergence = rt.divergence_rate();
    return r;
}

struct Row {
    std::string scene;
    uint32_t triangles = 0;
    Reading walk;
    Reading tiled;
    Reading shared;
    Reading ray;
};

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
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

}  // namespace

int main(int argc, char** argv)
{
    std::string prefix = "output/render_bench";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            prefix = argv[++i];
        }
    }

    const std::vector<std::pair<std::string, std::vector<Float3>>> scenes = {
        {"small, spread over the frame", spread(2)},
        {"small, spread over the frame", spread(4)},
        {"small, spread over the frame", spread(8)},
        {"medium, stacked at the centre", stacked(16, 0.5f)},
        {"full-frame, stacked", stacked(16, 4.0f)},
    };

    std::vector<Row> rows;
    for (const auto& [name, world] : scenes) {
        Row r;
        r.scene = name;
        r.triangles = static_cast<uint32_t>(world.size() / 3);
        r.walk = measure(draw_walk, world);
        r.tiled = measure(draw_tiled, world);
        r.shared = measure(draw_shared, world);
        r.ray = measure(trace, world);
        rows.push_back(r);
    }

    std::printf("\n[BENCH] four routes to one frame — %ux%u\n\n", WIDTH, HEIGHT);
    std::printf("  %-30s %5s %13s %13s %9s %13s %9s %13s %9s\n", "scene", "tris", "walk",
                "tiled", "vs walk", "shared", "vs walk", "raytrace", "vs walk");
    for (const Row& r : rows) {
        std::printf("  %-30s %5u %13s %13s %8.1f%% %13s %8.1f%% %13s %8.1f%%\n",
                    r.scene.c_str(), r.triangles, with_commas(r.walk.weighted).c_str(),
                    with_commas(r.tiled.weighted).c_str(),
                    change(r.walk.weighted, r.tiled.weighted),
                    with_commas(r.shared.weighted).c_str(),
                    change(r.walk.weighted, r.shared.weighted),
                    with_commas(r.ray.weighted).c_str(),
                    change(r.walk.weighted, r.ray.weighted));
    }

    std::printf("\n  %-30s %5s %9s %9s %9s %9s\n", "divergence", "tris", "walk", "tiled",
                "shared", "raytrace");
    for (const Row& r : rows) {
        std::printf("  %-30s %5u %8.1f%% %8.1f%% %8.1f%% %8.1f%%\n", r.scene.c_str(),
                    r.triangles, 100.0 * r.walk.divergence, 100.0 * r.tiled.divergence,
                    100.0 * r.shared.divergence, 100.0 * r.ray.divergence);
    }

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "render_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "scene,triangles,route,weighted_lane_ops,divergence_rate\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {{"walk", &r.walk},
                                                                 {"tiled", &r.tiled},
                                                                 {"shared", &r.shared},
                                                                 {"raytrace", &r.ray}};
        for (const auto& [name, reading] : routes) {
            csv << '"' << r.scene << "\"," << r.triangles << ',' << name << ','
                << reading->weighted << ',' << reading->divergence << '\n';
        }
    }

    // Laid out as the tables in RESULTS.md, so an update is a paste.
    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/render_bench; do not edit by hand -->\n\n"
       << "## Binned into tiles\n\n"
       << "| Scene | Triangles | walk | tiled | Change |\n"
       << "|---|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %s | %u | %s | %s | **%.1f%%** |\n",
                      r.scene.c_str(), r.triangles, with_commas(r.walk.weighted).c_str(),
                      with_commas(r.tiled.weighted).c_str(),
                      change(r.walk.weighted, r.tiled.weighted));
        md << buf;
    }

    md << "\n## Staged through shared memory\n\n"
       << "| Scene | Triangles | walk | tiled | shared | vs tiled | vs walk |\n"
       << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(
            buf, sizeof(buf), "| %s | %u | %s | %s | %s | %.1f%% | **%.1f%%** |\n",
            r.scene.c_str(), r.triangles, with_commas(r.walk.weighted).c_str(),
            with_commas(r.tiled.weighted).c_str(), with_commas(r.shared.weighted).c_str(),
            change(r.tiled.weighted, r.shared.weighted),
            change(r.walk.weighted, r.shared.weighted));
        md << buf;
    }

    md << "\n## Two renderers, one image\n\n"
       << "| Scene | Triangles | raster ops | raster divergence "
          "| ray ops | ray divergence |\n"
       << "|---|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %s | %u | %s | %.1f%% | %s | **%.1f%%** |\n",
                      r.scene.c_str(), r.triangles, with_commas(r.walk.weighted).c_str(),
                      100.0 * r.walk.divergence, with_commas(r.ray.weighted).c_str(),
                      100.0 * r.ray.divergence);
        md << buf;
    }

    md << "\n## Divergence, all four\n\n"
       << "| Scene | Triangles | walk | tiled | shared | raytrace |\n"
       << "|---|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf),
                      "| %s | %u | %.1f%% | %.1f%% | %.1f%% | %.1f%% |\n",
                      r.scene.c_str(), r.triangles, 100.0 * r.walk.divergence,
                      100.0 * r.tiled.divergence, 100.0 * r.shared.divergence,
                      100.0 * r.ray.divergence);
        md << buf;
    }

    std::printf("\nwrote %s.md and %s.csv\n\n", prefix.c_str(), prefix.c_str());
    return 0;
}
