// The four routes to one frame, over scenes chosen to make them disagree: three
// rasteriser variants and the ray tracer, plus the branch against the blend on
// the two routes where that trade is worth reading.
//
// Both renderers run from one DrawTarget, so the camera cannot differ between
// them — which is the whole basis of the comparison, and something the previous
// figures could not promise. Those came from scenes built by hand in a session
// and never committed, so when 1/w joined the screen vertex and moved every
// load count, nothing could be re-run to catch it. The scenes are code here,
// and the tables printed are meant to be pasted straight into RESULTS.md.
//
//   ./build/benchmarks/render_bench                stdout, plus
//   benchmarks/result/render_bench.*
//   ./build/benchmarks/render_bench --out some/dir writes into that directory
//
// Every column is a count of what the scheduler issued, so two runs on any
// machine agree. Nothing here is timed.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app_run.hpp"
#include "math3d.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"
#include "scenes.hpp"

namespace {

// The frame and the scenes are in scenes.hpp, shared with model_bench so that a
// scene named in both tables is the same triangles in both.
constexpr uint32_t WIDTH = BENCH_WIDTH;
constexpr uint32_t HEIGHT = BENCH_HEIGHT;

DrawTarget target()
{
    return bench_target();
}

struct Reading {
    uint64_t weighted = 0;
    double divergence = 0.0;
    bool ran = true;
    std::string refused;
};

// A callable rather than a function pointer: the routes take defaulted
// arguments that differ between them, and a caller naming one is what selects a
// variant here.
using Route = std::function<std::vector<Float3>(MyGPURuntime&, const std::vector<Float3>&,
                                                const DrawTarget&)>;

Reading measure(const Route& route, const std::vector<Float3>& world,
                const GPUSpec& machine)
{
    MyGPURuntime rt(1u << 26);
    rt.myrt_set_spec(machine);
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
    Reading walk_pred;
    Reading tiled_pred;
    Reading ray_pred;
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
    // --out names a directory, as it does for every executable here, so one run
    // can gather its images and numbers in one place.
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "render_bench";

    // --machine machines/a100.spec, or the defaults. Written into the run
    // directory beside the tables, so a result carries the machine that made it.
    const GPUSpec machine = machine_from(args);
    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << machine.to_text();

    const std::vector<std::pair<std::string, std::vector<Float3>>> scenes = {
        {"small, spread over the frame", spread(2)},
        {"small, spread over the frame", spread(4)},
        {"small, spread over the frame", spread(8)},
        {"medium, stacked at the centre", stacked(16, 0.5f)},
        {"full-frame, stacked", stacked(16, 4.0f)},
    };

    // Spelling the pointer type out is what picks the vector overload of a route
    // name that also has a Mesh one.
    using Raster = std::vector<Float3> (*)(MyGPURuntime&, const std::vector<Float3>&,
                                           const DrawTarget&, bool);
    const auto route = [](Raster fn, bool predicated) {
        return [fn, predicated](MyGPURuntime& rt, const std::vector<Float3>& w,
                                const DrawTarget& t) { return fn(rt, w, t, predicated); };
    };

    std::vector<Row> rows;
    for (const auto& [name, world] : scenes) {
        Row r;
        r.scene = name;
        r.triangles = static_cast<uint32_t>(world.size() / 3);
        r.walk = measure(route(draw_walk, false), world, machine);
        r.tiled = measure(route(draw_tiled, false), world, machine);
        r.shared = measure(route(draw_shared, false), world, machine);
        r.ray = measure([](MyGPURuntime& rt, const std::vector<Float3>& w,
                           const DrawTarget& t) { return draw_raytrace(rt, w, t); },
                        world, machine);
        r.walk_pred = measure(route(draw_walk, true), world, machine);
        r.tiled_pred = measure(route(draw_tiled, true), world, machine);
        r.ray_pred = measure(
            [](MyGPURuntime& rt, const std::vector<Float3>& w, const DrawTarget& t) {
                return draw_raytrace(rt, w, t, Shading{}, true);
            },
            world, machine);
        rows.push_back(r);
    }

    std::printf("\n[BENCH] four routes to one frame — %ux%u\n\n", WIDTH, HEIGHT);

    // Which machine, stated where the numbers are. The scenes were committed to
    // code for the same reason: a figure nobody can reproduce is a figure nobody
    // can check — and machines/ holds the files this flag takes.
    std::printf("%s\n", machine.describe().c_str());
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

    // Its own table rather than two more columns above: the question is what a
    // blend costs against the branch on the SAME route, so the pairs have to sit
    // next to each other. Divergence goes to zero either way, so the figure that
    // decides it is the work — a shade every lane runs against a shade only the
    // covering lanes run.
    std::printf("\n  %-30s %5s %9s %9s %9s %9s\n", "predication, vs its branch", "tris",
                "walk", "tiled", "raytrace", "ray div");
    for (const Row& r : rows) {
        std::printf("  %-30s %5u %8.1f%% %8.1f%% %8.1f%% %8.2f%%\n", r.scene.c_str(),
                    r.triangles, change(r.walk.weighted, r.walk_pred.weighted),
                    change(r.tiled.weighted, r.tiled_pred.weighted),
                    change(r.ray.weighted, r.ray_pred.weighted),
                    100.0 * r.ray.divergence);
    }

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "render_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "scene,triangles,route,weighted_lane_ops,divergence_rate\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {
            {"walk", &r.walk},           {"tiled", &r.tiled},
            {"shared", &r.shared},       {"raytrace", &r.ray},
            {"walk+pred", &r.walk_pred}, {"tiled+pred", &r.tiled_pred}};
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

    md << "\n## Branch against blend\n\n"
       << "| Scene | Triangles | walk | tiled | raytrace | ray divergence |\n"
       << "|---|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(
            buf, sizeof(buf), "| %s | %u | %.1f%% | %.1f%% | **%.1f%%** | %.2f%% |\n",
            r.scene.c_str(), r.triangles, change(r.walk.weighted, r.walk_pred.weighted),
            change(r.tiled.weighted, r.tiled_pred.weighted),
            change(r.ray.weighted, r.ray_pred.weighted), 100.0 * r.ray.divergence);
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
