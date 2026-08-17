// What a cache is worth as a scene outgrows it.
//
// Coalescing answered how many lines a warp touches. It said nothing about the
// same line being touched again, and the naive walk touches every triangle from
// every warp — so the question a cache answers is how much of that repetition
// costs nothing.
//
// The frame stays at 64x32 throughout. A framebuffer is written once per pixel and
// never read back, so it does not compete for the cache; what a warp re-reads is
// the screen buffer, and that grows with the triangle count alone. Sixteen bytes a
// vertex and three vertices a triangle puts L1's 1024 lines at about 2,700
// triangles, which is what the sweep is built to cross.
//
// The shared-memory route drops out above a thousand: a tile holds more triangles
// than SHARED_TRIANGLE_CAPACITY stages, and at this frame size there are only eight
// tiles to spread them over. That refusal is a result rather than a gap — staging
// cannot reach the scenes where a cache starts to matter.
//
// What the sweep turns up is that binning is a locality optimisation as well as a
// work-reducing one. A block of the naive walk reads every triangle, so its working
// set is the whole scene; a block of the tiled route reads one tile's list, which is
// a fraction of it. Scale the cache down until the scene outgrows it and the walk
// starts missing while the tiled route does not — a distinction the flat cost model
// had no way to express, and part of what it was attributing to dropped triangles.
//
// The last section asks what else the flat model was hiding. Ordering a mesh for a
// vertex cache is worth a factor of three or four on fixed-function hardware and
// was worth a thousandth of a percent here, because a flat charge per lane cannot
// tell a vertex that was just read from one that was not. A cache can, so the
// question is put again — the same reorder, measured through the new model.
//
//   ./build/benchmarks/cache_bench            benchmarks/result/cache.{md,csv}
//   ./build/benchmarks/cache_bench --out dir  writes into that directory
//   ./build/benchmarks/cache_bench m.obj      reorders that mesh instead of the
//                                             sphere, from the project root

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "math3d.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/types.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

namespace {

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

// Small triangles scattered over the frame, as many as asked for. Spread rather
// than stacked so that binning has something to drop and the tiled route stays
// worth comparing; the depths differ so that nearest-wins does real work.
std::vector<Float3> scene(uint32_t triangles)
{
    std::vector<Float3> world;
    world.reserve(triangles * 3);
    for (uint32_t i = 0; i < triangles; ++i) {
        const float depth = -1.0f - 0.0001f * static_cast<float>(i);
        const float x = -0.9f + 1.8f * static_cast<float>((i * 37) % 101) / 100.0f;
        const float y = -0.9f + 1.8f * static_cast<float>((i * 61) % 103) / 100.0f;
        world.push_back(Float3{x, y, depth});
        world.push_back(Float3{x + 0.06f, y, depth});
        world.push_back(Float3{x, y + 0.06f, depth});
    }
    return world;
}

DrawTarget target()
{
    return DrawTarget{
        WIDTH, HEIGHT,
        Camera{Float3{0.0f, 0.0f, 3.0f}, Float3{0, 0, 0}, Float3{0, 1, 0}, 60.0f}};
}

using Route = std::vector<Float3> (*)(MyGPURuntime&, const std::vector<Float3>&,
                                      const DrawTarget&, bool);

struct Reading {
    uint64_t weighted = 0;
    uint64_t l1_hits = 0;
    uint64_t l2_hits = 0;
    uint64_t misses = 0;
    // Zero unless the reading was taken under LatencyModel::Modelled, where a warp
    // that has issued waits before it issues again. Only the reorder section below
    // needs it: what moves there is which level answered, and the two levels are
    // ten cycles apart in cost and a hundred and seventy in time.
    uint64_t cycles = 0;
    bool ran = true;

    uint64_t lookups() const
    {
        return l1_hits + l2_hits + misses;
    }

    double hit_rate() const
    {
        return lookups() == 0 ? 0.0
                              : 100.0 * static_cast<double>(l1_hits + l2_hits) /
                                    static_cast<double>(lookups());
    }
};

// Cache capacities scaled to these scenes, for the sweep that means to reach them.
//
// The hardware sizes cannot be reached here: L2 holds 65,536 lines and the largest
// scene below touches 3,750, so nothing is ever evicted from it and what a cache
// buys comes out a constant. These are chosen so that the sweep crosses both
// capacities — 32 lines is passed by every scene and 512 by the last two — and they
// are not hardware, which is why they are reported apart from the figures that are.
constexpr size_t SCALED_L1 = 32;
constexpr size_t SCALED_L2 = 512;

Reading measure(Route route, const std::vector<Float3>& world, MemoryModel model,
                bool scaled = false)
{
    // A runtime each, so that no scene inherits the L2 another warmed. The level
    // outlives a launch by design, which is right within one draw and wrong
    // between two unrelated ones.
    MyGPURuntime rt(1u << 29);
    rt.myrt_set_memory_model(model);
    if (scaled) {
        rt.myrt_set_cache_lines(SCALED_L1, SCALED_L2);
    }

    Reading r;
    try {
        route(rt, world, target(), false);
    } catch (const std::exception&) {
        // A tile too full to stage. The shared route has that limit and the other
        // two do not, which is the whole of why it stops appearing below.
        r.ran = false;
        return r;
    }
    r.weighted = rt.stats().weighted_lane_ops;
    r.l1_hits = rt.stats().l1_hits;
    r.l2_hits = rt.stats().l2_hits;
    r.misses = rt.stats().cache_misses;
    return r;
}

// L1 sizes for the reorder sweep, in lines. A line holds eight screen vertices, so
// these hold 16 to 8,192 and the sphere's 182 sit inside the range — deliberately,
// since ordering triangles for a cache can only pay where the mesh does not fit in
// one. The last is the hardware size, and is there to report that it does not.
constexpr size_t REORDER_L1[] = {2, 4, 8, 16, 32, L1_LINES};

// The same question put to a mesh, with the order of its triangles as the variable.
//
// The indexed walk is the only route where that order can reach the cache: it
// carries the index buffer into pass 2 and reads a vertex through it, so a vertex
// two triangles apart in the list may still be resident. bin_triangles de-indexes
// on the host, which is why the tiled route appears below as a control.
//
// Taken with latency on, because what the reorder moves is which level answered
// rather than how many misses there were, and the levels are 8 against 30 in issue
// capacity but 30 against 200 in cycles.
Reading measure_mesh(const Mesh& mesh, size_t l1_lines, bool tiled = false)
{
    MyGPURuntime rt(1u << 29);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);
    rt.myrt_set_cache_lines(l1_lines, SCALED_L2);

    if (tiled) {
        draw_tiled(rt, mesh, target());
    } else {
        draw_walk(rt, mesh, target());
    }

    Reading r;
    r.weighted = rt.stats().weighted_lane_ops;
    r.l1_hits = rt.stats().l1_hits;
    r.l2_hits = rt.stats().l2_hits;
    r.misses = rt.stats().cache_misses;
    r.cycles = rt.stats().cycles;
    return r;
}

// How many lines of screen data a scene holds, which is what a warp re-reads.
size_t working_lines(uint32_t triangles)
{
    const size_t bytes = static_cast<size_t>(triangles) * 3 * SCREEN_VERTEX_BYTES;
    return (bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
}

double change(uint64_t from, uint64_t to)
{
    return from == 0 ? 0.0
                     : 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
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

struct ReorderRow {
    size_t l1_lines = 0;
    Reading mixed;
    Reading fixed;
};

struct Row {
    uint32_t triangles = 0;
    const char* route = "";
    Reading coalesced;
    Reading cached;
    Reading scaled;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "cache";

    const std::pair<const char*, Route> routes[] = {
        {"walk", draw_walk}, {"tiled", draw_tiled}, {"shared", draw_shared}};
    const uint32_t counts[] = {360, 1000, 3000, 10000};

    std::printf("\n[BENCH] a cache against a growing working set — %ux%u\n\n", WIDTH,
                HEIGHT);
    std::printf("  L1 holds %zu lines of %u bytes\n\n", L1_LINES, CACHE_LINE_BYTES);
    std::printf("  %6s %7s %8s %8s %14s %14s %9s %9s\n", "tris", "lines", "vs L1",
                "route", "coalesced", "cached", "change", "hit rate");

    std::vector<Row> rows;
    for (const uint32_t triangles : counts) {
        const std::vector<Float3> world = scene(triangles);
        const size_t lines = working_lines(triangles);

        for (const auto& [name, route] : routes) {
            Row row;
            row.triangles = triangles;
            row.route = name;
            row.coalesced = measure(route, world, MemoryModel::Coalesced);
            row.cached = measure(route, world, MemoryModel::Cached);
            row.scaled = measure(route, world, MemoryModel::Cached, true);
            rows.push_back(row);

            if (!row.cached.ran) {
                std::printf("  %6u %7zu %7.2fx %8s %14s\n", triangles, lines,
                            static_cast<double>(lines) / static_cast<double>(L1_LINES),
                            name, "over capacity");
                continue;
            }
            std::printf("  %6u %7zu %7.2fx %8s %14s %14s %8.1f%% %8.1f%%\n", triangles,
                        lines, static_cast<double>(lines) / static_cast<double>(L1_LINES),
                        name, with_commas(row.coalesced.weighted).c_str(),
                        with_commas(row.cached.weighted).c_str(),
                        change(row.coalesced.weighted, row.cached.weighted),
                        row.cached.hit_rate());
        }
    }

    std::printf("\n  Where the hits came from, which is what having two levels buys:\n");
    std::printf("  %6s %8s %14s %14s %14s\n", "tris", "route", "L1", "L2", "misses");
    for (const Row& r : rows) {
        if (!r.cached.ran) {
            continue;
        }
        std::printf("  %6u %8s %14s %14s %14s\n", r.triangles, r.route,
                    with_commas(r.cached.l1_hits).c_str(),
                    with_commas(r.cached.l2_hits).c_str(),
                    with_commas(r.cached.misses).c_str());
    }

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "cache_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "triangles,lines,route,coalesced_weighted,cached_weighted,l1_hits,l2_hits,"
           "misses,scaled_weighted,scaled_misses\n";
    for (const Row& r : rows) {
        if (!r.cached.ran) {
            continue;
        }
        csv << r.triangles << ',' << working_lines(r.triangles) << ',' << r.route << ','
            << r.coalesced.weighted << ',' << r.cached.weighted << ',' << r.cached.l1_hits
            << ',' << r.cached.l2_hits << ',' << r.cached.misses << ','
            << r.scaled.weighted << ',' << r.scaled.misses << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/cache_bench; do not edit by hand -->\n\n"
       << "## A cache against a growing working set\n\n"
       << "| Triangles | Lines | vs L1 | Route | Coalesced | Cached | Change "
          "| Hit rate |\n"
       << "|---:|---:|---:|---|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        if (!r.cached.ran) {
            std::snprintf(buf, sizeof(buf),
                          "| %u | %zu | %.2fx | %s | over capacity | | | |\n",
                          r.triangles, working_lines(r.triangles),
                          static_cast<double>(working_lines(r.triangles)) /
                              static_cast<double>(L1_LINES),
                          r.route);
            md << buf;
            continue;
        }
        std::snprintf(buf, sizeof(buf),
                      "| %u | %zu | %.2fx | %s | %s | %s | **%.1f%%** | %.1f%% |\n",
                      r.triangles, working_lines(r.triangles),
                      static_cast<double>(working_lines(r.triangles)) /
                          static_cast<double>(L1_LINES),
                      r.route, with_commas(r.coalesced.weighted).c_str(),
                      with_commas(r.cached.weighted).c_str(),
                      change(r.coalesced.weighted, r.cached.weighted),
                      r.cached.hit_rate());
        md << buf;
    }

    // The same sweep against a cache small enough for the scenes to outgrow it. The
    // hardware-sized figures above are flat because nothing ever leaves L2; these
    // show what the flatness is hiding.
    std::printf(
        "\n  With the cache scaled to the scenes — L1 %zu lines, L2 %zu, not "
        "hardware\n",
        SCALED_L1, SCALED_L2);
    // vs L2 is the whole scene against the cache, which overstates the pressure on
    // any route that does not read the whole scene from every block. Binning is
    // exactly such a route — each block sees one tile's list — and the difference
    // between the two rows below is that, not noise.
    std::printf("  %6s %8s %8s %10s %10s %10s\n", "tris", "route", "scene/L2", "hit rate",
                "misses", "vs cached");
    for (const Row& r : rows) {
        if (!r.scaled.ran) {
            continue;
        }
        std::printf("  %6u %8s %7.2fx %9.1f%% %10s %9.1f%%\n", r.triangles, r.route,
                    static_cast<double>(working_lines(r.triangles)) /
                        static_cast<double>(SCALED_L2),
                    r.scaled.hit_rate(), with_commas(r.scaled.misses).c_str(),
                    change(r.cached.weighted, r.scaled.weighted));
    }

    md << "\n## The same sweep against a cache the scenes outgrow\n\n"
       << "L1 " << SCALED_L1 << " lines and L2 " << SCALED_L2
       << ", scaled to these scenes rather than taken from hardware: at the real\n"
          "sizes nothing is ever evicted from L2, and the figures above are flat "
          "because of it.\n\n"
       << "| Triangles | Route | vs L2 | Hit rate | Misses | vs hardware sizes |\n"
       << "|---:|---|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        if (!r.scaled.ran) {
            continue;
        }
        std::snprintf(buf, sizeof(buf),
                      "| %u | %s | %.2fx | %.1f%% | %s | **%.1f%%** |\n", r.triangles,
                      r.route,
                      static_cast<double>(working_lines(r.triangles)) /
                          static_cast<double>(SCALED_L2),
                      r.scaled.hit_rate(), with_commas(r.scaled.misses).c_str(),
                      change(r.cached.weighted, r.scaled.weighted));
        md << buf;
    }

    // --- what the flat model was hiding: ordering a mesh for a vertex cache -----

    const std::string mesh_path = args.text(0, "assets/sphere.obj");
    const Mesh mesh = load_obj(mesh_path);
    const Mesh mixed = shuffled(mesh, 12345);
    const Mesh fixed = optimised_for_cache(mixed, 32);
    const auto acmr = [&mesh](const Mesh& m) {
        return static_cast<double>(simulated_cache_misses(m, 32)) /
               static_cast<double>(mesh.triangle_count());
    };

    std::vector<ReorderRow> reorder;
    for (const size_t l1 : REORDER_L1) {
        reorder.push_back(
            ReorderRow{l1, measure_mesh(mixed, l1), measure_mesh(fixed, l1)});
    }

    // The route that cannot see the reorder, at the size where the walk sees it most.
    // Without it the saving below could be read as the depth ordering that the
    // shuffle also changes, which is what the flat model measured and reported as a
    // thousandth of a percent.
    const size_t control_l1 = 8;
    const Reading tiled_mixed = measure_mesh(mixed, control_l1, true);
    const Reading tiled_fixed = measure_mesh(fixed, control_l1, true);

    std::printf("\n  Ordering %s for a vertex cache, through the indexed walk\n",
                mesh_path.c_str());
    std::printf("  %u vertices in %zu lines, %u triangles, ACMR %.2f -> %.2f\n\n",
                mesh.vertex_count(),
                (static_cast<size_t>(mesh.vertex_count()) * SCREEN_VERTEX_BYTES +
                 CACHE_LINE_BYTES - 1) /
                    CACHE_LINE_BYTES,
                mesh.triangle_count(), acmr(mixed), acmr(fixed));
    std::printf("  %8s %9s %14s %9s %14s %9s %12s\n", "L1 lines", "vertices", "cost",
                "change", "cycles", "change", "L2 hits");
    for (const ReorderRow& r : reorder) {
        std::printf(
            "  %8zu %9zu %14s %8.2f%% %14s %8.2f%% %6s ->%6s\n", r.l1_lines,
            r.l1_lines * CACHE_LINE_BYTES / SCREEN_VERTEX_BYTES,
            with_commas(r.fixed.weighted).c_str(),
            change(r.mixed.weighted, r.fixed.weighted),
            with_commas(r.fixed.cycles).c_str(), change(r.mixed.cycles, r.fixed.cycles),
            with_commas(r.mixed.l2_hits).c_str(), with_commas(r.fixed.l2_hits).c_str());
    }
    std::printf("  %8zu %9s %14s %8.2f%% %14s %8.2f%%   (tiled, de-indexed)\n",
                control_l1, "", with_commas(tiled_fixed.weighted).c_str(),
                change(tiled_mixed.weighted, tiled_fixed.weighted),
                with_commas(tiled_fixed.cycles).c_str(),
                change(tiled_mixed.cycles, tiled_fixed.cycles));

    csv << "\nl1_lines,route,order,weighted,cycles,l1_hits,l2_hits,misses\n";
    const auto reorder_csv = [&csv](size_t l1, const char* route, const char* order,
                                    const Reading& reading) {
        csv << l1 << ',' << route << ',' << order << ',' << reading.weighted << ','
            << reading.cycles << ',' << reading.l1_hits << ',' << reading.l2_hits << ','
            << reading.misses << '\n';
    };
    for (const ReorderRow& r : reorder) {
        reorder_csv(r.l1_lines, "walk", "shuffled", r.mixed);
        reorder_csv(r.l1_lines, "walk", "forsyth", r.fixed);
    }
    reorder_csv(control_l1, "tiled", "shuffled", tiled_mixed);
    reorder_csv(control_l1, "tiled", "forsyth", tiled_fixed);

    std::snprintf(buf, sizeof(buf), "ACMR %.2f to %.2f", acmr(mixed), acmr(fixed));
    md << "\n## Ordering a mesh for a vertex cache, once there is a cache\n\n"
       << mesh_path << ": " << mesh.vertex_count() << " vertices, "
       << mesh.triangle_count()
       << " triangles, shuffled and then reordered by\n"
          "Forsyth's heuristic — "
       << buf
       << ". Both orders draw the same frame. Measured through\n"
          "the indexed walk with L2 held at "
       << SCALED_L2 << " lines and latency modelled.\n\n"
       << "| L1 lines | Vertices held | Cost | | Cycles | | L2 hits |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const ReorderRow& r : reorder) {
        std::snprintf(
            buf, sizeof(buf),
            "| %zu | %zu | %s | **%.2f%%** | %s | **%.2f%%** | %s -> %s |\n", r.l1_lines,
            r.l1_lines * CACHE_LINE_BYTES / SCREEN_VERTEX_BYTES,
            with_commas(r.fixed.weighted).c_str(),
            change(r.mixed.weighted, r.fixed.weighted),
            with_commas(r.fixed.cycles).c_str(), change(r.mixed.cycles, r.fixed.cycles),
            with_commas(r.mixed.l2_hits).c_str(), with_commas(r.fixed.l2_hits).c_str());
        md << buf;
    }
    std::snprintf(buf, sizeof(buf),
                  "| %zu | tiled, de-indexed | %s | %.2f%% | %s | %.2f%% | |\n",
                  control_l1, with_commas(tiled_fixed.weighted).c_str(),
                  change(tiled_mixed.weighted, tiled_fixed.weighted),
                  with_commas(tiled_fixed.cycles).c_str(),
                  change(tiled_mixed.cycles, tiled_fixed.cycles));
    md << buf;

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
