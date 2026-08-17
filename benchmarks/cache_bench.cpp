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
//   ./build/benchmarks/cache_bench            benchmarks/result/cache.{md,csv}
//   ./build/benchmarks/cache_bench --out dir  writes into that directory

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "math3d.hpp"
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

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
