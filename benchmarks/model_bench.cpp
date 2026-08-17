// The two conclusions this project reached under the flat cost model, put to the
// others.
//
// Both were measured when a global access cost a fixed amount per lane and no
// instruction waited on another, and both are about memory traffic — so both had
// reason to move once the model could see lines, cache levels and time. Neither
// was re-measured with the model that displaced it, which is the gap this fills.
//
//   Branch against blend. Predication lost on every route: about 2% on the
//   rasteriser and 4.4% on the ray tracer. It removes divergence and pays for it
//   in work the branch would have skipped, so the question here is whether
//   divergence gets dearer when a warp can be made to wait.
//
//   Indexed against flattened. Carrying the index buffer into pass 2 costs 24%,
//   being three dependent loads a triangle before a vertex address is known.
//   Dependent is the word that should matter more once results take time to
//   arrive.
//
// Everything is measured on the routes and scenes render_bench already uses, so
// the Flat column here reproduces the tables there and the rest are read against
// it.
//
//   ./build/benchmarks/model_bench            benchmarks/result/models.{md,csv}
//   ./build/benchmarks/model_bench --out dir  writes into that directory
//
// Run it from the project root: the mesh comparison reads assets/sphere.obj, and
// says so and carries on if it is not there.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "app_run.hpp"
#include "math3d.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"
#include "scenes.hpp"

namespace {

// The four rulers, in the order the project acquired them. Cycles only mean
// anything in the last, where a warp that has issued waits for its result.
struct Model {
    const char* name;
    MemoryModel memory;
    LatencyModel latency;
    bool times;  // read the cycle count rather than the issue count
};

constexpr Model MODELS[] = {
    {"flat", MemoryModel::Flat, LatencyModel::Ignored, false},
    {"coalesced", MemoryModel::Coalesced, LatencyModel::Ignored, false},
    {"cached", MemoryModel::Cached, LatencyModel::Ignored, false},
    {"cycles", MemoryModel::Cached, LatencyModel::Modelled, true},
};

constexpr size_t MODEL_COUNT = sizeof(MODELS) / sizeof(MODELS[0]);

struct Reading {
    uint64_t cost = 0;  // weighted lane ops, or cycles where the model times
    double divergence = 0.0;
    bool ran = true;
};

// What a draw costs under one model. The draw routes clear their counters at the
// sync between the passes, so this reads pass 2 alone — which is where both
// comparisons live.
template <typename Draw>
Reading measure(const Draw& draw, const Model& model)
{
    MyGPURuntime rt(1u << 27);
    rt.myrt_set_memory_model(model.memory);
    rt.myrt_set_latency_model(model.latency);

    Reading r;
    try {
        draw(rt);
    } catch (const std::exception&) {
        // A tile too full to stage, which is the shared route's limit and a
        // result rather than a failure. Nothing below asks that route, but a
        // scene added later might.
        r.ran = false;
        return r;
    }
    r.cost = model.times ? rt.stats().cycles : rt.stats().weighted_lane_ops;
    r.divergence = rt.divergence_rate();
    return r;
}

double change(uint64_t from, uint64_t to)
{
    return from == 0 ? 0.0
                     : 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
                           static_cast<double>(from);
}

// One comparison under every model: a baseline and the variant being asked about.
struct Comparison {
    std::string subject;
    std::string variant;
    uint32_t triangles = 0;
    double delta[MODEL_COUNT] = {};
    double base_divergence[MODEL_COUNT] = {};
    double variant_divergence[MODEL_COUNT] = {};
    uint64_t base_cost[MODEL_COUNT] = {};
    uint64_t variant_cost[MODEL_COUNT] = {};
};

template <typename Base, typename Variant>
Comparison compare(std::string subject, std::string variant, uint32_t triangles,
                   const Base& base, const Variant& alternative)
{
    Comparison c;
    c.subject = std::move(subject);
    c.variant = std::move(variant);
    c.triangles = triangles;
    for (size_t i = 0; i < MODEL_COUNT; ++i) {
        const Reading b = measure(base, MODELS[i]);
        const Reading v = measure(alternative, MODELS[i]);
        c.base_cost[i] = b.cost;
        c.variant_cost[i] = v.cost;
        c.delta[i] = change(b.cost, v.cost);
        c.base_divergence[i] = 100.0 * b.divergence;
        c.variant_divergence[i] = 100.0 * v.divergence;
    }
    return c;
}

void print_table(const char* title, const char* legend,
                 const std::vector<Comparison>& rows)
{
    std::printf("\n  %s\n  %s\n\n", title, legend);
    std::printf("  %-30s %-10s %5s", "case", "route", "tris");
    for (const Model& m : MODELS) {
        std::printf(" %10s", m.name);
    }
    std::printf("\n");
    for (const Comparison& c : rows) {
        std::printf("  %-30s %-10s %5u", c.subject.c_str(), c.variant.c_str(),
                    c.triangles);
        for (size_t i = 0; i < MODEL_COUNT; ++i) {
            std::printf(" %9.1f%%", c.delta[i]);
        }
        std::printf("\n");
    }
}

void write_table(std::ofstream& md, const char* title, const char* legend,
                 const std::vector<Comparison>& rows)
{
    md << "\n## " << title << "\n\n" << legend << "\n\n| Case | Route | Triangles |";
    for (const Model& m : MODELS) {
        md << ' ' << m.name << " |";
    }
    md << "\n|---|---|---:|";
    for (size_t i = 0; i < MODEL_COUNT; ++i) {
        md << "---:|";
    }
    md << '\n';

    char buf[256];
    for (const Comparison& c : rows) {
        md << "| " << c.subject << " | " << c.variant << " | " << c.triangles << " |";
        for (size_t i = 0; i < MODEL_COUNT; ++i) {
            std::snprintf(buf, sizeof(buf), " %+.1f%% |", c.delta[i]);
            md << buf;
        }
        md << '\n';
    }
}

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "models";

    std::printf("\n[BENCH] the flat model's conclusions, put to the others — %ux%u\n",
                BENCH_WIDTH, BENCH_HEIGHT);

    // --- branch against blend -----------------------------------------------

    const std::pair<const char*, std::vector<Float3>> scenes[] = {
        {"small, spread over the frame", spread(4)},
        {"full-frame, stacked", stacked(16, 4.0f)},
    };

    std::vector<Comparison> predication;
    // Named rather than structured-bound: a lambda below captures them, and
    // capturing a structured binding is C++20.
    for (const auto& entry : scenes) {
        const char* const name = entry.first;
        const std::vector<Float3>& world = entry.second;
        const uint32_t triangles = static_cast<uint32_t>(world.size() / 3);
        const DrawTarget t = bench_target();

        predication.push_back(compare(
            name, "walk", triangles,
            [&](MyGPURuntime& rt) { return draw_walk(rt, world, t, false); },
            [&](MyGPURuntime& rt) { return draw_walk(rt, world, t, true); }));
        predication.push_back(compare(
            name, "tiled", triangles,
            [&](MyGPURuntime& rt) { return draw_tiled(rt, world, t, false); },
            [&](MyGPURuntime& rt) { return draw_tiled(rt, world, t, true); }));
        predication.push_back(compare(
            name, "raytrace", triangles,
            [&](MyGPURuntime& rt) {
                return draw_raytrace(rt, world, t, Shading{}, false);
            },
            [&](MyGPURuntime& rt) {
                return draw_raytrace(rt, world, t, Shading{}, true);
            }));
    }

    print_table("Branch against blend", "positive = the blend costs more", predication);

    std::printf(
        "\n  Divergence, branch against blend (the blend's is the coverage "
        "test's alone)\n\n");
    std::printf("  %-30s %-10s %10s %10s\n", "case", "route", "branch", "blend");
    for (const Comparison& c : predication) {
        std::printf("  %-30s %-10s %9.2f%% %9.2f%%\n", c.subject.c_str(),
                    c.variant.c_str(), c.base_divergence[0], c.variant_divergence[0]);
    }

    // --- indexed against flattened ------------------------------------------

    std::vector<Comparison> indexed;
    std::vector<std::pair<std::string, Mesh>> meshes;
    meshes.emplace_back("cube", cube_mesh());
    try {
        meshes.emplace_back("sphere", load_obj("assets/sphere.obj"));
    } catch (const std::exception& e) {
        std::printf("\n  (skipping the sphere: %s)\n", e.what());
    }

    for (const auto& entry : meshes) {
        const std::string& name = entry.first;
        const Mesh& mesh = entry.second;
        const DrawTarget t = bench_target();
        const std::vector<Float3> flat = mesh.flattened();
        indexed.push_back(compare(
            name, "walk", mesh.triangle_count(),
            [&](MyGPURuntime& rt) { return draw_walk(rt, flat, t, false); },
            [&](MyGPURuntime& rt) { return draw_walk(rt, mesh, t, false); }));
    }

    print_table("Indexed against flattened, pass 2",
                "positive = the index buffer costs more", indexed);

    // --- a depth prepass against a single pass -----------------------------

    // Stacked back to front, which is the walk's worst case and the prepass's
    // best: every triangle covers every pixel and each is nearer than the last,
    // so the walk shades all of them and early-Z shades one.
    //
    // Both shading modes, because what decides this trade is not depth
    // complexity but how much the shade costs against the coverage test that
    // finds it — and those two modes sit either side of a fiftyfold difference.
    std::vector<Comparison> prepass;
    for (const ShadingMode mode : {ShadingMode::Barycentric, ShadingMode::Diffuse}) {
        Shading shading;
        shading.mode = mode;
        const char* const name =
            mode == ShadingMode::Diffuse ? "stacked, lit" : "stacked, barycentric";

        for (const uint32_t complexity : {4u, 8u, 16u, 32u}) {
            const std::vector<Float3> world = stacked_back_to_front(complexity);
            const DrawTarget t = bench_target();

            prepass.push_back(compare(
                name, "early-Z", complexity,
                [&](MyGPURuntime& rt) {
                    DeviceGeometry g = upload(rt, world);
                    DeviceFrame f = allocate_frame(rt, t);
                    const std::vector<Float3> frame =
                        draw_walk(rt, g, f, t, false, shading);
                    release(rt, f);
                    release(rt, g);
                    return frame;
                },
                [&](MyGPURuntime& rt) {
                    DeviceGeometry g = upload(rt, world);
                    DeviceFrame f = allocate_frame(rt, t);
                    const std::vector<Float3> frame = draw_early_z(rt, g, f, t, shading);
                    release(rt, f);
                    release(rt, g);
                    return frame;
                }));
        }
    }

    print_table("A depth prepass against a single pass",
                "positive = the prepass costs more", prepass);

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "model_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "comparison,case,route,triangles,model,base,variant,change_pct\n";
    // The scene names hold commas, so that field is quoted. Unquoted, a reader
    // splits one row into nine fields and blames its own parser.
    const auto rows_to_csv = [&csv](const char* what,
                                    const std::vector<Comparison>& rows) {
        for (const Comparison& c : rows) {
            for (size_t i = 0; i < MODEL_COUNT; ++i) {
                csv << what << ",\"" << c.subject << "\"," << c.variant << ','
                    << c.triangles << ',' << MODELS[i].name << ',' << c.base_cost[i]
                    << ',' << c.variant_cost[i] << ',' << c.delta[i] << '\n';
            }
        }
    };
    rows_to_csv("predication", predication);
    rows_to_csv("indexed", indexed);
    rows_to_csv("prepass", prepass);

    std::ofstream md(prefix + ".md");
    md << "<!-- generated by benchmarks/model_bench; do not edit by hand -->\n";
    write_table(md, "Branch against blend, under every cost model",
                "Positive means the blend costs more. `flat` reproduces the table in "
                "*Branch against blend*;\n`cycles` is time under `Cached` and "
                "`LatencyModel::Modelled` rather than issue capacity.",
                predication);
    write_table(md, "Indexed against flattened, pass 2, under every cost model",
                "Positive means carrying the index buffer into pass 2 costs more.",
                indexed);
    write_table(md, "A depth prepass against a single pass",
                "Positive means the prepass costs more. Stacked back to front, so the "
                "walk shades\nevery triangle and early-Z shades one — the case a prepass "
                "exists for.",
                prepass);

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
