// Reordering the threads, on divergence nobody put there to be reordered.
//
// ser_bench measures REORDER on a key built for the purpose: a number chosen to
// be scattered or coherent, and a branch chosen to be long or short. It closed
// at -75% and +13%, and the objection to it was always the same — the answer
// might be a property of the key rather than of the technique.
//
// This is the same instruction on a ray tracer. The key is the material of
// whichever instance the ray hit, which the scene decides and nothing here
// arranges. Two neighbouring pixels are two rays, they meet different objects,
// and a shader with an arm per material splits the warp along that seam.
//
// Three arms rather than two, because two changes are involved and reporting
// their sum would credit the wrong one:
//
//   inline    shade every candidate that beats the running best, blend the
//             losers away — what this route did before there was a reason not to
//   deferred  keep four scalars and shade once at the end
//   reordered the same, with the block's threads regrouped by material first
//
//   ./build/benchmarks/material_bench                   result/material.{md,csv}
//   ./build/benchmarks/material_bench --materials 8     how many arms
//   ./build/benchmarks/material_bench --machine <file>  latencies come from it
//   ./build/benchmarks/material_bench --out dir         writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;
constexpr uint32_t INSTANCES = 64;

// A block four rows tall, so REORDER has warps to move threads between. One row
// is one warp and the instruction then regroups nothing — which is why every
// figure taken before this one is unaffected by it existing.
constexpr uint32_t BLOCK_ROWS = 4;

std::vector<Float3> geometry(uint32_t triangles, uint32_t seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> place(-0.08f, 0.08f);
    std::uniform_real_distribution<float> size(0.02f, 0.05f);

    std::vector<Float3> world;
    for (uint32_t i = 0; i < triangles; ++i) {
        const Float3 centre{place(rng), place(rng), place(rng)};
        const float s = size(rng);
        world.push_back(centre + Float3{-s, -s, 0.0f});
        world.push_back(centre + Float3{s, -s, 0.0f});
        world.push_back(centre + Float3{0.0f, s, s});
    }
    return world;
}

// Materials interleaved across the grid, so neighbouring pixels land on
// different ones. The scene's own arrangement rather than a scattering chosen to
// make a point — but it is worth saying that a scene sorted by material would
// give the reorder nothing, which is the coherent half of ser_bench's table.
std::vector<TlasInstance> scene(uint32_t materials)
{
    std::vector<TlasInstance> instances;
    for (uint32_t i = 0; i < INSTANCES; ++i) {
        Float4x4 model = Float4x4::identity();
        model.at(0, 3) = -0.9f + static_cast<float>(i % 16) * 0.12f;
        model.at(1, 3) = static_cast<float>(i / 16) * 0.2f - 0.3f;
        instances.push_back(TlasInstance{0, model, i % materials});
    }
    return instances;
}

// One arm a material, each a different length. `weight` is how much work an arm
// does, which is the axis: a reorder pays for the divergent stretch that follows
// it, so a short one cannot repay the sort.
Shading arms(uint32_t materials, uint32_t weight)
{
    Shading shading;
    shading.mode = ShadingMode::Custom;
    shading.shade = [materials, weight](IRBuilder& k, const Fragment& f) {
        const Reg<Scalar> step = k.constant(0.01f);
        const Reg<Scalar> half = k.constant(0.5f);
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> one = k.constant(1.0f);
        for (uint32_t m = 0; m < materials; ++m) {
            const Reg<Scalar> from_here =
                k.sub(f.material, k.constant(static_cast<float>(m) - 0.5f));
            k.if_(k.min(k.gt(from_here, zero), k.lt(from_here, one)), [&] {
                Reg<Scalar> acc = k.constant(0.1f * static_cast<float>(m + 1));
                for (uint32_t w = 0; w < (m + 1) * weight; ++w) {
                    k.fma(acc, f.w1, step);
                }
                k.copy_into(f.out.component(0), k.min(acc, half));
                k.copy_into(f.out.component(1), f.w1);
                k.copy_into(f.out.component(2), f.w2);
            });
        }
    };
    return shading;
}

struct Reading {
    uint64_t issued = 0;
    double divergence = 0.0;
};

Reading trace(const std::vector<Float3>& world,
              const std::vector<TlasInstance>& instances, const Shading& shading,
              ShadeWhen when)
{
    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    MyGPURuntime rt(1u << 27);
    rt.myrt_set_spec(MACHINE);

    DeviceGeometry g = upload_scene(rt, {world}, instances);
    DeviceFrame frame = allocate_frame(rt, target);
    draw_raytrace(rt, g, frame, target, shading, false, when,
                  when == ShadeWhen::DeferredReordered ? BLOCK_ROWS : 1u);

    Reading r;
    r.issued = rt.stats().warp_steps;
    r.divergence = rt.divergence_rate() * 100.0;
    release(rt, frame);
    release(rt, g);
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

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
}

struct Row {
    uint32_t weight = 0;
    Reading inlined;
    Reading deferred;
    Reading reordered;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "material";
    MACHINE = machine_from(args);
    const uint32_t materials = args.flag("materials", 4);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    const std::vector<Float3> world = geometry(8);
    const std::vector<TlasInstance> instances = scene(materials);

    std::printf("\n[BENCH] reordering on divergence the scene put there\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf("  %ux%u, %u instances, %u materials, block %u rows\n\n", WIDTH, HEIGHT,
                INSTANCES, materials, BLOCK_ROWS);

    std::vector<Row> rows;
    std::printf("  %7s | %11s %6s | %11s %6s | %11s %6s | %8s %8s\n", "arm work",
                "inline", "div", "deferred", "div", "reordered", "div", "deferring",
                "reorder");
    for (const uint32_t weight : {1u, 4u, 16u, 64u, 256u}) {
        const Shading shading = arms(materials, weight);
        Row row;
        row.weight = weight;
        row.inlined = trace(world, instances, shading, ShadeWhen::Inline);
        row.deferred = trace(world, instances, shading, ShadeWhen::Deferred);
        row.reordered = trace(world, instances, shading, ShadeWhen::DeferredReordered);
        rows.push_back(row);

        std::printf(
            "  %7u | %11s %5.1f%% | %11s %5.1f%% | %11s %5.1f%% | %+7.1f%% "
            "%+7.1f%%\n",
            weight, with_commas(row.inlined.issued).c_str(), row.inlined.divergence,
            with_commas(row.deferred.issued).c_str(), row.deferred.divergence,
            with_commas(row.reordered.issued).c_str(), row.reordered.divergence,
            change(row.inlined.issued, row.deferred.issued),
            change(row.deferred.issued, row.reordered.issued));
    }

    std::printf(
        "\n  The reorder repays the divergent stretch that follows it and "
        "nothing\n  before it. Traversal divergence is upstream — a ray "
        "tracer's largest, and\n  the one reordering cannot touch.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "material_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "materials,arm_work,shade_when,issued,divergence_percent\n";
    for (const Row& r : rows) {
        const auto line = [&](const char* name, const Reading& x) {
            csv << materials << ',' << r.weight << ',' << name << ',' << x.issued << ','
                << x.divergence << '\n';
        };
        line("inline", r.inlined);
        line("deferred", r.deferred);
        line("reordered", r.reordered);
    }

    std::ofstream md(prefix + ".md");
    char buf[420];
    md << "<!-- generated by benchmarks/material_bench; do not edit by hand -->\n\n"
       << "## Reordering on divergence the scene put there\n\n"
       << WIDTH << "x" << HEIGHT << ", " << INSTANCES << " instances, " << materials
       << " materials interleaved, block " << BLOCK_ROWS
       << " rows.\nThe key is the material of whichever instance the ray hit.\n\n"
       << "| Arm work | Inline | div | Deferred | div | Reordered | div | Deferring "
          "| Reordering |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf),
                      "| %u | %s | %.1f%% | %s | %.1f%% | %s | %.1f%% | %+.1f%% | "
                      "**%+.1f%%** |\n",
                      r.weight, with_commas(r.inlined.issued).c_str(),
                      r.inlined.divergence, with_commas(r.deferred.issued).c_str(),
                      r.deferred.divergence, with_commas(r.reordered.issued).c_str(),
                      r.reordered.divergence, change(r.inlined.issued, r.deferred.issued),
                      change(r.deferred.issued, r.reordered.issued));
        md << buf;
    }
    md << "\nThe reorder repays the divergent stretch that follows it and nothing "
          "before\nit. Traversal divergence is upstream of the rendezvous — a ray "
          "tracer's\nlargest, and the one reordering cannot touch.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
