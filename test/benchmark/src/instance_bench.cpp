// The same object drawn many times, and where its transform meets the camera's.
//
// A model matrix has to reach the view-projection somewhere. Fold them once an
// instance and pass 1 does the single MATVEC it always did; leave them apart and
// every vertex does two. V_MATMUL_MAT4_F32 costs four MATVECs, so the arithmetic
// says the fold pays past four vertices an instance.
//
// It does not, and the distance from that prediction is what this measures. The
// composition pass is not its multiply: sixteen floats read and sixteen written
// cost 3,200 against the MATMUL's 64, while pass 1 pays nothing for its matrix
// at all — the constant window hands a block one for the whole warp.
//
// Pass 2 is excluded on purpose. It grows with the instances as well and swamps
// the difference: through a draw route the two arms come out identical to the
// last lane operation.
//
//   ./build/benchmarks/instance_bench                  result/instance.{md,csv}
//   ./build/benchmarks/instance_bench --instances 64   how many copies
//   ./build/benchmarks/instance_bench --machine <file> latencies come from it
//   ./build/benchmarks/instance_bench --out dir        writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "isa.hpp"
#include "pipeline/draw.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t WIDTH = 64;
constexpr uint32_t HEIGHT = 32;

std::vector<Instance> spread(uint32_t count)
{
    std::vector<Instance> instances;
    for (uint32_t i = 0; i < count; ++i) {
        Float4x4 model = Float4x4::identity();
        model.at(0, 3) = -0.8f + static_cast<float>(i % 16) * 0.11f;
        model.at(1, 3) = static_cast<float>(i % 4) * 0.2f - 0.3f;
        instances.push_back(Instance{model});
    }
    return instances;
}

// A run of vertices, which is all pass 1 looks at. Their positions do not matter
// to the cost — every lane does the same work whatever it reads.
std::vector<Float3> vertices(uint32_t count)
{
    std::vector<Float3> world;
    world.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        world.push_back(Float3{static_cast<float>(i) * 0.01f - 0.3f,
                               static_cast<float>(i % 7) * 0.05f, 0.0f});
    }
    return world;
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
    uint32_t vertices = 0;
    uint64_t per_vertex = 0;
    uint64_t composed = 0;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "instance";
    MACHINE = machine_from(args);
    const uint32_t count = args.flag("instances", 16);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    const DrawTarget target{WIDTH, HEIGHT, Camera{}};
    const std::vector<Instance> instances = spread(count);

    std::printf("\n[BENCH] where a model matrix meets the view-projection\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf("  %u instances, pass 1 only\n", count);
    std::printf("  a MATMUL costs %u, a MATVEC %u, a global access %u\n\n",
                instruction_cost(Opcode::V_MATMUL_MAT4_F32),
                instruction_cost(Opcode::V_MATVEC_MAT4_F32),
                instruction_cost(Opcode::V_LD_GLOBAL_F32));

    std::vector<Row> rows;
    std::printf("  %9s | %13s %13s | %s\n", "vertices", "per-vertex", "composed",
                "composing");
    for (const uint32_t v : {8u, 32u, 64u, 96u, 128u, 256u, 512u}) {
        const std::vector<Float3> world = vertices(v);
        Row row;
        row.vertices = v;
        row.per_vertex =
            instanced_vertex_cost(world, target, instances, InstanceTransform::PerVertex)
                .weighted_lane_ops;
        row.composed = instanced_vertex_cost(world, target, instances,
                                             InstanceTransform::ComposePass)
                           .weighted_lane_ops;
        rows.push_back(row);
        std::printf(
            "  %9u | %13s %13s | %+8.1f%%\n", v, with_commas(row.per_vertex).c_str(),
            with_commas(row.composed).c_str(), change(row.per_vertex, row.composed));
    }

    std::printf(
        "\n  The arithmetic predicted a crossing at four vertices an "
        "instance. What\n  the pass actually spends is 32 global accesses "
        "around one multiply, and\n  pass 1 pays nothing for its matrix — "
        "so the crossing is two orders out.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "instance_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "instances,vertices,transform,weighted_lane_ops\n";
    for (const Row& r : rows) {
        csv << count << ',' << r.vertices << ",per-vertex," << r.per_vertex << '\n';
        csv << count << ',' << r.vertices << ",composed," << r.composed << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/instance_bench; do not edit by hand -->\n\n"
       << "## Where a model matrix meets the view-projection\n\n"
       << count
       << " instances, pass 1 only — pass 2 grows with the instances too and "
          "swamps the\ndifference. A MATMUL costs "
       << instruction_cost(Opcode::V_MATMUL_MAT4_F32) << ", a MATVEC "
       << instruction_cost(Opcode::V_MATVEC_MAT4_F32) << ", a global access "
       << instruction_cost(Opcode::V_LD_GLOBAL_F32) << ".\n\n"
       << "| Vertices an instance | Per-vertex | Composed | Composing |\n"
       << "|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(buf, sizeof(buf), "| %u | %s | %s | **%+.1f%%** |\n", r.vertices,
                      with_commas(r.per_vertex).c_str(), with_commas(r.composed).c_str(),
                      change(r.per_vertex, r.composed));
        md << buf;
    }
    md << "\nThe arithmetic predicts a crossing at four vertices an instance: a "
          "MATMUL is\nfour MATVECs, so folding once should beat an extra MATVEC "
          "at every vertex past\nfour of them. It is two orders out, because the "
          "pass is not its multiply —\nsixteen floats read and sixteen written "
          "cost 3,200 around a MATMUL of 64, and\npass 1 pays nothing at all for "
          "its matrix, the constant window handing a block\none for the whole "
          "warp.\n";

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
