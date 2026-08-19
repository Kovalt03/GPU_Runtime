// A block produces, its neighbours consume, and there are two ways to get it to
// them.
//
// Without clusters a block cannot see another's shared memory, so anything one
// block makes for another has to go out to global memory and come back — and
// since there is no block-wide rendezvous inside a launch, it has to cross a
// kernel boundary as well. A cluster removes both: the blocks are placed
// together, BARRIER_CLUSTER is the rendezvous, and V_LD_CLUSTER_F32 is the read.
//
// The two routes are not the same instructions, which is the point. What is
// held constant is the work: the same 256 elements are produced by one block and
// summed by four.
//
//   ./build/benchmarks/cluster_bench                     result/cluster.{md,csv}
//   ./build/benchmarks/cluster_bench --machine <file>    latencies come from it
//   ./build/benchmarks/cluster_bench --out dir           writes into that directory

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "gpu_spec.hpp"
#include "ir_builder.hpp"
#include "runtime.hpp"

namespace {

GPUSpec MACHINE;

constexpr uint32_t FLOATS = 256;  // what the producer makes
constexpr uint32_t WORK = 8;      // how expensive one element is to make

// Produce into shared memory (cluster) or into global memory (the other way).
Program produce(size_t out, bool to_shared)
{
    IRBuilder k;
    const Reg<Scalar> four = k.constant(4.0f);
    const Reg<Scalar> byte = k.mul(k.thread_x(), four);
    const Reg<Scalar> one = k.constant(1.0f);
    for (uint32_t i = 0; i < FLOATS; i += WARP_SIZE) {
        const Reg<Scalar> at = k.add(byte, k.constant(static_cast<float>(i * 4)));
        Reg<Scalar> v = k.constant(1.0f);
        for (uint32_t w = 0; w < WORK; ++w) {
            k.fma(v, v, one);
        }
        if (to_shared) {
            k.store_shared(at, v);
        } else {
            k.store(k.add(k.constant(static_cast<float>(out)), at), v);
        }
    }
    return k.build();
}

Program consume(size_t in, size_t out, bool from_cluster, bool producer_too)
{
    IRBuilder k;
    const Reg<Scalar> four = k.constant(4.0f);
    const Reg<Scalar> byte = k.mul(k.thread_x(), four);
    const Reg<Scalar> zero = k.constant(0.0f);
    const Reg<Scalar> one = k.constant(1.0f);

    if (producer_too) {
        // Rank 0 makes it, into its own shared memory.
        k.if_(k.lt(k.cluster_rank(), one), [&] {
            for (uint32_t i = 0; i < FLOATS; i += WARP_SIZE) {
                const Reg<Scalar> at = k.add(byte, k.constant(static_cast<float>(i * 4)));
                Reg<Scalar> v = k.constant(1.0f);
                for (uint32_t w = 0; w < WORK; ++w) {
                    k.fma(v, v, one);
                }
                k.store_shared(at, v);
            }
        });
        k.barrier_cluster();
    }

    Reg<Scalar> sum = k.constant(0.0f);
    for (uint32_t i = 0; i < FLOATS; i += WARP_SIZE) {
        const Reg<Scalar> at = k.add(byte, k.constant(static_cast<float>(i * 4)));
        const Reg<Scalar> v = from_cluster
                                  ? k.load_cluster(at, zero)
                                  : k.load(k.add(k.constant(static_cast<float>(in)), at));
        k.fma(sum, v, one);
    }
    k.store(k.constant(static_cast<float>(out)), sum);
    return k.build();
}

struct Reading {
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    uint32_t launches = 0;
};

Reading measure(uint32_t blocks, bool clustered)
{
    MyGPURuntime rt(1u << 22);
    GPUSpec spec = MACHINE;
    spec.sms.sm_count = blocks;
    spec.sms.blocks_per_sm = 1;
    rt.myrt_set_spec(spec);
    rt.myrt_set_memory_model(MemoryModel::Cached);
    rt.myrt_set_latency_model(LatencyModel::Modelled);

    void* buf = rt.myrt_malloc(FLOATS * sizeof(float));
    void* out = rt.myrt_malloc(64 * sizeof(float));
    const size_t in_at = rt.myrt_device_offset(buf);
    const size_t out_at = rt.myrt_device_offset(out);

    uint32_t launches = 0;
    if (clustered) {
        const Program p = consume(in_at, out_at, true, true);
        LaunchConfig config{dim3{blocks, 1, 1}, dim3{WARP_SIZE, 1, 1}};
        config.cluster_size = blocks;
        rt.myrt_launch_async([p](void**) { return p; }, config, nullptr);
        launches = 1;
    } else {
        const Program a = produce(in_at, false);
        const Program b = consume(in_at, out_at, false, false);
        rt.myrt_launch_async([a](void**) { return a; },
                             LaunchConfig{dim3{1, 1, 1}, dim3{WARP_SIZE, 1, 1}}, nullptr);
        rt.myrt_launch_async([b](void**) { return b; },
                             LaunchConfig{dim3{blocks, 1, 1}, dim3{WARP_SIZE, 1, 1}},
                             nullptr);
        launches = 2;
    }
    rt.myrt_wait();
    return Reading{rt.stats().weighted_lane_ops, rt.stats().cycles, launches};
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
    uint32_t blocks = 0;
    Reading global;
    Reading cluster;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "cluster";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] one block produces, its neighbours consume\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());
    std::printf("  %u elements made once and summed by every block\n\n", FLOATS);
    std::printf("  %8s %14s %14s %10s %12s %12s %10s\n", "blocks", "global work",
                "cluster work", "change", "global", "cluster", "change");

    std::vector<Row> rows;
    for (const uint32_t blocks : {2u, 4u, 8u}) {
        const Row row{blocks, measure(blocks, false), measure(blocks, true)};
        rows.push_back(row);
        std::printf("  %8u %14s %14s %9.1f%% %12llu %12llu %9.1f%%\n", blocks,
                    with_commas(row.global.weighted).c_str(),
                    with_commas(row.cluster.weighted).c_str(),
                    change(row.global.weighted, row.cluster.weighted),
                    (unsigned long long)row.global.cycles,
                    (unsigned long long)row.cluster.cycles,
                    change(row.global.cycles, row.cluster.cycles));
    }

    std::printf(
        "\n  The global route needs two launches: a block-wide rendezvous is a "
        "kernel\n  boundary when the blocks cannot see each other.\n");

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "cluster_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "blocks,route,launches,weighted_lane_ops,cycles\n";
    for (const Row& r : rows) {
        csv << r.blocks << ",global," << r.global.launches << ',' << r.global.weighted
            << ',' << r.global.cycles << '\n';
        csv << r.blocks << ",cluster," << r.cluster.launches << ',' << r.cluster.weighted
            << ',' << r.cluster.cycles << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[320];
    md << "<!-- generated by benchmarks/cluster_bench; do not edit by hand -->\n\n"
       << "## One block produces, its neighbours consume\n\n"
       << FLOATS
       << " elements made once and summed by every block. The global route "
          "takes two\nlaunches, because a block-wide rendezvous is a kernel boundary "
          "when the blocks\ncannot see each other.\n\n"
       << "| Blocks | Global work | cluster | change | Global cycles | cluster | "
          "change |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Row& r : rows) {
        std::snprintf(
            buf, sizeof(buf), "| %u | %s | %s | %.1f%% | %llu | %llu | **%.1f%%** |\n",
            r.blocks, with_commas(r.global.weighted).c_str(),
            with_commas(r.cluster.weighted).c_str(),
            change(r.global.weighted, r.cluster.weighted),
            (unsigned long long)r.global.cycles, (unsigned long long)r.cluster.cycles,
            change(r.global.cycles, r.cluster.cycles));
        md << buf;
    }

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
