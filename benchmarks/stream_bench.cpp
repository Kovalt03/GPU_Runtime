// What a second queue buys, and what a launch the host never sized costs.
//
// Every other benchmark here hands the machine one kernel at a time. A stream is
// the first thing that lets two arrive at once, and the two questions it raises
// are the ones below:
//
//   Filling.    A grid smaller than the machine leaves SMs with nothing to do.
//               Another stream's blocks can have them, and the same total work
//               then retires in a fraction of the cycles.
//   Covering.   Two kernels on one SM cover each other's waiting, exactly as two
//               blocks of one kernel do. What is new is that they need not be
//               the same kernel — and a kernel that waits pairs best with one
//               that does not.
//
// And one that is not about streams at all, but rides in with them: an indirect
// launch takes its grid from device memory, so a kernel can decide how much work
// the kernel after it does. The cost of asking is what this measures.
//
//   ./build/benchmarks/stream_bench                     result/stream.{md,csv}
//   ./build/benchmarks/stream_bench --machine <file>    caches and slots come from it
//   ./build/benchmarks/stream_bench --out dir           writes into that directory

#include <algorithm>
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

// The machine this run was asked for. The sweeps below vary the SM count and the
// residency from it; everything else — the caches, the warp slots, the prices —
// is whatever the file said.
GPUSpec MACHINE;

struct Reading {
    uint64_t weighted = 0;
    uint64_t cycles = 0;
    uint64_t stalls = 0;
};

// Scratch every kernel addresses. Wide enough that the memory kernel's loads
// fall in distinct lines, which is what makes them all miss, and split so that
// two concurrent kernels read different ones: sharing a line would mean one of
// them finding what the other had just fetched, and that is a cache result
// rather than a stream result.
constexpr uint32_t SCRATCH_BYTES = 256 * 1024;
constexpr uint32_t SCRATCH_STRIDE = 8 * 1024;
constexpr uint32_t LOADS = 16;
constexpr uint32_t FMAS = 16;

// A warp that waits. Every load is a line of its own, so none of them is in a
// cache when it is asked for and each costs a trip to memory.
Program memory_kernel(size_t base)
{
    IRBuilder k;
    Reg<Scalar> addr = k.add(k.constant(static_cast<float>(base)),
                             k.mul(k.thread_x(), k.constant(4.0f)));
    Reg<Scalar> acc = k.constant(0.0f);
    Reg<Scalar> one = k.constant(1.0f);
    for (uint32_t i = 0; i < LOADS; ++i) {
        k.fma(acc, k.load(addr, static_cast<float>(i * CACHE_LINE_BYTES)), one);
    }
    k.store(addr, acc);
    return k.build();
}

// A warp that does not. The same number of instructions, none of which sends the
// warp to memory.
Program compute_kernel(size_t base)
{
    IRBuilder k;
    Reg<Scalar> addr = k.add(k.constant(static_cast<float>(base)),
                             k.mul(k.thread_x(), k.constant(4.0f)));
    Reg<Scalar> acc = k.constant(1.0f);
    Reg<Scalar> step = k.constant(1.000001f);
    for (uint32_t i = 0; i < FMAS; ++i) {
        k.fma(acc, acc, step);
    }
    k.store(addr, acc);
    return k.build();
}

// One thread adding up a buffer of ones and zeroes, and writing the total where
// an indirect launch will read its grid.
//
// A loop rather than a warp reduction because the answer has to be one number in
// one place, and this ISA has no atomic to combine lanes with. It is the shape a
// culling pass has: look at everything, and say how much survived.
Program cull_kernel(size_t flags_base, uint32_t flags, size_t grid_offset)
{
    IRBuilder k;
    Reg<Scalar> base = k.constant(static_cast<float>(flags_base));
    Reg<Scalar> limit = k.constant(static_cast<float>(flags));
    Reg<Scalar> four = k.constant(4.0f);
    Reg<Scalar> one = k.constant(1.0f);
    Reg<Scalar> index = k.constant(0.0f);
    Reg<Scalar> survivors = k.constant(0.0f);

    Label top = k.label();
    k.place(top);
    // Accumulating through fma, which writes in place. An add would allocate a
    // fresh register every time it was emitted, and the branch back would then
    // read the total from where the previous iteration had not put it.
    k.fma(survivors, k.load(k.add(base, k.mul(index, four))), one);
    k.fma(index, one, one);
    k.branch_to(top, k.lt(index, limit));

    Reg<Scalar> grid = k.constant(static_cast<float>(grid_offset));
    k.store(grid, survivors);
    k.store(grid, one, 4.0f);
    k.store(grid, one, 8.0f);
    return k.build();
}

enum class Kind { Memory, Compute };

struct Job {
    Kind kind = Kind::Compute;
    uint32_t blocks = 1;
};

MyGPURuntime make_machine(uint32_t sm_count, uint32_t blocks_per_sm)
{
    MyGPURuntime rt(1u << 22);
    GPUSpec spec = MACHINE;
    spec.sms.sm_count = sm_count;
    spec.sms.blocks_per_sm = blocks_per_sm;
    rt.myrt_set_spec(spec);
    rt.myrt_set_memory_model(MemoryModel::Cached);

    // Cycles are the only thing a stream can move, and without latency there is
    // nothing to cover: Ignored makes every warp ready the instant it issues.
    rt.myrt_set_latency_model(LatencyModel::Modelled);
    return rt;
}

// Runs the jobs, each in its own stream or all in one, and reports what the
// machine came to. Same work either way — only the queueing differs.
Reading run(const std::vector<Job>& jobs, bool concurrent, uint32_t sm_count,
            uint32_t blocks_per_sm)
{
    MyGPURuntime rt = make_machine(sm_count, blocks_per_sm);
    void* scratch = rt.myrt_malloc(SCRATCH_BYTES);
    const size_t base = rt.myrt_device_offset(scratch);

    for (size_t i = 0; i < jobs.size(); ++i) {
        const Job& job = jobs[i];
        const size_t region = base + i * SCRATCH_STRIDE;
        const Program prog =
            job.kind == Kind::Memory ? memory_kernel(region) : compute_kernel(region);
        const StreamId stream = concurrent ? rt.myrt_stream_create() : DEFAULT_STREAM;
        rt.myrt_launch_async([prog](void**) { return prog; },
                             LaunchConfig{dim3{job.blocks, 1, 1}, dim3{WARP_SIZE, 1, 1}},
                             nullptr, stream);
    }
    rt.myrt_wait();
    return Reading{rt.stats().weighted_lane_ops, rt.stats().cycles,
                   rt.stats().stall_steps};
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

const char* name_of(Kind kind)
{
    return kind == Kind::Memory ? "memory" : "compute";
}

struct FillRow {
    uint32_t launches = 0;
    uint32_t blocks = 0;
    Reading ordered;
    Reading concurrent;
};

struct PairRow {
    Kind first = Kind::Compute;
    Kind second = Kind::Compute;
    Reading alone_first;
    Reading alone_second;
    Reading ordered;
    Reading concurrent;
};

struct CullRow {
    uint32_t visible = 0;
    uint32_t grid = 0;
    Reading indirect;
    Reading host_side;
};

}  // namespace

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const std::string prefix = args.out_dir + "stream";
    MACHINE = machine_from(args);

    std::ofstream machine_file(args.out_dir + "machine.spec");
    machine_file << MACHINE.to_text();

    std::printf("\n[BENCH] what a second queue buys\n\n");
    std::printf("%s\n", MACHINE.describe().c_str());

    // --- filling a machine one grid cannot -----------------------------------

    // Eight blocks of work however it is cut up, on eight SMs holding one block
    // each. Ordered, a launch of one block uses one SM and the other seven wait
    // their turn; concurrent, all eight are somewhere at once.
    constexpr uint32_t TOTAL_BLOCKS = 8;
    constexpr uint32_t SMS = 8;

    std::printf(
        "  %u blocks of compute on %u SMs, cut into launches. Same work every "
        "row\n\n",
        TOTAL_BLOCKS, SMS);
    std::printf("  %9s %10s %12s %12s %9s\n", "launches", "blocks", "one stream",
                "n streams", "speedup");

    std::vector<FillRow> fill;
    for (const uint32_t launches : {1u, 2u, 4u, 8u}) {
        const uint32_t blocks = TOTAL_BLOCKS / launches;
        const std::vector<Job> jobs(launches, Job{Kind::Compute, blocks});
        const FillRow row{launches, blocks, run(jobs, false, SMS, 1),
                          run(jobs, true, SMS, 1)};
        fill.push_back(row);
        std::printf("  %9u %10u %12s %12s %8.2fx\n", launches, blocks,
                    with_commas(row.ordered.cycles).c_str(),
                    with_commas(row.concurrent.cycles).c_str(),
                    static_cast<double>(row.ordered.cycles) /
                        static_cast<double>(row.concurrent.cycles));
    }

    // --- covering one kernel's waiting with another --------------------------

    std::printf("\n  Two launches on one SM holding two blocks, in cycles\n\n");
    std::printf("  %9s %9s %10s %10s %12s %12s %9s\n", "first", "second", "1st alone",
                "2nd alone", "one stream", "two streams", "hidden");

    std::vector<PairRow> pairs;
    for (const auto& [a, b] :
         std::vector<std::pair<Kind, Kind>>{{Kind::Compute, Kind::Compute},
                                            {Kind::Memory, Kind::Memory},
                                            {Kind::Memory, Kind::Compute}}) {
        const std::vector<Job> jobs{Job{a, 1}, Job{b, 1}};
        const PairRow row{a,
                          b,
                          run({Job{a, 1}}, false, 1, 2),
                          run({Job{b, 1}}, false, 1, 2),
                          run(jobs, false, 1, 2),
                          run(jobs, true, 1, 2)};
        pairs.push_back(row);

        // What concurrency took off the shorter of the two, which is the most it
        // could have taken: nothing can retire sooner than the longer kernel does
        // on its own.
        const double hidden =
            100.0 * static_cast<double>(row.ordered.cycles - row.concurrent.cycles) /
            static_cast<double>(
                std::min(row.alone_first.cycles, row.alone_second.cycles));
        std::printf("  %9s %9s %10s %10s %12s %12s %8.2f%%\n", name_of(a), name_of(b),
                    with_commas(row.alone_first.cycles).c_str(),
                    with_commas(row.alone_second.cycles).c_str(),
                    with_commas(row.ordered.cycles).c_str(),
                    with_commas(row.concurrent.cycles).c_str(), hidden);
    }

    // --- a grid the host never learns ----------------------------------------

    // A culling pass writes how much survived, and the launch after it runs at
    // that size. The host issues both and is told neither number.
    constexpr uint32_t CANDIDATES = 8;

    std::printf(
        "\n  A cull decides the next launch's grid, out of %u candidates. host is "
        "the same grid launched the ordinary way\n\n",
        CANDIDATES);
    std::printf("  %9s %8s %14s %12s %14s %12s\n", "visible", "grid", "indirect work",
                "cycles", "host work", "cycles");

    std::vector<CullRow> culls;
    for (const uint32_t visible : {8u, 4u, 2u, 0u}) {
        // Device-decided: the cull runs first in the same stream, and the grid is
        // read once it has retired.
        MyGPURuntime rt = make_machine(SMS, 1);
        void* scratch = rt.myrt_malloc(SCRATCH_BYTES);
        void* flags = rt.myrt_malloc(CANDIDATES * sizeof(float));
        void* grid = rt.myrt_malloc(3 * sizeof(float));
        const size_t base = rt.myrt_device_offset(scratch);

        std::vector<float> visibility(CANDIDATES, 0.0f);
        for (uint32_t i = 0; i < visible; ++i) {
            visibility[i] = 1.0f;
        }
        rt.myrt_memcpy(flags, visibility.data(), visibility.size() * sizeof(float),
                       Direction::HostToDevice);

        const Program cull = cull_kernel(rt.myrt_device_offset(flags), CANDIDATES,
                                         rt.myrt_device_offset(grid));
        rt.myrt_launch_async([cull](void**) { return cull; },
                             LaunchConfig{dim3{1, 1, 1}, dim3{1, 1, 1}}, nullptr);

        const Program work = compute_kernel(base);
        IndirectLaunchConfig indirect;
        indirect.grid_offset = rt.myrt_device_offset(grid);
        indirect.block = dim3{WARP_SIZE, 1, 1};
        rt.myrt_launch_indirect([work](void**) { return work; }, indirect, nullptr);
        rt.myrt_wait();
        const Reading device_side{rt.stats().weighted_lane_ops, rt.stats().cycles,
                                  rt.stats().stall_steps};

        // Host-decided: the same grid, told to the machine instead of computed by
        // it. The difference between the two is the cull, and nothing else.
        const Reading host_side =
            visible == 0 ? Reading{} : run({Job{Kind::Compute, visible}}, false, SMS, 1);

        culls.push_back(CullRow{visible, visible, device_side, host_side});
        std::printf("  %9u %8u %14s %12s %14s %12s\n", visible, visible,
                    with_commas(device_side.weighted).c_str(),
                    with_commas(device_side.cycles).c_str(),
                    with_commas(host_side.weighted).c_str(),
                    with_commas(host_side.cycles).c_str());
    }

    // --- files ---------------------------------------------------------------

    std::ofstream csv(prefix + ".csv");
    if (!csv) {
        std::fprintf(stderr, "stream_bench: cannot write %s.csv\n", prefix.c_str());
        return 1;
    }
    csv << "table,a,b,launches_or_alone_a,blocks_or_alone_b,ordered_cycles,concurrent_"
           "cycles,ordered_stalls,"
           "concurrent_stalls,ordered_work,concurrent_work\n";
    for (const FillRow& r : fill) {
        csv << "fill,compute,compute," << r.launches << ',' << r.blocks << ','
            << r.ordered.cycles << ',' << r.concurrent.cycles << ',' << r.ordered.stalls
            << ',' << r.concurrent.stalls << ',' << r.ordered.weighted << ','
            << r.concurrent.weighted << '\n';
    }
    for (const PairRow& r : pairs) {
        csv << "pair," << name_of(r.first) << ',' << name_of(r.second) << ','
            << r.alone_first.cycles << ',' << r.alone_second.cycles << ','
            << r.ordered.cycles << ',' << r.concurrent.cycles << ',' << r.ordered.stalls
            << ',' << r.concurrent.stalls << ',' << r.ordered.weighted << ','
            << r.concurrent.weighted << '\n';
    }
    for (const CullRow& r : culls) {
        csv << "cull,device,host," << r.visible << ',' << r.grid << ','
            << r.host_side.cycles << ',' << r.indirect.cycles << ",0,0,"
            << r.host_side.weighted << ',' << r.indirect.weighted << '\n';
    }

    std::ofstream md(prefix + ".md");
    char buf[256];
    md << "<!-- generated by benchmarks/stream_bench; do not edit by hand -->\n\n"
       << "## Filling a machine one grid cannot\n\n"
       << TOTAL_BLOCKS << " blocks of compute on " << SMS
       << " SMs holding one block each, cut into launches.\nThe work is the same in "
          "every row; only the number of queues changes.\n\n"
       << "| Launches | blocks each | one stream | n streams | speedup |\n"
       << "|---:|---:|---:|---:|---:|\n";
    for (const FillRow& r : fill) {
        std::snprintf(buf, sizeof(buf), "| %u | %u | %s | %s | **%.2fx** |\n", r.launches,
                      r.blocks, with_commas(r.ordered.cycles).c_str(),
                      with_commas(r.concurrent.cycles).c_str(),
                      static_cast<double>(r.ordered.cycles) /
                          static_cast<double>(r.concurrent.cycles));
        md << buf;
    }

    md << "\n## Covering one kernel's waiting with another\n\n"
       << "Two launches on one SM holding two blocks, in cycles. `memory` is " << LOADS
       << " loads of a line each,\n`compute` " << FMAS
       << " fused multiply-adds. *hidden* is what concurrency took off the "
          "shorter of\nthe two, which is the most it could take.\n\n"
       << "| First | second | 1st alone | 2nd alone | one stream | two streams | "
          "hidden |\n"
       << "|---|---|---:|---:|---:|---:|---:|\n";
    for (const PairRow& r : pairs) {
        std::snprintf(buf, sizeof(buf), "| %s | %s | %s | %s | %s | %s | **%.2f%%** |\n",
                      name_of(r.first), name_of(r.second),
                      with_commas(r.alone_first.cycles).c_str(),
                      with_commas(r.alone_second.cycles).c_str(),
                      with_commas(r.ordered.cycles).c_str(),
                      with_commas(r.concurrent.cycles).c_str(),
                      100.0 *
                          static_cast<double>(r.ordered.cycles - r.concurrent.cycles) /
                          static_cast<double>(
                              std::min(r.alone_first.cycles, r.alone_second.cycles)));
        md << buf;
    }

    md << "\n## A grid the host never learns\n\n"
       << "A cull pass adds up " << CANDIDATES
       << " visibility flags and writes the total where the\nlaunch after it reads "
          "its grid. `host` is that same grid launched the ordinary\nway, for the "
          "cost of asking.\n\n"
       << "| Visible | grid | indirect work | cycles | host work | cycles |\n"
       << "|---:|---:|---:|---:|---:|---:|\n";
    for (const CullRow& r : culls) {
        std::snprintf(buf, sizeof(buf), "| %u | %u | %s | %s | %s | %s |\n", r.visible,
                      r.grid, with_commas(r.indirect.weighted).c_str(),
                      with_commas(r.indirect.cycles).c_str(),
                      with_commas(r.host_side.weighted).c_str(),
                      with_commas(r.host_side.cycles).c_str());
        md << buf;
    }

    std::printf("\nwrote %s.md and %s.csv\n", prefix.c_str(), prefix.c_str());
    return 0;
}
