#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gpu_spec.hpp"

// Shared by the executables, not by the library: where a run puts its files and
// how long the run took. Nothing here touches the simulated device.

struct Args {
    // Everything that was not a flag, in order — the width, height and counts
    // the apps already read positionally.
    std::vector<std::string> positional;

    // --out <dir>, defaulting to test/benchmark/output. Created if missing, and
    // returned with a trailing slash so a filename concatenates onto it.
    std::string out_dir;

    // --name value, for the programs that sweep and so cannot spend their
    // positional slots on a resolution.
    std::map<std::string, std::string> flags;

    // out_dir/images/, created alongside it. A run writes twenty PPMs and a
    // handful of tables, and mixing them makes the tables hard to find.
    std::string images_dir() const;

    uint32_t number(size_t index, uint32_t fallback) const;
    std::string text(size_t index, const std::string& fallback) const;
    uint32_t flag(const std::string& name, uint32_t fallback) const;
    std::string text_flag(const std::string& name, const std::string& fallback) const;
};

Args parse_args(int argc, char** argv);

// The machine a run was asked for: --machine <path>, or the defaults when no
// path was given. Every benchmark takes it, and every benchmark prints what it
// got — a table whose machine is not stated beside it cannot be reproduced.
//
// Throws std::runtime_error if the file names a field that does not exist or one
// that is fixed rather than configured.
GPUSpec machine_from(const Args& args);

// A directory named for the moment it was made:
// test/benchmark/output/YYYY-MM-DD_HHMMSS.
//
// scripts/bench.sh makes one and hands it to every app, so a run's images and
// numbers land together and previous runs stay where they are. An app given no
// --out writes straight into test/benchmark/output/, which is where a bare
// ./build/apps/... has always put its files.
std::string timestamped_run_dir();

// Wall clock, and the one figure in any of these programs that does not
// reproduce: it measures this simulator on this host. The lane and warp counts
// beside it are what to compare across machines.
class Stopwatch {
public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}

    void restart()
    {
        start_ = std::chrono::steady_clock::now();
    }

    double seconds() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_)
            .count();
    }

private:
    // steady_clock rather than high_resolution_clock, which is permitted to run
    // backwards and would then report a negative interval.
    std::chrono::steady_clock::time_point start_;
};
