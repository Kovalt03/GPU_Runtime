#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "thread.hpp"

// The machine, in one place.
//
// Everything a figure in benchmarks/ depends on is either a constant here or a
// field of GPUSpec. Before this file the numbers were spread across three
// headers and a .cpp, and answering "what machine was this measured on" meant
// reading all four.
//
// Three kinds of number live here, and the difference matters:
//
//   fixed      the data structures are built from it — changing it means
//              changing a std::array, so it is a constant and not a knob
//   default    a knob's starting value, chosen to describe real hardware
//   model      a price rather than a size. Chosen ratios, not measurements:
//              there is no hardware here to time
//
// What is *not* here: instruction_cost and instruction_latency, which live in
// isa.cpp beside the opcodes they price. Moving them would put an opcode's cost
// two files from its definition, and the switch that lists them is what the
// compiler checks for completeness when an opcode is added.

// --- fixed by the data structures --------------------------------------------
// thread.hpp owns these because Thread and Warp are sized by them:
//
//   WARP_SIZE            32   lanes a warp
//   REGS_PER_THREAD     256   floats a thread, and running out throws
//   SHARED_MEM_FLOATS  4096   a block's scratchpad
//   WARP_MASK_REGISTERS   4   masks the warp-level primitives write

// --- the memory hierarchy ----------------------------------------------------

// Bytes fetched together. 128 is what NVIDIA moves for a warp-wide access, and
// at four bytes a float that is a warp's 32 lanes exactly.
inline constexpr uint32_t CACHE_LINE_BYTES = 128;

// L1 is exact: 128 KB per SM from Volta onwards. L2 is device-wide and has grown
// by generation — 6 MB on V100, 40 MB on A100 — and 8 MB is the low end of that.
// Nothing here fills L2, so what it demonstrably does is catch what L1 drops.
inline constexpr size_t L1_LINES = 1024;
inline constexpr size_t L2_LINES = 65536;

// --- what a line costs, and how long it takes --------------------------------
// Two independent scales. Cost is issue capacity, which is this project's own
// measure; latency is cycles, following hardware in putting L1 in the tens, L2
// in the low hundreds and a trip to memory beyond. Both are chosen ratios with
// the provenance instruction_cost has.

inline constexpr uint32_t L1_HIT_COST = 8;
inline constexpr uint32_t L2_HIT_COST = 30;

// What lanes colliding on one address cost each other.
//
// An atomic executes where the caches meet, not in a lane, and two lanes naming
// the same address cannot be served at once — the unit does them one after
// another. So a warp-wide atomic to a single counter is 32 operations deep, and
// to 32 separate counters is one deep.
//
// The step is the cost of one more collision, and it is a chosen ratio like the
// prices around it. Set at an L2 hit, which is where the operation happens.
inline constexpr uint32_t ATOMIC_CONFLICT_LATENCY = 30;

inline constexpr uint32_t L1_HIT_LATENCY = 30;
inline constexpr uint32_t L2_HIT_LATENCY = 200;
inline constexpr uint32_t MEMORY_LATENCY = 400;

// --- how many SMs, and how much fits on one ----------------------------------

// The whole of what a launch's blocks compete for.
//
// The defaults describe the machine every figure in benchmarks/ was taken on —
// one SM holding one block, which is the nested loop myrt_launch used to be — so
// turning any of it up is a deliberate act and nothing moves until someone
// performs it.
struct SMConfig {
    uint32_t sm_count = 1;

    // Residency is the smallest of what the three allow. Hardware states it the
    // same way, and the reason an occupancy calculator exists is that which
    // limit binds is rarely obvious.
    uint32_t blocks_per_sm = 1;
    uint32_t warp_slots_per_sm = 64;  // Volta onwards
    size_t shared_bytes_per_sm = SHARED_MEM_FLOATS * sizeof(float);
};

// A machine, and something to print at the head of a table.
//
// A benchmark that does not say which machine it ran on produces figures nobody
// can reproduce, and this project has been caught by that once already — the
// scenes were in a session rather than in the repository.
struct GPUSpec {
    SMConfig sms;

    // Both caches, in lines. Shrunk by benchmarks that mean to reach a capacity:
    // filling L2 for real would take about 175,000 triangles.
    size_t l1_lines = L1_LINES;
    size_t l2_lines = L2_LINES;

    // Multi-line, one field a row, ending in a newline. Written for a benchmark
    // header rather than for a log line, and it names the fixed constants beside
    // the configured ones so that a reader can see which is which.
    std::string describe() const;

    // The knobs alone, as `name = value` lines that parse_spec reads back. Kept
    // apart from describe() because a machine file may only hold what a machine
    // file can change: WARP_SIZE and the register count size a std::array, and a
    // file that appeared to set them would be lying.
    std::string to_text() const;

    // How many blocks fit on one SM, for a kernel with this many warps a block
    // and this much declared shared memory. The arithmetic an occupancy table is
    // made of, so that a benchmark need not rediscover it.
    uint32_t residency(uint32_t warps_per_block, size_t shared_bytes) const;
};

// A machine from `name = value` lines. Blank lines and `#` comments are skipped,
// spaces around the `=` are not required, and the order of the fields does not
// matter.
//
// An unknown name is an error rather than a warning, and so is one of the fixed
// constants: a run configured from a file nobody checked is worth less than a run
// that refused to start. The message names the field.
//
// Throws std::runtime_error on a name it does not know, a value it cannot read,
// or a line with no `=` in it.
GPUSpec parse_spec(const std::string& text);

// The same, from a file. Throws if it cannot be opened.
GPUSpec load_spec(const std::string& path);
