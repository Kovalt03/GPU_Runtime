# GPU Runtime Simulator

A software GPU runtime simulator written in C++17. It implements a SIMT
execution model, a custom Virtual ISA, and a Runtime API from scratch, using a
Ray-Triangle intersection kernel as the validation workload.

> **Portfolio goal**: to demonstrate a developer who understands the rendering
> pipeline from top to bottom. Built for system semiconductor SW roles.

---

## Architecture

```
  ┌──────────────────────────────────────────────────┐
  │       Ray-Triangle Kernel  (validation load)     │
  │             kernels/ray_triangle.cpp             │
  │       Möller–Trumbore · 256×256 · PPM output     │
  └────────────────────┬─────────────────────────────┘
                       │  myrt_launch(kernel, grid, block, args)
  ┌────────────────────▼─────────────────────────────┐
  │                 Runtime API                      │
  │              include/runtime.hpp                 │
  │   malloc / free / memcpy / launch / sync / stats │
  └──────┬───────────────────────────────────────────┘
         │  delegates warp execution
  ┌──────▼───────────────────────────────────────────┐
  │              Warp Scheduler                      │
  │            include/scheduler.hpp                 │
  │   Round-robin · divergence statistics counters   │
  └──────┬───────────────────────────────────────────┘
         │  executes instructions per warp
  ┌──────▼───────────────────────────────────────────┐
  │          Thread / Warp Structure                 │
  │             include/thread.hpp                   │
  │   Thread[256 regs, pc, active]                   │
  │   Warp[32 threads, activeMask]                   │
  │   Block[warps, 4096B sharedMem]                  │
  └──────┬───────────────────────────────────────────┘
         │  instruction fetch & decode
  ┌──────▼───────────────────────────────────────────┐
  │               Virtual ISA                        │
  │              include/isa.hpp                     │
  │   Opcode · Instruction · Program                 │
  │   V_MUL_F32 / V_DOT_VEC3_F32 / V_LD_GLOBAL_F32   │
  │   V_CMP_F32 / BRA_DIV / RET                      │
  └──────┬───────────────────────────────────────────┘
         │  physical memory access
  ┌──────▼───────────────────────────────────────────┐
  │               Memory Model                       │
  │              include/memory.hpp                  │
  │   Host / Device (separate address spaces)        │
  │   free-list allocation                           │
  └──────────────────────────────────────────────────┘
```

---

## Instruction naming

Opcodes follow a fixed scheme so that future extensions fill in a slot rather
than force a rename:

```
ALU        V_<OP>[_<SHAPE>]_<TYPE>          V_ADD_F32, V_CROSS_VEC3_F32
Memory     V_<LD|ST>_<SPACE>[_<SHAPE>]_<TYPE>   V_LD_GLOBAL_F32
Control    <OP>                             BRA, BRA_DIV, RET
```

- **`V_`** marks instructions that write the per-lane register file, following
  the AMD `v_add_f32` convention. Control flow only mutates warp state
  (`pc`, `activeMask`), so it carries no prefix.
- **`<SHAPE>`** is omitted for scalars; `VEC3` today, with `VEC4`, `MAT3` and
  `MAT4` reserved for matrix transforms.
- **`<TYPE>`** always comes last, leaving room for `F64` / `F16`.

`F32` rather than `FP32` because that is the mnemonic standard — PTX `add.f32`,
AMD `v_add_f32`, WGSL / SPIR-V / Rust `f32`. The rule is machine-checked by
`OpcodeNamesFollowScheme` in `tests/test_isa.cpp`, so it cannot drift as
opcodes are added.

---

## Implementation stages

| # | Layer | Header / Source | Status |
|---|-------|-----------------|--------|
| 1 | Virtual ISA | `include/isa.hpp` · `src/isa.cpp` | ✅ 17 tests passing |
| 2 | Memory Model | `include/memory.hpp` · `src/memory.cpp` | 🔲 |
| 3 | Thread / Warp | `include/thread.hpp` · `src/thread.cpp` | 🔲 |
| 4 | Warp Scheduler | `include/scheduler.hpp` · `src/scheduler.cpp` | 🔲 |
| 5 | Runtime API | `include/runtime.hpp` · `src/runtime.cpp` | 🔲 |
| 6 | Ray-Triangle Kernel | `kernels/ray_triangle.cpp` | 🔲 |
| 7 | Divergence Benchmark | `benchmarks/divergence_bench.cpp` | 🔲 |

---

## Build and run

```bash
# Build (GoogleTest is fetched automatically on first configure)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Full test suite
cd build && ctest --output-on-failure

# A single layer
./build/tests/test_isa
./build/tests/test_isa --gtest_filter=Isa.Vcmp*

# Ray-Triangle rendering (PPM output)
./build/kernels/ray_triangle
convert output/result.ppm output/result.png   # requires ImageMagick

# Warp divergence benchmark
./build/benchmarks/divergence_bench
```

### Formatting

Layout is decided by `.clang-format`, never by hand:

```bash
./scripts/format.sh          # rewrite in place
./scripts/format.sh --check  # list unformatted files, exit 1 (for CI)
```

`cmake --build build --target format` / `format-check` do the same. The config
is derived from the style already in the tree: 4-space indent, 90-column limit,
function braces on their own line, `int* ptr`.

---

## Tech stack

- **Language**: C++17 (`-std=c++17`, no compiler extensions)
- **Build**: CMake 3.16+
- **Tests**: GoogleTest (pinned to v1.17.0 via `FetchContent`)
- **Output format**: PPM (Portable Pixmap)

---

## Key metrics (to be filled in once implemented)

| Metric | Value |
|--------|-------|
| Warp divergence rate (ray kernel) | — |
| Throughput difference before/after masking | — |
| Rendering throughput | — GIOPS |
| Test coverage | — |

---

## Layout

```
gpu-runtime-sim/
├── .clang-format
├── CMakeLists.txt
├── scripts/
│   └── format.sh
├── include/
│   ├── isa.hpp
│   ├── memory.hpp
│   ├── thread.hpp
│   ├── scheduler.hpp
│   └── runtime.hpp
├── src/
│   ├── isa.cpp
│   ├── memory.cpp
│   ├── thread.cpp
│   ├── scheduler.cpp
│   └── runtime.cpp
├── kernels/
│   └── ray_triangle.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_isa.cpp
│   ├── test_memory.cpp
│   ├── test_scheduler.cpp
│   └── test_runtime.cpp
├── benchmarks/
│   └── divergence_bench.cpp
└── output/
    └── .gitkeep
```
