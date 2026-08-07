# GPU Runtime Simulator

A software GPU runtime simulator written in C++17. It implements a SIMT
execution model, a custom Virtual ISA, and a Runtime API from scratch, using a
Ray-Triangle intersection kernel as the validation workload.

> **Portfolio goal**: to demonstrate a developer who understands the rendering
> pipeline from top to bottom. Built for system semiconductor SW roles.

<img src="docs/triangle.png" width="256" alt="Ray-traced triangle shaded by barycentric coordinates">

256x256, one thread per pixel, Möller-Trumbore intersection written in the
project's own instruction set and executed by its own warp scheduler. The
colours are the barycentric coordinates, so each vertex resolves to a pure
primary — an intersection error shows up as a wrong gradient rather than
disappearing into flat shading.

```
$ ./build/kernels/ray_triangle 256 256
rendering 256x256 — 65536 pixels, 2048 blocks of 32 threads
[STATS] divergence: 2.6%, throughput: 0.20 GIOPS
```

Divergence is concentrated at the triangle's edges, where a warp's 32 lanes
disagree about whether they hit. It falls as resolution rises — 10.1% at 64x64
against 2.6% here — because the interior grows with the area while the edge
grows only with the perimeter.

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
  │   ThreadBlock[warps, 4096B sharedMem]            │
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
| 1 | Virtual ISA | `include/isa.hpp` · `src/isa.cpp` | ✅ |
| 2 | Memory Model | `include/memory.hpp` · `src/memory.cpp` | ✅ |
| 3 | Thread / Warp | `include/thread.hpp` · `src/thread.cpp` | ✅ |
| 4 | Warp Scheduler | `include/scheduler.hpp` · `src/scheduler.cpp` | ✅ |
| 5 | Runtime API | `include/runtime.hpp` · `src/runtime.cpp` | 🔲 |
| 6 | Ray-Triangle Kernel | `kernels/ray_triangle.cpp` | ✅ |
| 7 | Divergence Benchmark | `benchmarks/divergence_bench.cpp` | ✅ |

---

## Build and run

GoogleTest is fetched automatically on the first configure.

```bash
scripts/build.sh             # configure if needed, then build
scripts/build.sh --clean     # discard build/ and start over
scripts/build.sh --release   # with optimisations

scripts/test.sh              # build, then run every test
scripts/test.sh Scheduler    # only tests whose name matches

scripts/format.sh            # apply .clang-format in place
scripts/format.sh --check    # list unformatted files, exit 1

scripts/check.sh             # all three gates, as CI would run them
```

`scripts/test.sh` builds before running, because `ctest` on its own executes
whatever binaries are already in `build/` — after an edit that can report on
code which was never compiled.

The equivalent CMake invocations still work (`cmake -B build`, `ctest`,
`--target format`), but the scripts also recover from a `build/` directory left
behind by a different generator, which otherwise blocks configuring while
leaving the stale binaries in place.

```bash
# Render (PPM, no image library needed)
./build/kernels/ray_triangle 256 256
sips -s format png output/result.ppm --out result.png   # macOS
convert output/result.ppm result.png                    # ImageMagick

# Divergence benchmark
./build/benchmarks/divergence_bench
./build/benchmarks/divergence_bench --csv   # output/divergence.csv
```

### Formatting

Layout is decided by `.clang-format`, never by hand. The config is derived from
the style already in the tree: 4-space indent, 90-column limit, function braces
on their own line, `int* ptr`.

---

## Tech stack

- **Language**: C++17 (`-std=c++17`, no compiler extensions)
- **Build**: CMake 3.16+
- **Tests**: GoogleTest (pinned to v1.17.0 via `FetchContent`)
- **Output format**: PPM (Portable Pixmap)

---

## Key metrics

| Metric | Value |
|--------|-------|
| Warp divergence rate (ray kernel, 256x256) | 2.6% |
| Warp divergence rate (64x64) | 10.1% |
| Issue overhead once a warp splits | 1.80x |
| Rendering throughput | 0.20 GIOPS |
| Tests | 88 |

### What divergence costs

`benchmarks/divergence_bench` holds the work constant and varies only how a
warp splits.

```
  lanes | divergence | warp steps | issue | useful ops
  ------|------------|------------|-------|------------
   0/32 |      0.00% |      25600 | 1.00x |     819200
   1/32 |     44.51% |      46080 | 1.80x |     818176
  16/32 |     45.56% |      46080 | 1.80x |     802816
  31/32 |     46.60% |      46080 | 1.80x |     787456
  32/32 |      0.00% |      24576 | 0.96x |     786432
```

One lane taking the branch costs exactly what sixteen do. Both sides get issued
separately either way, and a warp step buys 32 lane slots whether one lane fills
them or all of them. Both extremes read 0%: **divergence is not about branching,
but about disagreeing.**

The issue column counts what the scheduler had to put through, so it is
identical on any machine. Wall-clock throughput is reported too, but it measures
this simulator on this host — where a masked lane is skipped outright and so
genuinely costs less, while hardware would keep its ALU busy regardless.

---

## Layout

```
gpu-runtime-sim/
├── .clang-format
├── CMakeLists.txt
├── docs/
│   └── triangle.png
├── scripts/
│   ├── build.sh
│   ├── check.sh
│   ├── format.sh
│   └── test.sh
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
│   ├── test_thread.cpp
│   ├── test_scheduler.cpp
│   └── test_runtime.cpp
├── benchmarks/
│   ├── CMakeLists.txt
│   └── divergence_bench.cpp
└── output/
    └── .gitkeep
```
