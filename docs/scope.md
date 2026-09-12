# What this models, and what it does not

A real GPU runs graphics through a chain of fixed-function blocks with two
programmable stages wired into it. This simulator has no fixed-function blocks
at all: rasterisation is a kernel like any other, written against the same 45
opcodes. That makes it a **compute-based software renderer on a simulated SIMT
machine** — the family Larrabee, CUDA rasterisation research and Nanite's
software path belong to — rather than a model of a graphics ASIC.

That is a deliberate position, not an omission. A simulated fixed-function
rasteriser would have nothing to say about warp divergence, which is the thing
this project exists to measure.

## The graphics pipeline

| Hardware block | Kind | Here |
|---|---|---|
| Input Assembler | fixed | **partly** — an index buffer is fetched and expanded, but by the raster kernel itself rather than a block ahead of it |
| Post-transform vertex cache | fixed | **absent** — pass 1 transforms each unique vertex once, which is the saving the cache exists to make, but nothing reuses a result *within* a launch. `simulated_cache_misses` scores meshes against the cache the hardware would have |
| Vertex shader | programmable | `build_vertex_program`, one thread per vertex, and a caller's `VertexFn` runs inside it |
| Primitive assembly, clip, cull | fixed | back-face culling inside the kernel, near-plane clipping as a pass of its own, and frustum culling per instance — the last of those decides the next launch's grid, so the host never learns how much survived |
| Rasteriser (edge functions, quad generation) | fixed | `build_raster_program`, one thread per pixel |
| Hi-Z / early-Z | fixed | **partly** — `DepthUse::EarlyZ` and a depth prepass are built and measured; the hierarchical form is not |
| Fragment shader | programmable | folded into the raster kernel |
| ROP (blend, depth write) | fixed | **absent** — the running nearest lives in a register |
| Texture units, samplers | fixed | **absent** |

## Ray tracing

| Hardware | Here |
|---|---|
| BVH / acceleration structure | **built** — SAH or a median split, leaf size a parameter, and a stack in shared memory because an instruction names its registers immediately |
| Traversal and intersection units | **absent** — traversal is instructions in the same kernel, the same choice made for rasterisation. So the divergence a tree causes is charged rather than hidden, which is where the interesting number is |
| Ray reordering hardware | **absent as a unit**, present as an instruction: `REORDER` regroups a block's threads by material, worth 30.6% on a heavy shader and negative on a light one. What it cannot touch is traversal divergence, which is upstream of the rendezvous and the larger half |
| Shader binding table | **absent** — the arms are a `BRA_DIV` chain, so how many there are is fixed when the kernel is built. A real SBT picks a shader address at run time, which needs an indirect branch this ISA does not have |
| Index buffer as BLAS input | **absent** — and note it is real hardware's too: DXR and Vulkan RT both name one. What differs is when it is read, the builder consuming it once where the raster side reads it every draw |
| Instance transforms (TLAS) | **built** — a tree over the placements, and the ray moved into an instance's space at its leaves. 12.5x less memory at 256 copies and 0.59x the work, an instance visited costing sixteen scalar loads for its matrix |
| Several bottom-level structures | **built** — an instance carries where its own tree and triangles begin, so a scene holds different meshes rather than copies of one |
| Instance ID / material in the hit | **built** — a number the runtime carries and never interprets, reaching a fragment shader. What a warp diverging on it splits along is the scene rather than a key invented to split it |
| Shadow rays | **built** — a hit asks what stands between it and the light, as a second traversal of the same loop rather than a second kernel. Costs 66% on the mirror room at 256 square: the traversals double and the per-turn arithmetic does not |
| Any-hit traversal | **absent** — the shadow ray finds the nearest blocker where any would do, because it reuses the loop that was already written. Hardware keeps a separate path that stops at the first |
| Refraction | **built** — Snell's law with the ratio picked by which side of the surface the ray is on, and total internal reflection as a clamped discriminant. What is absent is *splitting*: one ray leaves a surface, so a glass surface is either reflective or transmissive and never both. Splitting is what recursion buys and a loop keeping no stack does not have |
| Sampling | **absent** — one primary ray a pixel, one continuation a surface, one shadow ray a bounce, no randomness. This is a Whitted tracer, so shadows are hard-edged, reflections are mirror-sharp and there is no indirect diffuse. A Monte Carlo path tracer needs a random number, and the ISA has no integer or bitwise op to hash one from — a host-filled sample buffer read through `V_LD_GLOBAL_F32` is what would fit |
| Wide nodes | **absent** — two children a node. Hardware fetches four or eight bounds at once to spend one cache line rather than two |

The tree removes 16.17x of the lane work on 4,096 triangles and the warp keeps
2.89x of it: two adjacent pixels are two rays that leave the root for different
children, and from there the lanes are at different pcs. Divergence goes from
16.5% to 85.1%. That gap is the measurement worth having — it is the argument
ray-reordering hardware exists to make, arrived at rather than assumed.

## Memory

| | Here |
|---|---|
| L1 / L2 hierarchy | modelled: 1024 lines and 65,536 of 128 bytes, LRU, tags only. Nothing here outgrows L2, so what it demonstrably does is catch what L1 drops |
| Coalescing | modelled: a warp's 32 addresses are charged by the distinct lines they touch. Transaction counts, not bandwidth contention |
| DRAM, bandwidth saturation | **partly modelled** — an optional fixed-rate queue models memory saturation with Coalesced/Cached memory and modelled latency. DRAM row buffers and detailed read/write traffic are absent |
| Shared memory + barrier | modelled: 4096 floats a block, `BARRIER`, a load costing 8 |
| Constant window | modelled: `V_LD_CONST_F32` and a matrix form, normally addressed from the launch-seeded constant base; the executor rejects effective addresses that differ across active lanes — charged once for the warp rather than once a lane, which is the only such opcode here |
| Register file | modelled: 256 a thread, 248 of them assignable (r0–r247; r248–r255 are launch state), which is close to the 255 a real thread gets. What differs is what happens at the ceiling — hardware spills to local memory or holds fewer warps resident, and this throws. So registers are a wall here and a performance axis there; occupancy is limited by blocks, warp slots and shared memory, never by registers |

## Scheduling

| | Here |
|---|---|
| Warps of 32, `activeMask`, reconvergence | modelled — the centre of the project |
| Independent thread scheduling | modelled — `WarpPolicy::Independent`, per-thread pc with no lane starving another |
| Warp-level primitives | modelled — ballot, any, all, syncwarp, shuffle, each naming its participants |
| Multiple SMs, occupancy | modelled — SMs hold several blocks at once, and residency is the smallest of the block, warp-slot and shared-memory limits. One SM holding one block is the default, which is what every figure was taken on |
| Latency hiding | modelled — a result arrives some cycles after it is issued, and both the warps of a block and the blocks of an SM cover the wait |
| Instruction-level parallelism | **absent** — issue is in-order with no scoreboard, so a warp waits out every instruction whether or not the next one wanted the result. A dependent chain and independent accesses cost the same; occupancy is what covers a wait here, never the warp's own next instruction |
| Persistent buffers | modelled — `upload` / `release` hold geometry between draws, and a repeated draw of what fits refetches nothing. Capacity and compulsory misses are told apart by whether a second draw pays again |
| Early-Z / depth prepass | modelled — `draw_early_z` runs a depth pass and then shades only the triangle it names. It loses here, by a ratio the shade and the coverage test decide |
| Fragment shading | flat lighting: a face normal a triangle, one normalize a pixel, and a light that is a point on the inline path and a direction on the one that bounces. No textures |
| Streams, concurrent kernels | modelled — launches queue, one stream runs in order and separate streams overlap. Work partitions between them exactly; cycles are charged to every stream that was resident, so they add to more than the wall clock and the surplus is the overlap |
| Indirect launch | modelled — `myrt_launch_indirect` reads its grid from device memory when the launch reaches the machine, so the kernel before it in the stream decides its size. Reading it costs no lane-op at all |
| Asynchronous copy (`cp.async`) | modelled — `V_CP_ASYNC_SHARED_GLOBAL_F32` moves global memory into shared without a register and without the warp waiting, and the warp meets it at `S_CP_ASYNC_WAIT`. Reading bytes still in flight is refused rather than answered. Worth 87% of a fill on its own and 2% of a renderer that stages once and reads 256 times |
| Bandwidth | modelled, and off by default — `BandwidthModel::Modelled` gives memory a rate in lines a cycle and queues what arrives while it is busy. The kernels here ask for 0.02 to 0.60 lines a cycle against a ceiling of 8, so every scaling figure in this repository stands; what is still absent is read against write traffic, row buffers, and any ceiling on the on-chip paths |
| Thread block clusters | modelled — `LaunchConfig::cluster_size` places blocks together, `V_LD_CLUSTER_F32` reads a neighbour's shared memory and `BARRIER_CLUSTER` is the rendezvous across them. A producer-consumer pair that needed two launches through global memory becomes one launch: -63% of the cycles at four blocks, for twice the issued work |
| Shader execution reordering | modelled — `REORDER` regroups a block's threads by a key so that lanes about to do the same thing share a warp, registers and pc travelling with the thread. Worth -75% of the warp steps on a scattered key and +13% on one that was already coherent; the regroup is within a block, where hardware's pool is an SM |
| Matrix unit | modelled — `V_MMA_16X16X16_F32` has the warp compute a 16x16x16 product in one instruction against 128 fused multiply-adds a lane, with the fragments spread eight elements to a lane. The cost is a claim about a unit rather than a measurement; what is counted is that it would have to cost 128 before the two routes came level |
| Atomics | modelled — `V_ATOM_ADD_GLOBAL_F32` reads, adds and writes back indivisibly, and hands each lane what was there before, which is how a compaction pass gives every surviving item a slot. Lanes naming one address are charged for serialising, so coalescing cannot help them and a warp reduction before the atomic is worth 3.3x |

## The three that matter most

**Bandwidth.** With the default settings, requests do not queue. Enabling
`BandwidthModel::Modelled` together with `LatencyModel::Modelled` and
`MemoryModel::Coalesced` or `Cached` adds a fixed-rate memory queue. Cache hits
do not consume its capacity. This is a saturation model, not a detailed DRAM
controller: row buffers, on-chip bandwidth limits and complete read/write
traffic modelling remain absent.

**A shader worth skipping.** Early-Z is built and loses by 30% on the scene it
exists for, because what decides that trade is the shade against the coverage test
that finds it — and flat lighting costs about half a coverage test here. Textures
are what would carry it past the crossover, and the raster fragment stage has
none. The shadow rays built for the ray tracer are no help: they belong to a walk
that bounces, which the raster routes do not.

**Instruction-level parallelism.** Issue is in order with no scoreboard, so a warp
waits out every instruction whether or not the next one wanted the result. A
dependent chain and independent accesses cost the same, and occupancy is the only
thing that covers a wait. That is what makes the fragment load worth 42% — eight
loads are eight waits here and would overlap on hardware — so it is the assumption
the sharpest figure in this repository rests on.

---

## Where it goes next

Named because each is a wall the project has actually hit, rather than a wish.

**Spilling to local memory.** The register file is the binding constraint —
shadow rays hit the ceiling twice, and a two-level traversal cannot bounce at
all because the two do not fit together. Hardware answers this by spilling to
local memory, which holds what a CPU stack frame holds and is laid out
interleaved across a warp so that thirty-two lanes spilling the same variable
make one transaction rather than thirty-two. A spill needs no dynamic register
indexing — the slot is known when the kernel is built — so it fits the existing
`V_ST_GLOBAL_F32` and `V_LD_GLOBAL_F32`, and `LOCAL` was reserved as a memory
space before there was a reason to fill it.

**Registers as an occupancy limit.** Residency here is the smallest of the
block, warp-slot and shared-memory limits, and hardware's fourth is registers.
Adding it turns the wall above into the trade it is on real hardware.

**A wavefront split.** The tracer is a megakernel and fails the way the
literature says a megakernel fails. Splitting raygen, traversal and shading into
separate launches communicating through global queues is the documented answer,
and streams and indirect launch are already here to express it.

**Sampling.** A path tracer at image resolution is out of reach on a CPU
simulator, but the interesting number is not an image — it is what a warp loses
when its lanes scatter. Sixty-four pixels square with sixteen samples is about
two hundred thousand traversals, which is seconds. That would also give
`REORDER` a workload worth reordering: it has been measured on divergence a
scene happened to contain, never on divergence that is the point.

---

[← back to the README](../README.md)


## Validation boundaries

- Constant loads require equal effective addresses across the active lanes of
  an issued instruction. The executor checks this before reading or charging
  the broadcast. An arbitrary register is legal when its address is uniform.
- Async-copy read checks cover scalar and wide shared loads and cluster loads,
  checking every producer warp in the source block. A wait releases queue slots,
  but reads from another warp remain invalid until its completion time. Use a
  block or cluster barrier after producer waits to coordinate consumers.
- Copy data is stored immediately; completion is represented by metadata and
  read guards. Destination ranges are conservative bounding intervals, so holes
  between scattered destinations can also be rejected. This is not a full race
  detector or a hardware memory-consistency model.
- The IR builder reserves r248–r255. Raw ISA programs remain responsible for
  preserving launch state when writing registers directly.
