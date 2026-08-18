# Measurements

A running record, so that each change can be stated as a number rather than as
an intention. Every figure here comes from the counters below.

**Reproducing these.** Every table below comes from a program, not from a
session:

```
cmake --build build -j8
./build/benchmarks/render_bench     # stdout, plus benchmarks/result/render_bench.{md,csv}
./build/benchmarks/reduction_bench  #         plus benchmarks/result/reduction.{md,csv}
./build/benchmarks/cache_bench      #         plus benchmarks/result/cache.{md,csv}
./build/benchmarks/model_bench      #         plus benchmarks/result/models.{md,csv}
./build/benchmarks/occupancy_bench  #         plus benchmarks/result/occupancy.{md,csv}
```

**Which machine.** Every benchmark takes `--machine machines/<name>.spec` and
writes the machine it ran on into its output directory. With no flag it is
`machines/default.spec` — one SM holding one block — which is what every table
here was taken on unless it says otherwise.

**Which cost model.** Every figure here is taken with a global access charged per
lane and no instruction waiting on another — `MemoryModel::Flat` and
`LatencyModel::Ignored`, the defaults. The sections from *Charging memory by the
line* to *two conclusions, re-put to the new rulers* measure what the other
models change, and are the only ones that do.

The scenes are code in `benchmarks/render_bench.cpp`, and the routes it measures
are the same `draw_walk` / `draw_tiled` / `draw_shared` / `draw_raytrace` the
tests call — each raster route measured with its coverage branch and again with
the branch blended away.
That is the fix for how these figures went stale once already: they had been
taken from scenes built by hand and never committed, so when 1/w joined the
screen vertex — moving a binned triangle from nine floats to twelve — nothing
could be re-run to catch it.

## What is measured, and what is not

| Counter | Meaning | Reproducible |
|---|---|---|
| `warp_steps` | instructions issued to a warp | yes |
| `active_lane_ops` | lane-instructions that actually ran | yes |
| `weighted_lane_ops` | as above, weighted by `instruction_cost` | yes |
| `divergence_rate` | wasted fraction of issued lane capacity | yes |
| `throughput_giops` | how fast the *simulator* retires work | **no** |

Percentages quoted anywhere in this file are ratios of `weighted_lane_ops`
unless stated otherwise. GIOPS depends on the host machine and on what else it
was doing, so it is printed for interest and never compared across runs.

Reproduce with:

```
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j8
./build/kernels/raster_triangle <width> <height> <triangles>
```

The scene is fixed: one triangle at z = -2, repeated at 0.01 intervals behind
itself. The nearest copy always wins, so **the rendered image is identical for
every triangle count** — any growth in the counters is the cost of the walk and
nothing else.

---

## Rasteriser, naive walk (baseline)

One thread per pixel; each pixel visits every triangle. 128x128, pass 2 only.

| Triangles | warp steps | weighted lane ops | vs 1 triangle | divergence |
|---:|---:|---:|---:|---:|
| 1 | 43,000 | 20,881,408 | — | 0.6% |
| 2 | 70,136 | 36,397,056 | +74% | 0.4% |
| 4 | 124,408 | 67,428,352 | +223% | 0.2% |
| 8 | 232,952 | 129,490,944 | +520% | 0.1% |
| 16 | 450,040 | 253,616,128 | **+1,115%** | 0.1% |

Linear in the triangle count, at about 15.5M weighted lane ops each. Roughly
2,048 of the 16,384 pixels are ever covered, so at 16 triangles the pipeline
spends twelve times the work of one triangle to produce a picture that has not
changed.

That is the O(pixels x triangles) cost of assigning one thread per pixel with no
binning. Real hardware sorts triangles into tiles first, so a pixel only ever
sees the few that reach it, and the cost tracks the number of covered fragments
instead.

### Divergence falls as the waste grows

Worth recording because it is the wrong way round. The loop body is work every
lane performs; only the coverage test inside it splits a warp. Adding triangles
therefore adds uniform work, and the diverged fraction of a larger total gets
smaller — 0.6% down to 0.1% while the actual cost rises twelvefold.

`divergence_rate` measures how much of the issued capacity was wasted, not how
much capacity should have been issued. It cannot see work that should never have
been scheduled, so it is a poor thing to optimise on its own.

---

## Rasteriser, binned into tiles

Triangles sorted into 32x8 screen tiles on the host, one ThreadBlock per tile,
each pixel walking only its own tile's list. 64x32, pass 2 only, against the
naive walk on the same scene.

| Scene | Triangles | walk | tiled | Change |
|---|---:|---:|---:|---:|
| small, spread over the frame | 4 | 10,885,760 | 2,372,224 | **-78.2%** |
| small, spread over the frame | 16 | 41,534,048 | 6,210,144 | **-85.0%** |
| small, spread over the frame | 64 | 164,126,592 | 31,756,160 | **-80.7%** |
| medium, stacked at the centre | 16 | 41,532,192 | 21,526,304 | **-48.2%** |
| full-frame, stacked | 16 | 41,550,336 | 41,980,416 | **+1.0%** |

The saving is in the tiles a triangle never reaches. Sixteen small triangles
leave two per tile instead of sixteen, and most tiles hold none at all — those
skip the walk outright. Stacked triangles that cover half the frame still empty
half the tiles, which is where the 47.6% comes from.

The last row is the case this loses. Triangles covering the whole frame land in
every tile, so nothing is removed and the two extra table loads per pixel are
pure overhead: 2 x 100 x 2,048 pixels is 409,600 of the 430,080 difference.

Binning costs O(tiles x triangles) on the host — 8 tiles against 2,048 pixels
here, and a box test rather than a full coverage test — so it is not close to
paying for itself in either direction.

### Divergence rises while the work falls

The prediction made from the naive numbers above, and it holds. Removing
triangles removes *uniform* work: every lane of a warp walked them together.
What is left is a larger proportion of the coverage test, which is the part that
splits a warp.

| Scene | Cost | divergence, walk | divergence, tiled |
|---|---:|---:|---:|
| 16 small, spread | -85.0% | 1.3% | 7.4% |
| 64 small, spread | -80.7% | 1.4% | 6.5% |

Five times the divergence rate for a sixth of the work. Anyone tuning on
`divergence_rate` alone would read this change as a regression.

---

## Rasteriser, staged through shared memory

Each block loads its tile's triangles into shared memory once, then meets at a
barrier before any pixel reads them back. A shared load costs 8 against a
global load's 100, and every pixel of a block was making the same twelve.

| Scene | Triangles | walk | tiled | shared | vs tiled | vs walk |
|---|---:|---:|---:|---:|---:|---:|
| small, spread | 4 | 10,885,760 | 2,372,224 | 1,290,160 | -45.6% | -88.1% |
| small, spread | 16 | 41,534,048 | 6,210,144 | 1,752,864 | -71.8% | **-95.8%** |
| small, spread | 64 | 164,126,592 | 31,756,160 | 4,797,440 | -84.9% | **-97.1%** |
| medium, stacked | 16 | 41,532,192 | 21,526,304 | 3,568,160 | -83.4% | -91.4% |
| full-frame, stacked | 16 | 41,550,336 | 41,980,416 | 6,021,120 | -85.7% | -85.5% |

### The two changes attack different waste

The last row is the one worth reading twice. Binning *lost* 1.0% there, having
nothing to remove: every triangle covers every tile. Staging wins 85.7% on the
same scene, because it never depended on binning — the waste it removes is 256
threads issuing the same twelve loads, which happens whether or not the list is
short.

Tiling removes triangles a pixel need not see. Staging removes re-reads of the
triangles it does. Neither subsumes the other, and a scene can be hostile to one
while the other still pays.

### The divergence prediction failed, and the reason is the useful part

Both earlier sections ended with the same finding — cost falls, `divergence_rate`
rises — and predicted it would happen again here. It did not:

| Scene | vs tiled | divergence, tiled | divergence, shared |
|---|---:|---:|---:|
| 16 small, spread | -66.9% | 2.1% | 2.0% |
| 64 small, spread | -82.9% | 1.7% | 1.7% |
| 16 medium, stacked | -81.0% | 0.1% | 0.2% |

Flat, against a two-thirds cut in cost.

`divergence_rate` is `masked_lane_slots / lane_slots`, and both are counted in
warp steps rather than weighted by `instruction_cost`. It cannot see cost at
all. Binning changed the numbers because it removed whole *iterations* and with
them their lane masks; staging changes what an instruction costs while leaving
every mask exactly as it was.

So the earlier readings were right about their own cause and wrong as a rule.
Removing uniform work raises the rate; making uniform work cheaper does not move
it. Both are invisible to a metric that counts slots.

---

## Two renderers, one image, different divergence

The comparison the project was built for. Same scene, same launch geometry — a
warp covering 32 adjacent pixels of one row — and the same frame out. Both
routes take their camera from one `DrawTarget`, so they cannot have seen it
from different places.

64x32, one launch each. `raster` is the naive walk, the ray tracer having no
binning to compare against.

| Scene | Triangles | raster ops | raster divergence | ray ops | ray divergence |
|---|---:|---:|---:|---:|---:|
| small, spread over the frame | 4 | 10,885,760 | 1.2% | 8,446,072 | **10.1%** |
| small, spread over the frame | 16 | 41,534,048 | 1.3% | 31,556,398 | **15.0%** |
| small, spread over the frame | 64 | 164,126,592 | 1.4% | 124,000,212 | **16.7%** |
| medium, stacked at the centre | 16 | 41,532,192 | 0.4% | 31,602,380 | **20.8%** |
| full-frame, stacked | 16 | 41,550,336 | 0.6% | 32,165,408 | **13.2%** |

The ray tracer issues 22-24% *less* work while diverging eight to fifty times
as much. An earlier reading of this table had the two within 3% of each other,
which was wrong in the direction that mattered: it made the trade look free.

Both figures come from the same cause, which is the paragraph below.

### Why the gap widens as the triangles shrink

The rasteriser branches once per triangle, on coverage. A warp is 32 pixels of
one row, and for a small triangle almost every warp is *entirely* outside it —
all 32 lanes compute the same three edge functions, all fail the same test, and
the warp never splits. The more triangles miss, the more converged it gets,
which is why 64 small triangles read 0.1%.

The ray tracer leaves for the next triangle from four separate tests. Lanes of
one warp miss the same triangle *for different reasons at different points* —
one fails on the determinant, its neighbour survives to fail on u, a third on
v. A unanimous miss still splits the warp.

So the early exits that make Möller-Trumbore cheap are the same thing that
makes it diverge — and the 22-24% is what they are worth. The walk computes
three edge functions and a normalisation for every triangle whatever the
answer; the ray tracer stops at the first test that fails, which is most of
them on a scene of small triangles.

Removing the early exits would flatten the rate and issue more work. The
rasteriser's version of that trade is the section below; the tracer's is still
untried, and is the one worth doing next — its divergent region is four early
exits rather than one shade, which is the shape predication needs.

---

## Branch against blend

The masking comparison the project set out to make. `V_CMP_F32` yields exactly
1.0 or 0.0, so coverage can select arithmetically instead of branching:

```
kept = take * new + (1 - take) * old
```

No lane disagrees, and the coverage test stops contributing to divergence
altogether. The frames are bit-identical to the branch's — asserted with EQ,
not NEAR, in `PredicationChangesTheCostAndNotThePixels`.

| Scene | Triangles | walk | tiled | raytrace | ray divergence |
|---|---:|---:|---:|---:|---:|
| small, spread over the frame | 4 | 1.9% | 1.1% | **4.1%** | 10.11% |
| small, spread over the frame | 16 | 2.0% | 1.6% | **4.4%** | 15.01% |
| small, spread over the frame | 64 | 2.0% | 1.9% | **4.5%** | 16.72% |
| medium, stacked at the centre | 16 | 2.0% | 1.9% | **4.3%** | 20.81% |
| full-frame, stacked | 16 | 1.9% | 1.9% | **2.5%** | 13.16% |

Each figure is the blend against the branch on that route, so positive is worse.

**The branch wins every scene and every route.** The plan predicted the
opposite, that small triangles would favour the blend, and that prediction was
written before anything was measured.

The raster blend loses about 2% because most of its loop sits outside the
branch: three edge functions, the weights and the depth all run whatever the
coverage test says, and only the shade is skipped. Its cost stops depending on
the scene — 42,350,592 for every 16-triangle scene whatever its shape — because
every lane shades every triangle. That is what predication means.

**The ray tracer was the case that looked most likely to pay, and lost by
twice as much.** Four early exits guard nearly the whole intersection, and this
kernel diverges eight to fifty times as much as the walk. Predicating it means
every lane finishes an intersection it would have abandoned at the first test —
and on a scene of small triangles, most rays leave at the first test. The one
scene where it loses least, at 2.5%, is the full-frame one, where fewest rays
leave early at all.

So the trade is not decided by how much divergence there is to remove.
**Removing divergence and going faster are different things**, and what decides
it is how much work the branch was skipping. More of the loop inside the branch
cuts both ways: more to gain by not splitting the warp, and more to pay for the
lanes that would have left.

Two details worth keeping:Two details worth keeping:

- `old + take*(new-old)` is the same select an instruction cheaper and is *not*
  bit-exact — subtracting `old` and adding it back rounds. 69,715 pixels of
  200,000 drifted, and at 64 triangles it changed the image.
- The ray tracer needs two guards the raster kernels never did, because its
  exits protect arithmetic as well as control flow: the determinant reciprocal
  divides by zero once nothing leaves early, and the running distance cannot
  start at infinity when the blend multiplies it by a zero weight. Both are in
  PredicatedRayTracerAgreesWithTheBranchItReplaces, each with the scene that
  detects it — a cube detects neither.
- The shared route does not reach zero divergence (0.02% at 128x128). Its
  cooperative fill is a second source, and the coverage flag has no bearing on
  it. An earlier test asserted a flat zero for all three routes and passed only
  because its 64x32 frame had too few tiles to split the fill.

---

## Charging memory by the line

A warp issues one load and its 32 lanes name 32 addresses. `MemoryModel::Flat`
charges every lane; `Coalesced` charges the distinct 128-byte lines they land in.

| Route | Flat | Coalesced | |
|---|---:|---:|---:|
| walk | 31,317,520 | 2,210,320 | **-92.9%** |
| tiled | 11,311,632 | 854,032 | -92.4% |
| shared | 2,354,064 | 1,383,664 | -41.2% |
| raytrace | 23,656,822 | 1,692,022 | -92.8% |

About 93% comes off three of the four, and all of it is one thing: these kernels
read their triangle **warp-uniformly**, so 32 lanes name one address and the flat
model was charging that as 32 loads. The comments had said hardware would broadcast
it from a scalar unit and that this is why the ISA reserves an `S_` prefix; here is
the figure.

### It reverses shared-memory staging

| Against tiled | Flat | Coalesced |
|---|---:|---:|
| staging | **-79.2%** | **+62.0%** |

Staging exists to stop 32 lanes issuing the same global load. That load costs one
line either way, so it was saving nothing — while still paying its shared stores,
its loads and two barriers a round.

The section above says the 96% staging wins is arithmetic on a flat 100-against-8
rather than a cache story. It is worse than that: it is not a win at all once a
line is the unit.

`shared` drops only 41% for the same reason — its share of global traffic was
already small, so there was less being overcharged.

---

## A wide load, now that transactions can be counted

`V_LD_GLOBAL_VEC3_F32` was specified and left unbuilt because a flat charge per
lane had nothing to show for it. The ray tracer reads three vertices a triangle
and every lane of a warp is on the same triangle, so its nine scalar loads became
three wide ones. The frames are identical.

| | Three scalar loads | One wide load | |
|---|---:|---:|---:|
| Flat | 31,517,216 | 31,517,216 | **0.0%** |
| Coalesced | 2,410,016 | 1,814,816 | **-24.7%** |
| Cached | 1,509,612 | 1,461,996 | -3.2% |
| Cycles, cached and latency | 376,542 | 192,222 | **-49.0%** |
| Warp steps | 33,436 | 27,292 | -18.4% |

64x32, 16 triangles. The rows are the same kernel under different cost models, so
only the last two are times rather than issue capacity.

**The flat row does not move by one lane-op**, which is the whole reason the
opcode waited: three floats cost three floats however many instructions ask for
them. Charged by the line, twelve bytes at one address are one transaction where
three loads are three.

**A cache had already absorbed most of the issue saving.** 24.7% against 3.2% —
the second and third scalar loads were finding their line in L1 at a cost of 8
rather than missing at 100. The transactions still fall, L1 hits going 9,787 to
3,835, but what a wide load saves in capacity a cache mostly saves first.

**What the cache does not absorb is the waiting.** Issue is in-order with no
scoreboard: a warp that has issued a load cannot issue again until the result
arrives, so three loads are three waits whatever each one cost. Cycles fall 49.0%
with the cache and 64.9% without it.

**The divergence rate rises, 12.5% to 15.0%, and nothing diverged.** The removed
instructions were warp-uniform, leaving the coverage tests a larger share of what
is issued. Tiling produced the same reading from the other direction: a rate is
not a cost.

The raster routes keep their scalar loads. Their screen vertex is x, y, z and
1/w, so a VEC3 load would leave the fourth behind — `VEC4` is the slot the naming
scheme reserves for that, and it is not built.

---

## What a cache is worth

`MemoryModel::Cached` asks, of each line, whether it had to be fetched. L1 holds
1024 lines and is emptied when a block starts; L2 holds 65,536 and is the device's,
outliving a launch as it does on hardware.

| Triangles | Lines | vs L1 | Coalesced | Cached | | Hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| 360 | 135 | 0.13x | 62,432,720 | 37,136,442 | **-40.5%** | 99.9% |
| 1,000 | 375 | 0.37x | 173,189,296 | 103,011,098 | -40.5% | 100.0% |
| 3,000 | 1,125 | 1.10x | 519,302,256 | 308,868,058 | -40.5% | 100.0% |
| 10,000 | 3,750 | 3.66x | 1,730,694,480 | 1,029,364,282 | -40.5% | 100.0% |

**A constant, not a curve.** The working set grows 27-fold past L1's capacity and
nothing moves, because pass 1's stores leave the whole screen buffer in L2 and pass
2 finds it there. At this scale L2 is effectively infinite: filling it would take
about 175,000 triangles.

### Binning is a locality optimisation too

Scaled to L1 32 lines and L2 512 — chosen so the scenes outgrow them, and not
hardware sizes:

| Triangles | scene / L2 | Route | Hit rate | Misses |
|---:|---:|---|---:|---:|
| 1,000 | 0.73x | walk | 100.0% | 383 |
| 3,000 | 2.20x | walk | 96.9% | **72,318** |
| 3,000 | 2.20x | tiled | 99.5% | 1,398 |
| 10,000 | 7.32x | walk | 96.9% | 240,318 |
| 10,000 | 7.32x | tiled | 99.6% | 4,242 |

A block of the naive walk reads every triangle, so its working set is the whole
scene. A block of the tiled route reads one tile's list. Shrink the cache until the
scene outgrows it and the walk starts missing while the tiled route does not.

The flat model attributed all of binning's win to dropping triangles. Part of it is
locality, and there was no way to say so before.

Total cost moves only 1.6% for all that — the misses are a hundred times more
numerous and still a minority. The knee is in the hit rate, not in the bill.

### What is still invisible

**Capacity misses do not arise from drawing more.** Twenty-four draws in
succession leave the hit rate at 99.4%: each reads only its own data, so a line
evicted by a later draw is never asked for again. Every miss is compulsory.

Drawing the *same* geometry twice does not help either, because `draw_*` allocates
per call and never frees — the second copy lands at a different address, so it is
new data as far as the cache is concerned.

That is a gap in the runtime API rather than in the cache: there is no way to
upload once and draw many times, which is what real code does and what a depth
prepass would need.

---

## Waiting, and covering the wait

`LatencyModel::Modelled` makes a result available some cycles after it is issued,
and a warp that has issued cannot issue again until then. Latencies are chosen
ratios with the provenance the costs have: arithmetic 4, a special function 16,
shared memory 30, and a global load from wherever its line was found — L1 30, L2
200, memory 400.

| Warps | Steps | Cycles | Stalls | Cycles/warp |
|---:|---:|---:|---:|---:|
| 1 | 6 | 45 | 39 | 45.0 |
| 2 | 12 | 46 | 34 | 23.0 |
| 4 | 24 | 48 | 24 | 12.0 |
| 8 | 48 | 64 | 16 | 8.0 |
| 16 | 96 | 96 | 0 | **6.0** |

The program is the one in `MoreWarpsCoverMoreOfTheWaiting`: a mov, an add, a
reciprocal, an add, a reciprocal, each waiting on the one before. An earlier
printing of this table came from a program that was never committed, which is the
failure this project has recorded once already — the numbers here are from the
chain a test holds.

A chain of dependent arithmetic, and the reason warps are batched at all: one warp
waits out every latency alone, sixteen cover it completely. This is the first thing
the simulator could not say before — with nothing ever waiting, resident warps
neither helped nor hindered each other.

**Whether occupancy is enough depends on what is being hidden.** A trip to memory
is 400 cycles against a block's 32 warps of a few each, so filling the block
spreads it rather than removing it: 32 warps still stall.

That is also where the cache shows up in time rather than in issue capacity. Every
warp reading one line, 16 warps: `Flat` 448 cycles against `Cached` 421 — the first
warp goes to memory and the rest find it in L1.

**Cycles are hidden within a block only.** Hardware puts several blocks on an SM
and lets all their warps cover each other; here a block runs to completion before
the next starts.

---

## What the flat model was hiding: ordering a mesh

Reordering a mesh's triangles for a vertex cache is worth a factor of three or four
on fixed-function hardware. Measured here before there was a cache, it was worth
0.004% — and even that was depth ordering rather than reuse, since a flat charge
per lane cannot tell a vertex just read from one that was not.

A cache can. The same sphere, shuffled and then reordered by Forsyth's heuristic —
ACMR 2.54 to 0.64 — drawn by the indexed walk, with L2 at 512 lines and latency
modelled:

| L1 lines | Vertices held | Cost | | Cycles | | L2 hits |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 43,413,362 | -0.23% | 26,486,224 | -2.9% | 71,837 → 67,229 |
| 4 | 32 | 42,348,914 | -1.33% | 18,347,984 | -19.4% | 44,765 → 18,845 |
| 8 | 64 | 42,057,458 | **-1.60%** | 16,095,824 | **-24.7%** | 36,637 → 5,597 |
| 16 | 128 | 42,044,786 | -0.85% | 15,997,904 | -14.9% | 21,473 → 5,021 |
| 32 | 256 | 42,016,736 | +0.07% | 15,781,154 | +1.5% | 2,354 → 3,746 |
| 1024 (hardware) | 8,192 | 41,934,830 | -0.00% | 15,158,954 | 0.0% | 23 → 23 |

**It is not fewer fetches.** The misses are 226 in every row, before and after: each
line is compulsory, touched once by someone whatever the order. What the reorder
moves is which level answered the rest — L2 hits fall by a factor of six, and the
two levels are 8 against 30 in issue capacity and 30 against 200 in cycles. Hence
the two columns differing by a factor of fifteen: reordering buys a little issue
capacity and a great deal of time.

**The saving has a window, and the window is the mesh.** 64 vertices held is the
peak because the heuristic is scored against a 32-entry FIFO. Below it nothing
stays resident whatever the order; above 256 the whole 182-vertex sphere fits and
there is nothing to evict, which is where the hardware size sits — a mesh would
need more than 8,192 vertices to reach it, and the walk being O(pixels x triangles)
puts that out of reach of a benchmark.

**The route has to be able to see the order.** The tiled route at the same L1 moves
-0.01% and +0.12%: `bin_triangles` de-indexes on the host, so a block streams each
triangle's three vertices and has nothing to re-read. That control is what
separates this from the depth ordering a shuffle also changes.

---

## Uploading once and drawing many

Every miss above was compulsory, and by construction: `draw_*` uploaded its
buffers, drew, and abandoned them, so each draw asked the cache about an address
nobody had touched. `upload` / `release` hold the geometry instead, and the
question a repeated draw asks is whether the cache could keep it.

Three draws of one upload, latency modelled:

| Triangles | Cache | Draw | Misses | Cost | Cycles |
|---:|---|---:|---:|---:|---:|
| 360 | hardware | 1 | 192 | 36,945,024 | 12,519,376 |
| 360 | hardware | 2 | **0** | 36,931,584 | 12,519,376 |
| 360 | hardware | 3 | **0** | 36,931,584 | 12,519,376 |
| 3,000 | scaled | 1 | 72,255 | 313,909,546 | 130,742,840 |
| 3,000 | scaled | 2 | **72,255** | 313,909,546 | 130,742,840 |
| 3,000 | scaled | 3 | **72,255** | 313,909,546 | 130,742,840 |

**The distinction is now measurable, and the answer is that both kinds exist.**
At the hardware sizes the scene fits and a second draw fetches nothing at all —
every one of those 192 misses was compulsory. Scaled down until the scene
outgrows L2, the same 72,255 misses recur on every draw: those are capacity
misses, and holding the geometry buys nothing, because what evicted it was the
rest of the same draw.

**And it is worth almost nothing here.** 192 misses out of 346,000 lookups is
0.04% of the issue cost, and the cycles do not move at all — a block of 64 warps
covers a trip to memory with somebody else's work. The same reading as the cache
section above: the knee is in the hit rate, not in the bill. What upload-once
buys at this scale is the ability to ask the question.

**An upload drops the lines it overwrites.** The caches hold tags and no data, so
a buffer released, reallocated at the same address and refilled would otherwise
read as resident — `myrt_memcpy` invalidates, and the draw after a re-upload
misses again on exactly the lines it replaced, 34 of them for the sphere's index
buffer.

---

## Two conclusions, re-put to the new rulers

Everything above the memory-model sections was measured when a global access cost
100 a lane and nothing waited. Two of those conclusions trade loads against
arithmetic, so both had reason to move once a warp's load became one transaction
and results took time to arrive. `model_bench` puts them again, on the routes and
scenes `render_bench` already uses — the `flat` column reproduces the tables
above to the lane-op.

### Branch against blend

Positive means the blend costs more. `cycles` is time under `Cached` and
`LatencyModel::Modelled`; the rest are issue capacity.

| Scene | Route | flat | coalesced | cached | cycles |
|---|---|---:|---:|---:|---:|
| small, spread | walk | +2.0% | +28.3% | +47.5% | +17.9% |
| small, spread | tiled | +1.6% | +20.2% | +32.5% | +17.7% |
| small, spread | raytrace | +4.4% | **+75.4%** | **+93.1%** | +58.2% |
| full-frame, stacked | walk | +1.9% | +27.6% | +46.1% | +17.9% |
| full-frame, stacked | tiled | +1.9% | +27.3% | +45.4% | +24.0% |
| full-frame, stacked | raytrace | +2.5% | +32.0% | +37.4% | +32.7% |

**The flat model was flattering predication by a factor of twenty.** Both variants
make the same loads, and at 100 a lane those loads were most of the bill —
whatever the blend added in arithmetic was a rounding error against them. Charged
by the line, a warp's load is one transaction, and what is left to compare is
mostly the arithmetic the blend performs for lanes the branch would have skipped.
The ray tracer, where the branch guards nearly the whole of Möller-Trumbore,
nearly doubles.

The prediction written before the measurement was that predication might *win*
once divergence grew dearer. It lost by an order of magnitude more instead. The
divergence it removes is still removed — the blend is at exactly 0.00% on every
route and scene here, against 0.58% to 15.01% for the branch — which makes this
the strongest version yet of the finding the section above states: **removing
divergence and going faster are different things.**

### Indexed against flattened, pass 2

| Mesh | Triangles | flat | coalesced | cached | cycles |
|---|---:|---:|---:|---:|---:|
| cube | 12 | +24.0% | +17.4% | **+13.0%** | +20.6% |
| sphere | 360 | +24.5% | +18.2% | **+13.5%** | +20.9% |

Predicted to get worse, being three dependent loads a triangle before a vertex
address is known, and it got better: the index buffer costs half what the flat
model said. Those loads are warp-uniform — every lane of a warp is on the same
triangle, so all 32 read the same three indices — which is exactly the case a
per-lane charge overstated thirty-twofold.

**On the dependence itself this machine cannot be asked.** Issue is in-order with
no scoreboard, so a warp waits out every instruction's latency whether or not
anything needed the result: a chain of three loads and three independent ones
cost the same cycles here, which is pinned in
`DependenceIsNotDistinguishedFromIssueOrder`. That is pessimistic rather than
generous — every load in the indexed path is charged a full wait — so 20.6% is an
upper bound on what the extra loads cost in time, and it is still inside the 24%
the flat model charged for them in issue capacity.

**Both movements are the same error seen from two sides.** A flat charge per lane
overprices loads relative to arithmetic, so it flattered the variant that trades
arithmetic for loads and penalised the variant that trades loads for arithmetic.
Neither conclusion reverses; both change size enough that the ranking they were
quoted to support has to be read against the model that produced it.

---

## A depth prepass, and the ratio that decides it

Early-Z is two launches over one upload: the first keeps the nearest depth a
pixel and colours nothing, the second colours only the triangle that depth names.
A pixel is shaded once however many cover it — and every triangle is walked
twice.

Measured on a full-frame stack submitted farthest first, which is the walk's
worst case and the prepass's best: every triangle covers every pixel and each is
nearer than the last, so the walk shades all of them.

| Scene | Depth complexity | flat | coalesced | cached | cycles |
|---|---:|---:|---:|---:|---:|
| stacked, barycentric | 4 | +97.1% | +82.0% | +77.4% | +91.9% |
| stacked, barycentric | 32 | +98.4% | +81.9% | +72.1% | +81.2% |
| stacked, lit | 4 | +36.9% | +40.9% | +35.4% | +51.2% |
| stacked, lit | 32 | **+29.9%** | +30.4% | **+20.8%** | +35.8% |

Positive means the prepass costs more. It loses everywhere, and the frames are
identical to the walk's — asserted with EQ in
`EarlyZDrawsWhatTheWalkDrawsAndCostsTwiceTheCoverage`.

**Depth complexity is not what decides it.** That is the surprise: it is the one
thing early-Z exists to exploit, and past about eight the figure stops moving.
Doubling the geometry doubles both sides.

**What decides it is the shade against the coverage test that finds it.** Write
`c` for what one triangle's coverage costs a frame and `s` for what one shade
costs; the walk pays `Nc + Ns` and the prepass pays `2Nc + s`, so as depth
complexity grows the ratio goes to

```
2 / (1 + s/c)
```

The two shading modes are one measurement of `c` and `s` apart. Barycentric at 32
gives `Nc` = 82,969,600, and lighting the same scene adds 44,916,352 — so `c` is
2,592,800 a triangle and the diffuse shade `s` is 1,403,636, a ratio of **0.54**.
The formula then predicts 1.980 and 1.298 against measured 1.984 and 1.299.

**So the crossover is a shader 1.85x this one.** `s/c > 1` is what it takes, and
a diffuse point light with one normalize is a little over half way there. That is
a falsifiable number rather than an intuition: a shader with a texture fetch and
a shadow term would pass it, and this one does not.

**Where the other half of the answer is.** `c` is what a prepass repeats, and the
tiled routes have already cut it — binning removes the triangles a pixel never
sees. Lighting them means growing the tile lists by a world position a vertex and
a normal a triangle, which is why the walk is the only route that can be lit
today, and why the more interesting version of this measurement is not yet
available.

---

## Several SMs, and the two things they buy

Until now a launch ran its blocks one after another: `myrt_launch` looped, and
`[m3]`'s warps could only cover each other *inside* one block. An SM is the thing
a block sits on, and there is now more than one of them, each holding more than
one block.

The defaults are one SM holding one block, which is the machine every figure
above was taken on — `machines/default.spec`, and turning it up is what the rest
of this section does. `draw_walk` at 64x32 over 64 blocks of one warp, with a
cache and latency modelled.

| SMs | blocks an SM | issued work | cycles | vs 1x1 | stalls |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1,718,778 | 567,142 | 1.00x | 507,020 |
| 1 | 8 | 1,718,778 | 76,490 | **7.41x** | 16,368 |
| 2 | 1 | 1,718,932 | 284,166 | 2.00x | 508,210 |
| 4 | 1 | 1,719,240 | 142,759 | 3.97x | 510,800 |
| 8 | 1 | 1,719,856 | 71,998 | 7.88x | 515,520 |
| 8 | 8 | 1,719,856 | 10,553 | **53.74x** | 23,498 |
| 16 | 8 | 1,721,088 | 9,446 | **60.04x** | 88,921 |
| 32 | 8 | 1,723,552 | 9,967 | 56.90x | 244,621 |
| 64 | 1 | 1,728,480 | 10,095 | 56.18x | 581,990 |

**The work does not move.** 1,718,778 against 1,728,480 across the whole table,
half a percent — and that residue is itself a finding, below. A cycle count that
fell without the work following it would mean blocks were being dropped rather
than overlapped, which is what `MoreSMsFinishSoonerWithoutDoingLessWork` exists
to catch.

**The two axes do different things.** More SMs divides the time — 2.00x, 3.97x,
7.88x, almost exactly linear — and leaves the stalls alone: each SM still holds
one warp, which still waits by itself. More blocks an SM removes the waiting:
stalls collapse from 507,020 to 16,368 while the same single SM does the work.
Parallelism splits the time; occupancy stops wasting it.

**And they trade against each other once the grid runs out.** The best row is 16
SMs of 8 blocks, not 32 of 8: sixty-four blocks spread over thirty-two SMs is two
apiece, and two warps cover less of a wait than eight do. Stalls say the same
thing louder — 88,921 against 244,621. A machine can be too wide for its work in
a way that shows up as waiting rather than as idleness.

### The residue: an SM keeps its own L1

Spreading the same blocks over more SMs costs half a percent more issued work,
and it is not noise. Every block of the naive walk reads the same triangles; on
one SM the first block fetches a line and the rest find it, and on sixty-four SMs
it is fetched sixty-four times. Splitting a cache is what that costs.

With L1 scaled to 16 lines so the scene outgrows it, both sides of co-residency
show:

| Route | blocks an SM | L1 hits | L2 hits | issued work |
|---|---:|---:|---:|---:|
| walk | 1 | 48,000 | 1,536 | 6,675,072 |
| walk | 8 | 49,344 | **192** | 6,645,504 |
| tiled | 1 | 9,684 | **7** | 1,341,542 |
| tiled | 8 | 9,488 | **203** | 1,345,854 |

The walk's blocks all read the same triangles, and sharing an SM turns seven L2
hits in eight into L1 hits. The tiled route's blocks each read their own tile,
have nothing to hand each other, and evict each other instead — L2 hits go the
other way and the bill rises 0.3%. **Co-residency is worth what the blocks have
in common**, and the two routes sit either side of that.

### Occupancy is not a knob every kernel can turn

The same sweep, one SM, all three raster routes. Cycles, so lower is better.

| blocks an SM | walk | tiled | shared |
|---:|---:|---:|---:|
| 1 | 567,142 | 15,463 | 19,356 |
| 2 | 283,962 | 11,557 | 19,356 |
| 4 | 143,477 | 10,730 | 19,356 |
| 8 | 76,490 | 10,714 | 19,356 |
| 16 | 60,548 | 10,714 | 19,356 |
| 32 | 60,124 | 10,714 | 19,356 |

Three shapes, and each says something.

**The walk gains most and saturates last.** Its blocks are one warp each, so a
block has nobody of its own to hide behind and every gain comes from
co-residency.

**The tiled route is nearly done at one.** Its blocks are 256 threads, eight
warps that already cover each other, so occupancy adds a little and then stops.

**The shared route cannot move at all.** It stages a tile through the whole
scratchpad and now says so at the launch, and residency is the smallest of what
the three limits allow: 16,384 bytes declared against 16,384 an SM is one block,
whatever is asked for. **Its occupancy is pinned at one.**

That is the price the flat cost model could not charge. Staging won 96% when a
global load cost 100 a lane; charging by the line took the win away (*Charging
memory by the line*); and this is the third bill — the route that trades global
traffic for shared memory cannot use the machine's latency hiding, because the
memory it traded for is the resource occupancy is made of.

### How the blocks are handed out decides where the ceiling sits

They go breadth first, one to each SM before any gets a second. Filling each SM
to capacity first looks equivalent and is not: `sphere.obj` at 128x64 is 256
blocks, and on 108 SMs holding 32 each the greedy order lands all of them on the
first eight while a hundred sit out — 80x against the 250x spreading gives.
Hardware's work distributor spreads for the same reason.

### What this cost every figure above

At the defaults — one SM holding one block — the machine is the one every earlier
table was taken on, with a single exception: **L1 belongs to an SM and is no
longer emptied between blocks.** Hardware does not flush it per block, and the
old model did, so blocks were being made to re-fetch what the block before them
had just read.

Everything under `MemoryModel::Flat` and `Coalesced` is unmoved to the byte —
neither consults a cache — and `render_bench.csv` is byte-identical. What moved is
every `Cached` and every cycle figure, and all of it in one direction:

| | before | after |
|---|---:|---:|
| Ray tracer, L2 hits (16 triangles) | 378 | **0** |
| Ordering a mesh, L2 hits at the hardware L1 | 3,677 | **23** |
| Wide load against three, in cycles | -42.9% | **-49.0%** |
| Indexed pass 2, in cycles | +13.9% | **+20.6%** |
| Predication on the ray tracer, in cycles | +46.0% | **+58.2%** |

A block used to start cold. Now it starts on whatever the block before it left,
which is worth more to a route whose blocks read the same data than to one whose
blocks read their own — and the figures that moved most are the ones where a
second reader was being charged for a line that was already there.

**No conclusion moved.** Predication still loses on every route and still loses by
an order of magnitude more once loads are charged by the line. The index buffer
still costs half what the flat model said. Early-Z still loses, and the ratio it
turns on is still `2/(1 + s/c)` with `s/c` still 0.54 and the crossover still a
shader 1.85x this one. Reordering a mesh still buys 1.6% of the issue capacity
and 24.7% of the cycles. The digits are new; the findings are the same ones.

---

## Summing a warp

The reduction is five rounds either way — the live values halve each time, and
log2(32) is five. What differs is a round.

```
cmake --build build -j8
./build/benchmarks/reduction_bench
```

| Warps | Route | Instructions | Warp steps | Lane ops |
|---|---|---:|---:|---:|
| 1 | shared | 68 | 68 | 2,176 |
| 1 | shuffle | 33 | 33 | **1,056** |
| 8 | shared | 68 | 544 | 17,408 |
| 8 | shuffle | 33 | 264 | **8,448** |

Shared memory spends a round on a store, a barrier, a load, a second barrier to
keep a fast lane from overwriting a slot a slow one has yet to read, and the
address arithmetic for both ends. The exchange spends it on one instruction.
Computing which lane to take from costs three instructions in both and cancels
out; it is written branchless so that a divergent wrap does not put warp splits
into a measurement about something else.

### What priced the primitives

`S_BALLOT`, `S_ANY`, `S_ALL` and `V_SHUFFLE_F32` were placeholders at 1 until
this ran, because a ballot reduces 32 lanes and a shuffle permutes them, and
neither is one lane-op.

The measurement does not name a cost on its own — it bounds one. A shuffle
would have to reach **22** before the two routes came level, so the 8 they now
carry, the same as a shared load, is the conservative end of what is open. The
independent argument for that number is that hardware runs the exchange through
the permute network the shared memory already has: AMD's `ds_bpermute` borrows
the LDS wiring outright.

Re-running with the primitives at 8 leaves the break-even at 22, which is the
check that the figure is not arguing in a circle: it is derived from the
instruction mix, not from the price.

### And what it leaves out

**This machine charges nothing for a barrier's stall.** `warp_steps` comes out
at exactly one per warp per instruction however many warps are in the block —
68, 136, 272, 544 — so waiting for the slowest warp is free here and is most of
what a barrier costs on hardware.

The comparison above therefore flatters shared memory rather than the exchange,
and the exchange wins anyway.

---

## Two renderers, one image

`ray_triangle` and `raster_triangle` write byte-identical PPMs on a square frame
at 64x64 and 128x128. Möller-Trumbore and edge functions share no arithmetic, so
agreement is evidence neither program can produce alone — the unit tests compare
each kernel against a host reference written from the same conventions, and a
sign wrong in both would pass.
