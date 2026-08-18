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
| 1 | 8 | 1,718,778 | 76,484 | **7.42x** | 16,362 |
| 2 | 1 | 1,718,932 | 284,166 | 2.00x | 508,210 |
| 4 | 1 | 1,719,240 | 142,710 | 3.97x | 510,590 |
| 8 | 1 | 1,719,856 | 71,982 | 7.88x | 515,350 |
| 8 | 8 | 1,719,856 | 10,553 | **53.74x** | 23,170 |
| 16 | 8 | 1,721,088 | 9,347 | **60.68x** | 88,628 |
| 32 | 8 | 1,723,552 | 9,593 | 59.12x | 244,259 |
| 64 | 1 | 1,728,480 | 10,095 | 56.18x | 581,990 |

**The work does not move.** 1,718,778 against 1,728,480 across the whole table,
half a percent — and that residue is itself a finding, below. A cycle count that
fell without the work following it would mean blocks were being dropped rather
than overlapped, which is what `MoreSMsFinishSoonerWithoutDoingLessWork` exists
to catch.

**The two axes do different things.** More SMs divides the time — 2.00x, 3.97x,
7.88x, almost exactly linear — and leaves the stalls alone: each SM still holds
one warp, which still waits by itself. More blocks an SM removes the waiting:
stalls collapse from 507,020 to 16,362 while the same single SM does the work.
Parallelism splits the time; occupancy stops wasting it.

**And they trade against each other once the grid runs out.** The best row is 16
SMs of 8 blocks, not 32 of 8: sixty-four blocks spread over thirty-two SMs is two
apiece, and two warps cover less of a wait than eight do. Stalls say the same
thing louder — 88,628 against 244,259. A machine can be too wide for its work in
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
| 2 | 284,185 | 11,557 | 19,356 |
| 4 | 143,332 | 10,730 | 19,356 |
| 8 | 76,484 | 10,714 | 19,356 |
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

They go breadth first, one to each SM before any gets a second — from an empty
machine, which is newer than it sounds: the block that used to decide residency was
placed before the fill began, so SM 0 received a second before SM 1 had its first.
Residency is counted a block at a time now, and that prologue went with it.

Filling each SM to capacity first looks equivalent and is not: `sphere.obj` at 128x64 is 256
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

## Two queues

An SM can hold two blocks; until now they always came from the same launch. A
stream is a queue, launches in one run in order, and launches in different ones
are free to overlap — which is the first time this machine has been asked to run
two kernels at once.

```
cmake --build build -j8
./build/benchmarks/stream_bench
./build/benchmarks/stream_bench --machine machines/a100.spec
```

### A machine one grid cannot fill

Eight blocks of compute on eight SMs holding one block each, cut into launches.
The work is identical in every row; only the number of queues changes.

| Launches | blocks each | one stream | n streams | speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 90 | 90 | 1.00x |
| 2 | 4 | 180 | 90 | **2.00x** |
| 4 | 2 | 360 | 90 | **4.00x** |
| 8 | 1 | 720 | 90 | **8.00x** |

**A grid that fits gains nothing, and one that does not gains everything.** The
first row is the control: eight blocks already reach eight SMs, and a second
queue has nowhere to put anything. Every row below it is the same work through a
narrower door, and concurrency opens the door back up — the eight single-block
launches take eight times as long in one queue and exactly as long as one launch
in eight.

This is the case a stream exists for, and it is common: a shadow map, a
post-process, a reduction over a small buffer. None of them is wide enough to
fill a modern machine, and the machine only gets wider.

### Covering one kernel's waiting with another

Two launches on one SM holding two blocks. `memory` is sixteen loads of a line
each, `compute` sixteen fused multiply-adds; they read separate buffers, so
nothing here is one of them finding a line the other fetched.

| First | second | 1st alone | 2nd alone | one stream | two streams | hidden |
|---|---|---:|---:|---:|---:|---:|
| compute | compute | 90 | 90 | 180 | 92 | 97.78% |
| memory | memory | 6,490 | 6,490 | 12,980 | 6,492 | **99.97%** |
| memory | compute | 6,490 | 90 | 6,580 | 6,490 | **100.00%** |

*hidden* is what concurrency took off the shorter of the two, which is the most it
could take: nothing retires sooner than the longer kernel does alone.

**Two memory kernels cost one.** A warp waiting 400 cycles for a line leaves 399
issue slots behind it, and a second kernel needs 22. The waiting is so much larger
than the work that the second launch is free — `[m3]` found this among warps of
one kernel, and it turns out not to care whether the warps came from the same
launch at all.

**Which is the honest form of the usual advice.** Pairing a memory-bound kernel
with a compute-bound one is what streams are recommended for, and it is the row
that hides everything — but so is memory with memory, on this machine. The
mechanism is idle issue slots, not complementary units: there are no separate
units here to complement. A machine with distinct load and arithmetic pipes would
separate those two rows, and this one cannot.

### A grid the host never learns

A cull pass adds up eight visibility flags and writes the total where the launch
after it reads its grid. `host` is that same grid launched the ordinary way.

| Visible | grid | indirect work | cycles | host work | cycles |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 6,924 | 900 | 6,688 | 90 |
| 4 | 4 | 3,580 | 900 | 3,344 | 90 |
| 2 | 2 | 1,908 | 900 | 1,672 | 90 |
| 0 | 0 | 236 | 810 | 0 | 0 |

**Reading the grid from memory costs nothing.** Every row's difference in issued
work is 236, and the last row says what 236 is: the cull, on its own, deciding
nothing. The indirect launch itself adds not one lane-op — the grid is read once
on the host side of the simulator, between one launch retiring and the next
starting, which is where hardware's command processor reads it too.

**What it does cost is a serial pass.** 810 of those 900 cycles are the cull, and
it is one thread walking eight flags because this ISA has no atomic to combine
lanes with. The shape is right — a pass that looks at everything and says how much
survived — and the implementation is the slowest possible one. A real culling
pass is one thread per candidate and a reduction; that needs an atomic or a
prefix sum, and neither exists here yet.

**Zero survivors is a launch that runs.** The last row's grid is empty and the
draw issues nothing at all, without the host ever being told. That is the whole
premise of GPU-driven rendering, and it is now expressible.

### What this cost the figures above

Two scheduler rules changed with it, and both were wrong before streams made them
visible.

**The clock no longer jumps past a block that arrived this cycle.** A slot taking a
new block while the SM's other warps were waiting used to be skipped over to the
moment the waiters wake, so a block that could have issued immediately sat out the
gap. A second stream's block arriving into an SM stalled on the first would have
been made to wait for a kernel it has nothing to do with.

**Blocks are handed out breadth first from an empty machine.** Residency is counted
a block at a time now — two kernels on one SM have no single residency number
between them — and the prologue that placed the first block before the fill went
with it. That prologue was an off-by-one at the head of a breadth-first pass: SM 0
received a second block before SM 1 had its first, and at 64 blocks on 64 SMs it
left the last SM empty.

Only rows with more than one SM or more than one block an SM can reach either, so
the defaults are untouched: `render_bench`, `cache_bench` and `model_bench` are
byte-identical. What moved is inside *Several SMs*, by tenths of a percent and in
both directions:

| | before | after | which rule |
|---|---:|---:|---|
| 1 SM, 8 blocks | 76,490 | 76,484 | the clock |
| 8 SMs, 1 block | 71,998 | 71,982 | the clock |
| 16 SMs, 8 blocks | 9,446 | **9,347** | the fill |
| 32 SMs, 8 blocks | 9,967 | **9,593** | the fill |
| 1 SM, 2 blocks | 283,962 | 284,185 | the clock |

Issuing sooner can cost more, which is why the last row goes the other way: it
reorders which warp reaches a line first, and a cache answers the second asker more
cheaply than the first.

---

## A tiled matrix multiply

Two instructions were measured and neither had a kernel worth measuring in.
`V_MMA_16X16X16_F32` had none at all; `cp.async` had one whose staged tile was
read 256 times, so hiding the fetch could not matter and the section above says
where to ask again. This is that place.

```
cmake --build build -j8
./build/benchmarks/gemm_bench
```

C[64x256] = A × B, a block of four warps holding a 16×64 strip of C in registers
for the whole K loop, staging one A tile and four B tiles a step. Cycles.

| K | fma, sync | mma, sync | change | mma, staged ahead | change | + wide fragments | change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 202,556 | 44,220 | **-78.2%** | 33,056 | **-25.2%** | 20,256 | **-38.7%** |
| 64 | 401,734 | 85,062 | **-78.8%** | 62,340 | **-26.7%** | 36,746 | **-41.1%** |
| 128 | 812,660 | 179,316 | **-77.9%** | 120,977 | **-32.5%** | 69,809 | **-42.3%** |

Issued work at K = 128: 27,145,216, then 5,960,704, then 4,762,624 — and 4,762,624
again. **The last two are equal to the lane-op**, which is the whole of what the
fragment load is: eight floats are eight floats however they are asked for.

End to end, the last column is 11.6x the first.

**cp.async is worth 33% here and was worth 1.8% in the renderer.** Same
instruction, same double buffering, same scheduler. What changed is the ratio the
earlier section named: a staged tile is consumed by a fixed, small number of
operations instead of being read by every thread of the block. The prediction was
written before this kernel existed and is what the kernel was built to test.

**The matrix unit is worth 78%, which is less than the instruction counts
suggest.** One MMA replaces 128 multiply-adds a lane, but it does not replace the
loads that assemble its fragments: sixteen shared loads a lane a k-step, against
one instruction that consumes them. At a cost of 8 each that is 128 a lane of
loading to feed 16 of multiplying — **the fragment load, not the multiply, is
what the inner loop spends its capacity on.**

### The instruction that measurement asked for

`V_LD_SHARED_16X16_F32` reads a whole fragment: eight consecutive floats into the
eight consecutive registers the matrix instruction wants. Hardware's `ldmatrix`,
and the argument `V_LD_GLOBAL_VEC3_F32` already made once — a wide load is not
several narrow ones.

**What it buys is different, though, and the two are worth reading together.**
The wide global load buys *transactions*: 32 lanes asking for twelve bytes touch
fewer cache lines than three instructions do, so it took 24.7% off a count of
lines and nothing at all off `Flat`. Shared memory has banks rather than lines,
and this machine does not model a bank conflict — so there is no transaction to
save, and the cost table says so by charging exactly eight floats for eight
floats.

**It buys waiting instead.** This machine issues in order with no scoreboard, so
eight round trips to shared memory are eight waits with nothing between them; one
is one. That is 42% of the cycles at K = 128, with the issued work identical to
the lane-op — the sharpest separation in this file between what a kernel costs
and how long it takes.

### What the shape of the kernel had to work around

The block is 2 × 16 × 4 threads. Every index the kernel needs — which half of a
fragment row a lane holds, which row, which warp — is a coordinate, because the
ISA has no integer divide and a flat thread index cannot be taken apart. It is the
first kernel where that constraint decided the launch geometry rather than being
worked around inside the kernel.

Addresses are advanced in place rather than rebuilt. Every arithmetic call in the
builder allocates a register, the staging is emitted twice in the double-buffered
form, and the obvious spelling wanted about 120 registers of the 250 a thread has.

---

## A block reading its neighbour's shared memory

Blocks cannot see each other. Anything one makes for another goes out to global
memory and comes back — and since there is no block-wide rendezvous inside a
launch, it crosses a kernel boundary as well. A cluster removes both: the blocks
are placed together, `BARRIER_CLUSTER` is the rendezvous, and `V_LD_CLUSTER_F32`
is the read.

```
cmake --build build -j8
./build/benchmarks/cluster_bench
```

256 elements made once and summed by every block.

| Blocks | Global work | cluster | change | Global cycles | cluster | change |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7,676 | 15,490 | +101.8% | 2,223 | 976 | **-56.1%** |
| 4 | 11,320 | 26,046 | +130.1% | 2,623 | 976 | **-62.8%** |
| 8 | 18,608 | 47,158 | +153.4% | 3,223 | 976 | **-69.7%** |

**The cluster's clock does not move with the block count and the global route's
does.** Every consumer reads out of one producer's shared memory at the same
time; through global memory they queue behind a second launch that cannot start
until the first has entirely finished.

**It costs issue capacity to save time.** A neighbour's shared memory is charged
per lane, like this machine's own shared memory and for the same reason — banks,
not cache lines. A coalesced global read of the same 32 addresses is one line.
So the cluster route does more than twice the issued work and still finishes in
a third of the cycles, and which of those a kernel cares about is the trade.

**What it does not show is a saving on fetches.** The first thing tried here was
four blocks each staging the same table, against one staging it and three reading
across. That barely moved: the cache had already done it — the second block's
fetch was an L2 hit and the third's an L1 hit. **Distributed shared memory is for
what a block computes, not for what it fetches**, and the benchmark had to be
rewritten around a producer before it measured anything.

### What co-residency costs the scheduler

A cluster is placed whole or not at all, and none of its blocks is freed until
all of them have retired — a neighbour may still be reading the shared memory a
finished block owns. Hardware makes the same promise for the same reason.

Two consequences are stated as errors rather than left to be discovered: a
cluster larger than the machine can hold at once is a launch that cannot start,
and a grid that ends part-way through a cluster is refused. Both would otherwise
appear as a barrier nobody can pass.

---

## Regrouping the threads before they disagree

A warp issues one instruction, so lanes that disagree take turns — that is the
cost this whole project is about. `REORDER` moves the block's threads between
its warps by a key the kernel supplies, so the ones about to do the same thing
end up together. Nothing a thread holds changes; registers and pc travel with
it, which is why the answer is identical and only the divergence moves.

```
cmake --build build -j8
./build/benchmarks/ser_bench
```

### It is worth what the key is scattered

A block of 8 warps and a 32-instruction branch, by how the threads taking it sit.

| Spread | Taking | Warp steps | reordered | change | Cycles | reordered | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| scattered | 32 | 288 | 71 | **-75.3%** | 288 | 164 | **-43.1%** |
| scattered | 128 | 288 | 164 | **-43.1%** | 288 | 164 | -43.1% |
| scattered | 224 | 288 | 257 | -10.8% | 288 | 257 | -10.8% |
| coherent | 32 | 63 | 71 | **+12.7%** | 153 | 164 | +7.2% |
| coherent | 128 | 156 | 164 | +5.1% | 156 | 164 | +5.1% |
| coherent | 224 | 249 | 257 | +3.2% | 249 | 257 | +3.2% |

**The same instruction is a 75% saving and a 13% tax**, and what decides which is
whether the threads were already sorted. A coherent block has nothing to gain and
still pays for the sort — reordering is not a thing to switch on, it is a thing
to reach for when the key is known to be scattered.

**The scattered rows converge from above.** At 32 takers the reorder empties
seven warps of the branch entirely; at 224 there is only one warp's worth of
non-takers to gather, and the saving shrinks to what that is worth.

### And what follows it

Half the block taking a scattered branch, by how long the branch is.

| Branch | Warp steps | reordered | change | Cycles | reordered | change |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 44 | -8.3% | 48 | 44 | -8.3% |
| 8 | 96 | 68 | -29.2% | 96 | 68 | -29.2% |
| 32 | 288 | 164 | -43.1% | 288 | 164 | -43.1% |
| 128 | 1,056 | 548 | **-48.1%** | 1,056 | 548 | **-48.1%** |

**The sort is a fixed cost against a saving that grows with the stretch it
protects.** Two instructions barely repay it; 128 approach the ceiling, which is
half — a warp that visited both arms visiting one.

### What this can and cannot say

The regroup is **within a block**. Hardware reorders across an SM's resident
warps, which is a wider pool; a block here is at most 8 warps, so these figures
are the pessimistic end of what the mechanism can do.

And the keys are synthetic. What this repository has that a real reordering
workload has is the ray tracer's hit-and-miss divergence, measured at 10-21%,
against a rasteriser's 1-7%. Neither has the thing SER was built for — many
distinct materials behind one intersection, so that lanes diverge into *different
shaders* rather than into two arms. That needs a BVH and a material table, which
is `[g]`, and this measurement is meant to be repeated there rather than to stand
in for it.

---

## A matrix unit against the arithmetic pipe

`V_MMA_16X16X16_F32` has the warp compute a whole 16x16x16 product at once —
4,096 multiply-adds, 128 for each of its 32 lanes — where the arithmetic pipe
issues one fused multiply-add at a time.

```
cmake --build build -j8
./build/benchmarks/mma_bench
```

| Tiles | Route | Instructions | Warp steps | Lane ops | Cycles |
|---:|---|---:|---:|---:|---:|
| 1 | fma | 129 | 129 | 4,128 | 513 |
| 1 | mma | **2** | **2** | **544** | **33** |
| 4 | fma | 513 | 513 | 16,416 | 2,049 |
| 4 | mma | **5** | **5** | **2,080** | **129** |
| 16 | fma | 2,049 | 2,049 | 65,568 | 8,193 |
| 16 | mma | **17** | **17** | **8,224** | **513** |

**The instruction count is the part that is not a claim.** One tile is 128
multiply-adds a lane against one instruction, and that is arithmetic: the shape
says so. The 16x-deep latency and the cost of 16 are chosen numbers, so the
table is read the way `reduction_bench`'s is — counted columns first, price last.

**What would make the two level is 128.** Priced at 16, the matrix instruction
claims a unit that retires the work eight times faster than the lanes would one
FMA at a time; 8x is the conservative end of what hardware's tensor cores are
quoted at, and the honest form of the claim is that number rather than the ratio
it produces.

### What it took to add

The first instruction here whose operands are the warp's registers rather than a
lane's. A `V_SHUFFLE_F32` reaches across lanes but computes something per lane;
this one cannot be started by any lane alone, because no lane holds a whole row
of A or column of B. Every lane must take part, and a participation mask naming
fewer is refused rather than producing a partial answer.

The fragment layout is row-major across the warp, eight elements a lane, chosen
for being explainable. Hardware's layouts are chosen for the datapath and
modelling one would say nothing this does not.

### A latency that was never being applied

Adding it found a bug three opcodes old. The warp-level path — ballot, any, all,
shuffle, and now this — counted its work and returned without setting a latency,
so every one of them was charged whatever the instruction before it happened to
cost. It showed up here because a matrix instruction is 32 cycles deep and was
reporting one.

The reduction figures above move with the fix: a warp's five shuffles are 149
cycles rather than 129, and the atomic's margin over the exchange becomes 3.1x
where it read 3.3x. Nothing else in `RESULTS.md` touches those opcodes.

---

## A copy the warp does not wait for

`cp.async` moves global memory into shared memory without a register in between,
and the warp does not wait for it — it issues the copy, carries on, and meets it
later at `S_CP_ASYNC_WAIT`. What it replaces is a load into a register and a
store out of it, with the warp waiting out the load in the middle.

```
cmake --build build -j8
./build/benchmarks/async_bench
```

### The mechanism

A fill and nothing else: one warp, a cache line a float, so every float misses.

| Floats | sync issued | async issued | change | sync cycles | async cycles | change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3,232 | 1,216 | -62.4% | 3,250 | 449 | **-86.2%** |
| 16 | 6,336 | 2,272 | -64.1% | 6,490 | 849 | **-86.9%** |
| 32 | 12,544 | 4,384 | -65.1% | 12,970 | 1,649 | **-87.3%** |
| 64 | 24,960 | 8,608 | -65.5% | 25,930 | 3,249 | **-87.5%** |

**The issued work falls because the shared store is gone**, two instructions
becoming one and the register between them never written.

**The cycles fall by the queue depth.** A warp may hold eight copies in flight,
so eight trips to memory overlap instead of happening one after another —
1/8 is 87.5%, and 87.5% is what the last row says. The constant is not a
guess that happened to work: it is the number the measurement recovers.

### The same mechanism in a renderer

`draw_shared` stages a tile into shared memory and then walks it. The
asynchronous form takes the tile a chunk at a time and issues the next chunk's
copies before walking this one, so the fetch and the walk overlap.

| Scene | triangles | sync issued | async issued | change | sync cycles | async cycles | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| small, spread | 16 | 739,844 | 833,028 | +12.6% | 19,500 | 19,976 | **+2.4%** |
| full-frame, stacked | 16 | 4,861,500 | 4,936,892 | +1.6% | 100,834 | 100,359 | -0.5% |
| full-frame, stacked | 64 | 16,491,932 | 16,517,068 | +0.2% | 335,258 | 330,398 | -1.4% |
| full-frame, stacked | 128 | 25,913,884 | 25,955,404 | +0.2% | 524,954 | 515,678 | **-1.8%** |

**Eighty-seven percent becomes two.** The frames are identical and the mechanism
is the same one; what changed is what fraction of the kernel it applies to.

**A staged float is fetched once and read 256 times** — once by each thread of the
block, since every pixel of the tile walks every triangle in it. Hiding the fetch
can therefore save at most 1/256 of the walk, and no scene can move that: it is a
property of the kernel. The gain grows with the triangle count only because more
triangles mean more chunks and so more overlap, and it converges on that ceiling.

**The small scene loses outright.** A chunk is 64 triangles because
`S_CP_ASYNC_WAIT` takes a count and the kernel has to know how many copies a
chunk is; a tile of 16 triangles therefore stages four times what it needs. That
is +12.6% of issued work on a route where the fetch is 2% of the bill, and it
prices exactly what a compile-time wait count costs.

### Where it would have paid

The trade is fetch against reuse, and this kernel is at the wrong end of it: it
stages once and reads 256 times. `cp.async` was introduced for the other end —
a matrix multiply's inner loop stages a tile and consumes it in a fixed, small
number of multiply-accumulates, so the fetch is a real fraction of the work and
double buffering hides it. That kernel is `[m6]`, and it is where this
instruction will be measured again.

### What it bought that is not a number

The double-buffered route has no tile it cannot hold. The synchronous one stages
the tile in a single pass and refuses anything over 341 triangles — real hardware
splits an overfull tile across passes, and taking it a chunk at a time is that
split. The refusal was a property of filling in one pass, not of the route.

---

## Summing a warp

The reduction is five rounds either way — the live values halve each time, and
log2(32) is five. What differs is a round.

```
cmake --build build -j8
./build/benchmarks/reduction_bench
```

| Warps | Route | Instructions | Warp steps | Lane ops | Cycles |
|---|---|---:|---:|---:|---:|
| 1 | shared | 68 | 68 | 2,176 | 354 |
| 1 | shuffle | 33 | 33 | **1,056** | **149** |
| 1 | atomic a lane | **5** | **5** | **160** | 1,143 |
| 1 | shuffle+atomic | 39 | 39 | 1,155 | 363 |
| 8 | shared | 68 | 544 | 17,408 | 654 |
| 8 | shuffle | 33 | 264 | **8,448** | **264** |
| 8 | atomic a lane | **5** | 40 | 1,280 | 1,162 |
| 8 | shuffle+atomic | 39 | 312 | 9,240 | 504 |

Shared memory spends a round on a store, a barrier, a load, a second barrier to
keep a fast lane from overwriting a slot a slow one has yet to read, and the
address arithmetic for both ends. The exchange spends it on one instruction.
Computing which lane to take from costs three instructions in both and cancels
out; it is written branchless so that a divergent wrap does not put warp splits
into a measurement about something else.

### The atomic routes answer a different question

They leave the total **in memory**, so it accumulates across the block's warps
where the two above leave a per-warp answer in a register. That is the reason to
reach for one at all, and it is also what makes the choice between them real.

**The issue count says the atomic wins and the clock says it loses by 3.1x.**
Five instructions against thirty-nine, 160 lane-ops against 1,155 — and 1,143
cycles against 363. Thirty-two lanes naming one address are thirty-two
operations, and the unit works through them one at a time.

That serialisation is charged rather than assumed: `atomic_access` counts how
many lanes pile onto the deepest address and adds a step for each one past the
first. Coalescing cannot help — the test `CoalescingCannotHelpAnAtomic` pins that
an atomic costs the same under `Flat`, `Coalesced` and `Cached`, where a load of
the same 32 addresses costs a thirty-second as much under the last two.

**So the reduction is what carries a value out of a warp**, and the atomic is
what carries it out of the block. Doing the first before the second is not an
optimisation to reach for later — it is 3.1x here, and it grows with the warp.

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
