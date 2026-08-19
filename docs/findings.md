# What the optimisations bought

Two renderers draw the same triangle by different arithmetic —
Möller-Trumbore against edge functions — and their PPMs are byte-identical.
That is the check neither can make alone: each kernel is tested against a host
reference written from the same conventions, so a sign wrong in both would pass.

`raster_triangle` draws the frame twice, once per route, and writes the file
from the tiled one — so the PPM that matches the ray tracer is the optimised
path's output rather than something claimed to match it. It prints what binning
removed on the way: 32.6% for one triangle, 71.7% for sixteen.

The rasteriser was then made faster twice, measured in issued work rather than
wall clock so the figures reproduce. Sixteen small triangles over a 64x32 frame:

| | weighted lane ops | against the previous | against the walk |
|---|---:|---:|---:|
| every pixel walks every triangle | 41,534,048 | — | — |
| triangles binned into 32x8 tiles | 6,210,144 | -85.0% | -85.0% |
| each tile staged in shared memory | 1,752,864 | -71.8% | **-95.8%** |

The two changes remove different waste, which shows on a scene built to defeat
the first: when every triangle covers every tile, **binning loses 1.0%** having
nothing to drop, while staging still wins 85.7% — 256 threads were issuing the
same twelve global loads either way.

Staging needed `BARRIER`, the project's `__syncthreads()`. The fill and the
barrier sit outside the kernel's bounds check, because a thread whose pixel is
off screen still has to arrive; reaching a barrier under divergence is refused
rather than left to hang.

## An index buffer costs one pass what it saves the other

A cube has eight corners and thirty-six flattened vertices. Transforming each
corner once instead of once per triangle that uses it takes **76.4%** off pass
1 — the most expensive instruction in the set, run four and a half times less
often. The saving grows with how much a mesh shares: 83.1% on a 360-triangle
sphere.

Pass 2 then pays for it. A vertex's address is not known until its index has
arrived, so each triangle makes fifteen dependent global loads where the
flattened walk made twelve, and the pass costs **24.0%** more.

| cube, 64x32 | flattened | indexed | |
|---|---:|---:|---:|
| pass 1 — vertex transforms | 27,868 | 6,584 | **-76.4%** |
| pass 2 — coverage | 31,317,520 | 38,841,872 | +24.0% |

That 24% is what a flat charge per lane says, and it is roughly double what the
other cost models say: **+13.0% with a cache**, +20.6% in cycles with every load
charged a full wait. Every lane of a warp is on the same triangle, so all
thirty-two read the same three indices — one transaction, which is precisely the
case a per-lane charge overstates.

Which way the total falls depends on the ratio of vertices to pixels, so both
paths are kept and neither is the successor of the other. The tiled routes sit
outside the trade entirely: binning de-indexes on the host, so they take pass
1's saving and pay nothing in pass 2.

`simulated_cache_misses` counts what a 32-entry post-transform FIFO would have
missed, which is ACMR — the number the literature quotes. Reordering triangles
by Forsyth's linear-speed heuristic moves a shuffled sphere from **2.54 to
0.64** against a floor of 0.51.

What that is worth here depends on which cost model is asked, and the answer
changed when one arrived. A flat charge per lane cannot tell a vertex just read
from one that was not, and reported 0.004% — all of it triangles arriving in a
different depth order. Against a memory cache the mesh does not fit in, the same
reorder takes **1.6% off the issued work and 25% off the cycles** of the indexed
walk. Not by fetching less: every miss is compulsory either way, and what moves is
which level answered, L2 hits falling six-fold. The window closes as the cache
grows — at the hardware L1 this sphere fits with room to spare and the reorder wins
nothing, so what is measured is a mesh larger than its cache rather than this mesh
on this hardware.

## Removing divergence is not the same as going faster

The masking comparison the project set out to make. `V_CMP_F32` yields exactly
1.0 or 0.0, so coverage can select arithmetically instead of branching:

```
kept = take * new + (1 - take) * old
```

No lane disagrees, and the coverage test stops contributing to divergence
entirely. It is also slower — on every scene measured, including the tiled
route where warps straddle edges most:

| 16 small triangles, 64x32 | branch | blend | |
|---|---:|---:|---:|
| every pixel walks every triangle | 41,534,048 | 42,350,592 | +2.0% |
| binned into tiles (7.4% diverged) | 6,210,144 | 6,309,888 | +1.6% |
| ray tracer (15.0% diverged) | 31,556,398 | — | **+4.4%** |

The blend's cost barely moves between scenes — every lane shading every
triangle is the same work whatever is on screen — while the branch's tracks
what is actually covered.

**The ray tracer was the case that looked most likely to pay, and lost by
twice as much.** Four early exits guard nearly the whole of Möller-Trumbore
there, where the rasteriser's branch guards only a shade. Predicating them
means every lane finishes an intersection it would have abandoned at the first
test, and on a scene of small triangles most rays leave at the first test. The
scene where it loses least is the full-frame one, at 2.5%, where fewest rays
leave early at all.

So the trade is not decided by how much divergence there is to remove — the
rasteriser's quietest route and the tracer's noisiest both lose. What decides
it is how much work the branch was skipping, and more of the loop inside the
branch cuts both ways.

**And the flat charge above was flattering it by a factor of twenty.** Both
variants make the same loads; at 100 a lane those were most of the bill, and the
arithmetic the blend adds was lost against them. Charge a warp's load as the
cache line it touches and that arithmetic is most of what is left to compare:

| 16 small triangles, 64x32 | per lane | per line | with a cache | in cycles |
|---|---:|---:|---:|---:|
| every pixel walks every triangle | +2.0% | +28.3% | +47.5% | +17.9% |
| binned into tiles | +1.6% | +20.2% | +32.5% | +17.7% |
| ray tracer | +4.4% | +75.4% | **+93.1%** | +58.2% |

The divergence still goes — the blend measures exactly 0.00% on every route —
which makes this the strongest form of the finding rather than a retraction. What
changed is the price of buying it.

Folding the exits away also removes what they were guarding, which coverage
never had to worry about: the determinant reciprocal divides by zero once
nothing leaves early, and the running distance cannot start at infinity when
the blend multiplies it by a zero weight. Neither shows up on a cube — one
needs a degenerate triangle, the other two triangles ordered far-first — so
both scenes are in the tests.

The blend is written as `take*new + (1-take)*old` rather than the cheaper
`old + take*(new-old)`, which rounds: 69,715 pixels of 200,000 drifted, and at
64 triangles that reached the image. A variant built to be compared against the
branch has to agree with it to the bit.

## Lanes talking, rather than lanes disagreeing

Everything above is about warps that split. The other half of a SIMT machine is
warps that cooperate, and until `S_BALLOT` and `V_SHUFFLE_F32` the ISA could not
express it — summing 32 lanes meant a round trip through shared memory.

| Summing a warp | Instructions | Warp steps |
|---|---:|---:|
| through shared memory | 68 | 68 |
| through the lane exchange | **33** | **33** |

Five rounds either way, the live values halving each time. A round through
shared memory is a store, a barrier, a load, a second barrier and the addressing
for both ends; through the exchange it is one instruction.

That measurement is also what priced the primitives. They sat at 1 while
unmeasured, so weighting by them would only have reported the guess back —
instead the weighting is turned around to ask how expensive a shuffle would have
to be before the two came level. It is 22, so the 8 they now carry, the same as
a shared load, is the conservative end. And the comparison understates shared
memory rather than the exchange: this machine charges nothing for a barrier's
stall, which on hardware is most of what a barrier is.

Every figure above is reproducible from a committed scene, and each names the
tool that produced it: the frame-level tables come from
`./build/test/benchmark/render_bench`, which defines its scenes in code and drives
the same routes the tests call; the per-pass index-buffer figures come from
`./build/apps/mesh_render` reading `assets/`, because the draw routes clear
the counters between passes and a caller otherwise reads pass 2 alone; the ACMR
numbers come from `simulated_cache_misses`, which scores a mesh rather than
running one; the reduction comes from `./build/test/benchmark/reduction_bench`.

Both renderers take their camera from one `DrawTarget`, so a comparison between
them cannot be of two different views. `test/benchmark/RESULTS.md` carries the full
tables, the method, and a prediction about divergence that the measurements
contradicted.

---

[← back to the README](../README.md)
