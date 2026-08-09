# Measurements

A running record, so that each change can be stated as a number rather than as
an intention. Every figure here comes from the counters below, at the revision
named in its own section.

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

| Scene | Triangles | Worst tile | walk | tiled | Change |
|---|---:|---:|---:|---:|---:|
| small, spread over the frame | 4 | 1 | 8,427,632 | 2,064,496 | **-75.5%** |
| small, spread over the frame | 16 | 2 | 31,701,456 | 4,979,152 | **-84.3%** |
| small, spread over the frame | 64 | 16 | 124,796,616 | 24,374,984 | **-80.5%** |
| medium, stacked at the centre | 16 | 16 | 31,701,064 | 16,610,376 | -47.6% |
| full-frame, stacked | 16 | 16 | 31,709,184 | 32,139,264 | **+1.4%** |

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
| 16 small, spread | -84.3% | 0.4% | 2.1% |
| 64 small, spread | -80.5% | 0.3% | 1.7% |

Five times the divergence rate for a sixth of the work. Anyone tuning on
`divergence_rate` alone would read this change as a regression.

---

## Two renderers, one image

`ray_triangle` and `raster_triangle` write byte-identical PPMs on a square frame
at 64x64 and 128x128. Möller-Trumbore and edge functions share no arithmetic, so
agreement is evidence neither program can produce alone — the unit tests compare
each kernel against a host reference written from the same conventions, and a
sign wrong in both would pass.
