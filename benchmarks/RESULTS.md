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

## Two renderers, one image

`ray_triangle` and `raster_triangle` write byte-identical PPMs on a square frame
at 64x64 and 128x128. Möller-Trumbore and edge functions share no arithmetic, so
agreement is evidence neither program can produce alone — the unit tests compare
each kernel against a host reference written from the same conventions, and a
sign wrong in both would pass.
