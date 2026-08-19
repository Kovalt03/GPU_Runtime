# The programs, and what each one asks

Every executable writes into `test/benchmark/output/` — tables as `.md` and
`.csv`, images and animations under `images/`. `test/benchmark/RESULTS.md`
carries the accumulated record, the method behind each table, and the
predictions the measurements contradicted.

```bash
# A shader of your own — read this one first
./build/apps/hello_shader

# A camera going round something, written out as an animation
./build/apps/orbit                        # result/images/orbit_cube.gif
./build/apps/orbit --shape sphere

# Render (PPM, no image library needed)
./build/apps/ray_triangle 256 256
sips -s format png test/benchmark/output/result.ppm --out result.png   # macOS
convert test/benchmark/output/result.ppm result.png                    # ImageMagick

# Benchmarks
./build/test/benchmark/divergence_bench
./build/test/benchmark/divergence_bench --csv   # test/benchmark/output/divergence.csv

# Four routes to one frame — walk, tiled, shared memory, ray tracer
./build/test/benchmark/render_bench             # test/benchmark/output/render_bench.{md,csv}

# An .obj down every route, compared pixel for pixel
./build/apps/mesh_render assets/sphere.obj

# Summing a warp: shared memory against the lane exchange
./build/test/benchmark/reduction_bench          # test/benchmark/output/reduction.{md,csv}

# A cache against a growing working set
./build/test/benchmark/cache_bench              # test/benchmark/output/cache.{md,csv}

# The flat model's conclusions, put to the other cost models
./build/test/benchmark/model_bench              # test/benchmark/output/models.{md,csv}

# What several SMs buy, and what stops a kernel from using them
./build/test/benchmark/occupancy_bench          # test/benchmark/output/occupancy.{md,csv}

# What a second queue buys, and a grid the host never learns
./build/test/benchmark/stream_bench             # test/benchmark/output/stream.{md,csv}

# A copy the warp does not wait for, on its own and in a renderer
./build/test/benchmark/async_bench              # test/benchmark/output/async.{md,csv}

# One instruction against 4,096 multiply-adds
./build/test/benchmark/mma_bench                # test/benchmark/output/mma.{md,csv}

# What regrouping a block's threads is worth, and when it is a tax
./build/test/benchmark/ser_bench                # test/benchmark/output/ser.{md,csv}

# A block reading its neighbour's shared memory
./build/test/benchmark/cluster_bench            # test/benchmark/output/cluster.{md,csv}

# A tiled matrix multiply — what the matrix unit and cp.async were waiting for
./build/test/benchmark/gemm_bench               # test/benchmark/output/gemm.{md,csv}

# The device deciding how much to draw
./build/test/benchmark/cull_bench                # test/benchmark/output/cull.{md,csv}
./build/test/benchmark/cull_bench --machine machines/four-sm.spec

# Reordering the threads, on divergence the scene put there
./build/test/benchmark/material_bench            # test/benchmark/output/material.{md,csv}

# One tree over the copies, or one over all of them
./build/test/benchmark/tlas_bench                # test/benchmark/output/tlas.{md,csv}

# Where an instance's model matrix meets the view-projection
./build/test/benchmark/instance_bench            # test/benchmark/output/instance.{md,csv}

# A tree against every triangle for every pixel
./build/test/benchmark/bvh_bench                # test/benchmark/output/bvh.{md,csv}
./build/test/benchmark/bvh_bench --leaf 8       # triangles a leaf may hold

# What happens when the memory system has a ceiling
./build/test/benchmark/bandwidth_bench          # test/benchmark/output/bandwidth.{md,csv}

# Any of them on another machine — machines/ holds the files
./build/test/benchmark/render_bench --machine machines/a100.spec
```

### Formatting

Layout is decided by `.clang-format`, never by hand. The config is derived from
the style already in the tree: 4-space indent, 90-column limit, function braces
on their own line, `int* ptr`.

---

[← back to the README](../README.md)
