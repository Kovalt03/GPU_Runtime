#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ir_builder.hpp"
#include "pipeline/raster_tiled.hpp"
#include "raster_emit.hpp"
#include "thread.hpp"  // WARP_SIZE, the launch width

// ---------------------------------------------------------------------------
// Tiling
//
// The same picture as the walk above, reached by reading a shorter list. What
// changes is only which triangles a pixel ever sees, so the two are directly
// comparable — and have to produce identical frames.
// ---------------------------------------------------------------------------

TileBinning bin_triangles(const std::vector<ScreenTriangle>& triangles, uint32_t width,
                          uint32_t height)
{
    // tiles_x = ceil(width / TILE_WIDTH), tiles_y likewise. Rounding up, so the
    // last tile is partly off screen and its threads fall out on the bounds
    // check the kernel already has.
    TileBinning binning;
    binning.tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    binning.tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    // For each tile, walk every triangle and keep the ones whose screen
    // bounding box overlaps it:
    //
    //   min/max of the three x and the three y
    //   overlap when  min.x < (tx + 1) * TILE_WIDTH  and  max.x >= tx * TILE_WIDTH
    //   and the same along y
    struct Box {
        float min_x, min_y, max_x, max_y;
    };
    std::vector<Box> boxes;

    for (const ScreenTriangle& t : triangles) {
        boxes.push_back(
            Box{std::min({t.v0.x, t.v1.x, t.v2.x}), std::min({t.v0.y, t.v1.y, t.v2.y}),
                std::max({t.v0.x, t.v1.x, t.v2.x}), std::max({t.v0.y, t.v1.y, t.v2.y})});
    }

    for (uint32_t ty = 0; ty < binning.tiles_y; ++ty) {
        for (uint32_t tx = 0; tx < binning.tiles_x; ++tx) {
            const float left = static_cast<float>(tx * TILE_WIDTH);
            const float top = static_cast<float>(ty * TILE_HEIGHT);

            binning.table.push_back(
                static_cast<float>(binning.vertices.size() / TILE_TRIANGLE_FLOATS));
            uint32_t count = 0;

            for (size_t i = 0; i < triangles.size(); ++i) {
                const Box& b = boxes[i];
                // Written as "no overlap", which is four independent rejections
                // rather than four conditions that all have to line up.
                if (b.max_x < left || b.min_x >= left + TILE_WIDTH) {
                    continue;
                }
                if (b.max_y < top || b.min_y >= top + TILE_HEIGHT) {
                    continue;
                }

                // Interleaved as pass 1 writes it — position then 1/w, three
                // times — so a kernel reads a binned triangle and a screen one
                // with the same offsets.
                const ScreenTriangle& t = triangles[i];
                const float reciprocals[3] = {t.inv_w0, t.inv_w1, t.inv_w2};
                const Float3 corners[3] = {t.v0, t.v1, t.v2};
                for (uint32_t c = 0; c < 3; ++c) {
                    binning.vertices.push_back(corners[c].x);
                    binning.vertices.push_back(corners[c].y);
                    binning.vertices.push_back(corners[c].z);
                    binning.vertices.push_back(reciprocals[c]);
                }
                ++count;
            }
            binning.table.push_back(static_cast<float>(count));
        }
    }
    return binning;
}

Program build_tiled_raster_program(void** args)
{
    const TiledRasterStageArgs& a = *static_cast<const TiledRasterStageArgs*>(args[0]);
    IRBuilder k;

    // The last tile in each direction hangs off the screen, so the bounds check
    // is the same one the untiled kernel needs.
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> half = k.constant(0.5f);

        const Reg<Scalar> cx = k.add(px, half);
        const Reg<Scalar> cy = k.add(py, half);

        // Which tile this block covers — the whole reason blockIdx exists. A
        // global coordinate cannot be divided back down to it.
        //
        // Uniform across the block, so all 32 lanes of a warp load the same two
        // floats. A scalar unit is what real hardware would use for this, and
        // the S_ prefix the ISA reserves is for exactly that.
        const Reg<Scalar> tile = k.add(
            k.mul(k.block_y(), k.constant(static_cast<float>(a.tiles_x))), k.block_x());
        const Reg<Scalar> table_addr = k.mul(tile, k.constant(2.0f * sizeof(float)));

        const float table_base = static_cast<float>(a.tile_table_offset);
        const Reg<Scalar> first = k.load(table_addr, table_base + 0.0f);
        const Reg<Scalar> count = k.load(table_addr, table_base + 4.0f);

        // The running best, as in the untiled kernel: one thread owns one pixel,
        // so nothing is shared and no atomic is needed. Started beyond the far
        // plane so the first covering triangle takes it.
        const Reg<Scalar> best_z = k.constant(2.0f);
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // An empty tile has to skip the walk outright. The loop tests its
        // counter at the bottom, so without this guard it would read one
        // triangle's worth of whatever follows the tile's run — and empty tiles
        // are most of the screen, which is both where the saving comes from and
        // where the corruption would be worst.
        k.if_(k.gt(count, zero), [&] {
            const Reg<Scalar> stride =
                k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
            const Reg<Scalar> tri_addr =
                k.add(k.mul(first, stride),
                      k.constant(static_cast<float>(a.tile_vertices_offset)));
            const Reg<Scalar> i = k.constant(0.0f);

            const Label top = k.label();
            k.place(top);

            // Position then 1/w, three times — the layout pass 1 writes and
            // bin_triangles copies.
            const Reg<Scalar> x0 = k.load(tri_addr, 0.0f);
            const Reg<Scalar> y0 = k.load(tri_addr, 4.0f);
            const Reg<Scalar> z0 = k.load(tri_addr, 8.0f);
            const Reg<Scalar> iw0 = k.load(tri_addr, 12.0f);
            const Reg<Scalar> x1 = k.load(tri_addr, 16.0f);
            const Reg<Scalar> y1 = k.load(tri_addr, 20.0f);
            const Reg<Scalar> z1 = k.load(tri_addr, 24.0f);
            const Reg<Scalar> iw1 = k.load(tri_addr, 28.0f);
            const Reg<Scalar> x2 = k.load(tri_addr, 32.0f);
            const Reg<Scalar> y2 = k.load(tri_addr, 36.0f);
            const Reg<Scalar> z2 = k.load(tri_addr, 40.0f);
            const Reg<Scalar> iw2 = k.load(tri_addr, 44.0f);

            const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
            const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
            const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

            const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
            const Reg<Scalar> inv_area = k.rcp(area);
            const Reg<Scalar> w0 = k.mul(e0, inv_area);
            const Reg<Scalar> w1 = k.mul(e1, inv_area);
            const Reg<Scalar> w2 = k.mul(e2, inv_area);

            const Reg<Scalar> inside =
                k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

            const Reg<Scalar> depth = k.mul(w0, z0);
            k.fma(depth, w1, z1);
            k.fma(depth, w2, z2);

            const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));
            emit_keep(k, a.predicated, take, best_z, best, depth, one,
                      [&](Reg<Vec3> dst) {
                          Fragment fragment;
                          emit_covered_pixel(k, a.shading, dst, fragment,
                                             emit_correct(k, w0, w1, w2, iw0, iw1, iw2),
                                             cx, cy, depth, zero);
                      });

            k.fma(tri_addr, stride, one);
            k.fma(i, one, one);
            k.branch_to(top, k.lt(i, count));
        });

        // One pixel, once, after the whole scene has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });

    return k.build();
}

void run_tiled_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_tiled_raster_stage: an image with no pixels");
    }
    if (args.shading.mode == ShadingMode::Diffuse) {
        throw std::runtime_error(
            "run_tiled_raster_stage: a tile carries screen positions only, so this "
            "route has no world position to light — the walk and the tracer do");
    }
    if (args.shading.mode == ShadingMode::Custom && !args.shading.shade) {
        throw std::runtime_error(
            "run_tiled_raster_stage: Custom shading with nothing to emit — set "
            "Shading::shade");
    }

    // One block per tile, which is what makes blockIdx the tile index. A block
    // is 256 threads, so eight warps, each covering one row of the tile.
    // Nothing is shared between them — the ISA has no barrier — so they are
    // independent beyond reading the same triangle list.
    const dim3 block{TILE_WIDTH, TILE_HEIGHT, 1};
    const dim3 grid{args.tiles_x, (args.height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1};

    void* raw[] = {const_cast<TiledRasterStageArgs*>(&args)};
    rt.myrt_launch(build_tiled_raster_program, grid, block, raw);
}

// ---------------------------------------------------------------------------
// Tiling, through shared memory
//
// Same frame, same walk. The tile's triangles are staged once per block rather
// than read from global by every pixel — the first kernel here that has two
// warps depend on each other, and so the first that needs BARRIER.
// ---------------------------------------------------------------------------

namespace {

// The walk, emitted twice: once over a whole tile in shared memory, once over a
// chunk of one. Same instructions either way — what differs is where the run
// starts and how many triangles are in it.
//
// One function rather than two copies because the two forms have to stay
// identical: a fix landing in one of them would leave the double-buffered route
// drawing a different frame from the one-shot route, and the test that compares
// them pixel for pixel is the reason this exists.
void emit_shared_walk(IRBuilder& k, const TiledRasterStageArgs& a,
                      Reg<Scalar> shared_base, Reg<Scalar> triangles, Reg<Scalar> cx,
                      Reg<Scalar> cy, Reg<Scalar> best_z, Reg<Vec3> best,
                      Reg<Scalar> zero, Reg<Scalar> one)
{
    k.if_(k.gt(triangles, zero), [&] {
        const Reg<Scalar> shared_stride =
            k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
        const Reg<Scalar> shared_addr = k.copy(shared_base);
        const Reg<Scalar> i = k.copy(zero);

        const Label top = k.label();
        k.place(top);

        // Position then 1/w, three times — the layout pass 1 writes and
        // bin_triangles copies.
        const Reg<Scalar> x0 = k.load_shared(shared_addr, 0.0f);
        const Reg<Scalar> y0 = k.load_shared(shared_addr, 4.0f);
        const Reg<Scalar> z0 = k.load_shared(shared_addr, 8.0f);
        const Reg<Scalar> iw0 = k.load_shared(shared_addr, 12.0f);
        const Reg<Scalar> x1 = k.load_shared(shared_addr, 16.0f);
        const Reg<Scalar> y1 = k.load_shared(shared_addr, 20.0f);
        const Reg<Scalar> z1 = k.load_shared(shared_addr, 24.0f);
        const Reg<Scalar> iw1 = k.load_shared(shared_addr, 28.0f);
        const Reg<Scalar> x2 = k.load_shared(shared_addr, 32.0f);
        const Reg<Scalar> y2 = k.load_shared(shared_addr, 36.0f);
        const Reg<Scalar> z2 = k.load_shared(shared_addr, 40.0f);
        const Reg<Scalar> iw2 = k.load_shared(shared_addr, 44.0f);

        const Reg<Scalar> e0 = emit_edge(k, x1, y1, x2, y2, cx, cy);
        const Reg<Scalar> e1 = emit_edge(k, x2, y2, x0, y0, cx, cy);
        const Reg<Scalar> e2 = emit_edge(k, x0, y0, x1, y1, cx, cy);

        const Reg<Scalar> area = k.add(k.add(e0, e1), e2);
        const Reg<Scalar> inv_area = k.rcp(area);
        const Reg<Scalar> w0 = k.mul(e0, inv_area);
        const Reg<Scalar> w1 = k.mul(e1, inv_area);
        const Reg<Scalar> w2 = k.mul(e2, inv_area);

        const Reg<Scalar> inside =
            k.min(k.min(k.ge(w0, zero), k.ge(w1, zero)), k.ge(w2, zero));

        const Reg<Scalar> depth = k.mul(w0, z0);
        k.fma(depth, w1, z1);
        k.fma(depth, w2, z2);

        const Reg<Scalar> take = k.min(inside, k.lt(depth, best_z));
        emit_keep(k, a.predicated, take, best_z, best, depth, one, [&](Reg<Vec3> dst) {
            Fragment fragment;
            emit_covered_pixel(k, a.shading, dst, fragment,
                               emit_correct(k, w0, w1, w2, iw0, iw1, iw2), cx, cy, depth,
                               zero);
        });

        k.fma(shared_addr, shared_stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, triangles));
    });
}

// One chunk's worth of the tile, handed to memory and not waited for.
//
// Unrolled rather than looped, and that is what makes the double buffering
// expressible: S_CP_ASYNC_WAIT takes a count, so the kernel has to know how many
// copies a chunk is. A rotated loop issues a number that depends on the tile,
// and there is no instruction that waits for "whatever the last chunk was".
//
// The source is clamped to the tile's last float, so the final chunk restages
// something already there rather than reading past the run. The walk visits only
// the triangles that exist, so what lands in the tail slots is never read — and
// a clamped address is a line the cache already holds.
void emit_chunk_fill(IRBuilder& k, Reg<Scalar> buffer_base, Reg<Scalar> tri_addr,
                     Reg<Scalar> first_float, Reg<Scalar> last_float, Reg<Scalar> lane)
{
    const Reg<Scalar> four = k.constant(4.0f);
    for (uint32_t step = 0; step < CHUNK_COPIES_A_WARP; ++step) {
        const Reg<Scalar> slot =
            k.add(lane, k.constant(static_cast<float>(step * TILE_BLOCK_THREADS)));
        const Reg<Scalar> to = k.add(buffer_base, k.mul(slot, four));
        const Reg<Scalar> from = k.min(k.add(first_float, slot), last_float);
        k.cp_async(to, k.add(tri_addr, k.mul(from, four)));
    }
}

}  // namespace

Program build_shared_raster_program(void** args)
{
    const TiledRasterStageArgs& a = *static_cast<const TiledRasterStageArgs*>(args[0]);
    IRBuilder k;

    // Which tile, and where its run starts — unchanged from
    // build_tiled_raster_program.
    const Reg<Scalar> tile =
        k.add(k.mul(k.block_y(), k.constant(static_cast<float>(a.tiles_x))), k.block_x());
    const Reg<Scalar> table_addr = k.mul(tile, k.constant(2.0f * sizeof(float)));

    const float table_base = static_cast<float>(a.tile_table_offset);
    const Reg<Scalar> first = k.load(table_addr, table_base + 0.0f);
    const Reg<Scalar> count = k.load(table_addr, table_base + 4.0f);

    const Reg<Scalar> stride = k.constant(static_cast<float>(3 * SCREEN_VERTEX_BYTES));
    const Reg<Scalar> tri_addr = k.add(
        k.mul(first, stride), k.constant(static_cast<float>(a.tile_vertices_offset)));
    // thread_x and thread_y are global coordinates, so the block's own origin
    // has to come off before they can index anything the block owns. Both are
    // kept: the pixel work below still wants them as they are.
    const Reg<Scalar> tile_w = k.constant(static_cast<float>(TILE_WIDTH));
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> tx = k.sub(px, k.mul(k.block_x(), tile_w));
    const Reg<Scalar> ty =
        k.sub(py, k.mul(k.block_y(), k.constant(static_cast<float>(TILE_HEIGHT))));

    // 0 .. 255, and the stride the cooperative fill steps by.
    const Reg<Scalar> lane = k.add(k.mul(ty, tile_w), tx);

    const Reg<Scalar> one = k.constant(1.0f);
    const Reg<Scalar> zero = k.constant(0.0f);
    const Reg<Scalar> half = k.constant(0.5f);
    const Reg<Scalar> cx = k.add(px, half);
    const Reg<Scalar> cy = k.add(py, half);

    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    // What the walk keeps. Declared before the staging because the chunked form
    // walks several times and has to carry its answer between the passes.
    const Reg<Scalar> best_z = k.constant(2.0f);
    const Reg<Vec3> best = k.vec3();
    k.set(best.component(0), 0.0f);
    k.set(best.component(1), 0.0f);
    k.set(best.component(2), 0.0f);

    if (a.staging == TileStaging::Synchronous) {
        // The whole tile, once, through a register a float.
        const Reg<Scalar> four = k.constant(4.0f);
        const Reg<Scalar> block_threads =
            k.constant(static_cast<float>(TILE_BLOCK_THREADS));
        const Reg<Scalar> staged =
            k.mul(count, k.constant(static_cast<float>(TILE_TRIANGLE_FLOATS)));

        // A copy, because fma advances the cursor in place and a loop counter
        // that doubles as the thread's identity reads badly.
        const Reg<Scalar> cursor = k.copy(lane);

        // Rotated rather than wrapped in if_: a lane with nothing to stage leaves
        // before the body instead of entering it against a guard, and the rest
        // drop out one at a time as the cursor passes the end. min-PC keeps
        // issuing the body for whoever is left, and they all meet again at
        // fill_done.
        const Label fill_done = k.label();
        const Label fill_top = k.label();

        k.branch_to(fill_done, k.ge(cursor, staged));
        k.place(fill_top);

        const Reg<Scalar> byte = k.mul(cursor, four);
        k.store_shared(byte, k.load(k.add(tri_addr, byte), 0.0f), 0.0f);

        k.fma(cursor, block_threads, one);
        k.branch_to(fill_top, k.lt(cursor, staged));
        k.place(fill_done);

        // Everything above runs for every thread of the block, including the
        // ones whose pixel is off screen in an edge tile. A thread that branched
        // past this would be one the rest wait for and never see, and the
        // scheduler refuses that rather than hanging.
        k.barrier();

        k.if_(in_image, [&] {
            emit_shared_walk(k, a, zero, count, cx, cy, best_z, best, zero, one);
        });
    } else {
        // Two buffers, and the tile crossed a chunk at a time. The copies for the
        // next chunk are issued before this one is walked, so the fetch and the
        // walk overlap — which is what cp.async is for, and what the one-shot
        // form has no room to show: it fills once and is read by every thread.
        //
        // It is also what lifts the limit the synchronous route refuses at. A
        // tile larger than shared memory is a tile that takes more chunks.
        const Reg<Scalar> chunk_triangles =
            k.constant(static_cast<float>(CHUNK_TRIANGLES));
        const Reg<Scalar> chunk_floats =
            k.constant(static_cast<float>(CHUNK_TRIANGLES * TILE_TRIANGLE_FLOATS));
        const Reg<Scalar> buffer_bytes =
            k.constant(static_cast<float>(CHUNK_TRIANGLES * 3 * SCREEN_VERTEX_BYTES));

        // The float the tail is clamped to. Floored at zero for the empty tile:
        // the first chunk is issued before the loop can test anything, so a tile
        // with no triangles would otherwise clamp to -1 and ask for a negative
        // address. It stages the first float of the run and walks none of it.
        const Reg<Scalar> last_float = k.max(
            zero,
            k.sub(k.mul(count, k.constant(static_cast<float>(TILE_TRIANGLE_FLOATS))),
                  one));

        const Reg<Scalar> base = k.copy(zero);         // first triangle of this chunk
        const Reg<Scalar> buffer = k.copy(zero);       // which buffer, as bytes
        const Reg<Scalar> first_float = k.copy(zero);  // where this chunk starts

        emit_chunk_fill(k, buffer, tri_addr, first_float, last_float, lane);

        const Label chunk_top = k.label();
        k.place(chunk_top);

        // The next chunk goes into the other buffer, and is issued before this
        // one is waited for. Both arms wait, and both are taken by the whole
        // block: count is the same number in every thread, so this branch is
        // uniform and the barriers below are reached by everyone.
        const Reg<Scalar> next_base = k.add(base, chunk_triangles);
        const Reg<Scalar> other = k.sub(buffer_bytes, buffer);
        const Reg<Scalar> next_first = k.add(first_float, chunk_floats);
        k.if_else(
            k.lt(next_base, count),
            [&] {
                emit_chunk_fill(k, other, tri_addr, next_first, last_float, lane);
                // Leaves exactly the copies just issued outstanding, which is the
                // count this chunking exists to make knowable.
                k.cp_async_wait(CHUNK_COPIES_A_WARP);
            },
            [&] { k.cp_async_wait(0); });

        // This chunk has landed, for every warp of the block.
        k.barrier();

        const Reg<Scalar> remaining = k.sub(count, base);
        const Reg<Scalar> triangles = k.min(chunk_triangles, remaining);
        k.if_(in_image, [&] {
            emit_shared_walk(k, a, buffer, triangles, cx, cy, best_z, best, zero, one);
        });

        // Before the buffer this chunk used is filled again, two iterations from
        // here. Without it a fast warp would refill what a slow one is reading.
        k.barrier();

        k.copy_into(buffer, other);
        k.copy_into(base, next_base);
        k.copy_into(first_float, next_first);
        k.branch_to(chunk_top, k.lt(base, count));
    }

    k.if_(in_image, [&] {
        // One pixel, once, after the whole tile has been walked.
        const Reg<Scalar> row = k.mul(py, k.constant(static_cast<float>(a.width)));
        const Reg<Scalar> index = k.add(row, px);
        const Reg<Scalar> addr =
            k.mul(index, k.constant(static_cast<float>(PIXEL_BYTES)));

        const float frame_base = static_cast<float>(a.framebuffer_offset);
        k.store(addr, best.component(0), frame_base + 0.0f);
        k.store(addr, best.component(1), frame_base + 4.0f);
        k.store(addr, best.component(2), frame_base + 8.0f);
    });

    return k.build();
}

void run_shared_raster_stage(MyGPURuntime& rt, const TiledRasterStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_shared_raster_stage: an image with no pixels");
    }
    if (args.shading.mode == ShadingMode::Diffuse) {
        throw std::runtime_error(
            "run_shared_raster_stage: a tile carries screen positions only, so this "
            "route has no world position to light — the walk and the tracer do");
    }
    if (args.shading.mode == ShadingMode::Custom && !args.shading.shade) {
        throw std::runtime_error(
            "run_shared_raster_stage: Custom shading with nothing to emit — set "
            "Shading::shade");
    }

    // Real hardware splits an overfull tile across passes, and so does the
    // double-buffered form: a tile larger than shared memory is more chunks. The
    // synchronous one stages the tile in a single pass and has nowhere to put the
    // rest, so it still refuses.
    if (args.staging == TileStaging::Synchronous &&
        args.max_tile_triangles > SHARED_TRIANGLE_CAPACITY) {
        throw std::runtime_error("run_shared_raster_stage: a tile holds " +
                                 std::to_string(args.max_tile_triangles) +
                                 " triangles, and shared memory " + "stages " +
                                 std::to_string(SHARED_TRIANGLE_CAPACITY) +
                                 " — TileStaging::AsyncDoubleBuffered has no such "
                                 "limit, taking the tile a chunk at a time");
    }

    const dim3 block{TILE_WIDTH, TILE_HEIGHT, 1};
    const dim3 grid{args.tiles_x, (args.height + TILE_HEIGHT - 1) / TILE_HEIGHT, 1};

    // What this route stages, declared so that residency can charge it for it.
    // The synchronous form fills the whole scratchpad, which is what makes it the
    // one kernel here whose occupancy shared memory decides — the trade the flat
    // cost model could not express and this one can. The chunked form declares its
    // two buffers and no more, so it can hold several blocks an SM.
    void* raw[] = {const_cast<TiledRasterStageArgs*>(&args)};
    const size_t declared =
        args.staging == TileStaging::Synchronous
            ? SHARED_MEM_FLOATS * sizeof(float)
            : 2 * CHUNK_TRIANGLES * TILE_TRIANGLE_FLOATS * sizeof(float);
    rt.myrt_launch(build_shared_raster_program, LaunchConfig{grid, block, declared}, raw);
}
