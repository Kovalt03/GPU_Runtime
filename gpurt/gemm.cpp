#include <stdexcept>
#include <string>

#include "gemm.hpp"
#include "half.hpp"
#include "ir_builder.hpp"

namespace {

// Where a lane sits inside its fragment, and which warp it is in. All three are
// coordinates rather than divisions of a flat index — the block is shaped 2 x 16
// x 4 so that they can be.
struct Position {
    Reg<Scalar> half;  // 0 or 1: which eight columns of the row this lane holds
    Reg<Scalar> row;   // 0 .. 15
    Reg<Scalar> warp;  // 0 .. 3
};

Position lane_position(IRBuilder& k)
{
    const Reg<Scalar> half = k.sub(k.thread_x(), k.mul(k.block_x(), k.constant(2.0f)));
    const Reg<Scalar> row = k.sub(
        k.thread_y(), k.mul(k.block_y(), k.constant(static_cast<float>(GEMM_TILE))));
    const Reg<Scalar> warp = k.sub(
        k.thread_z(), k.mul(k.block_z(), k.constant(static_cast<float>(GEMM_WARPS))));
    return Position{half, row, warp};
}

// The ten floats a thread stages, as (shared byte, global byte) pairs built from
// the coordinates. Two of A and eight of B, so that every warp issues the same
// number of copies — S_CP_ASYNC_WAIT takes a count, and a count that varied by
// warp could not be written into the program.
//
// A tile element (row, col) is A[(m0 + row) * k + k0 + col]; the four B tiles
// together are one 16 x 64 slab, element (row, col) = B[(k0 + row) * n + n0 + col].
// How wide an element is. Halves are packed two to a word, so a tile is half
// the bytes and the staging moves half as many words — every address below is the
// same expression scaled, which is what makes the narrow path a width rather than
// a second kernel.
float element_bytes(const GemmArgs& a)
{
    return a.half_inputs ? 2.0f : 4.0f;
}

// Words a thread copies of each tile. Two elements of A and eight of B either
// way; at half precision those are one word and four.
uint32_t a_words(const GemmArgs& a)
{
    return a.half_inputs ? 1u : 2u;
}

uint32_t b_words(const GemmArgs& a)
{
    return a.half_inputs ? 4u : 8u;
}

void emit_staging(IRBuilder& k, const GemmArgs& a, const Position& at, Reg<Scalar> buffer,
                  Reg<Scalar> k0, Reg<Scalar> m0, Reg<Scalar> n0, bool asynchronous)
{
    const float wide = element_bytes(a);
    const Reg<Scalar> four = k.constant(4.0f);
    const Reg<Scalar> one = k.constant(1.0f);

    // Addresses are advanced in place rather than rebuilt. Every arithmetic call
    // allocates a register, and this is emitted twice in the double-buffered form
    // — the file holds 250 and the obvious spelling wanted about 120.
    const Reg<Scalar> to = k.copy(buffer);
    const Reg<Scalar> from = k.copy(buffer);

    // A: columns 4*warp + 2*half + {0, 1} of this thread's row.
    //
    // shared = buffer + (row * 16 + col) * wide
    // global = a_offset + ((m0 + row) * k + k0 + col) * wide
    k.copy_into(to, buffer);
    k.fma(to, at.row, k.constant(static_cast<float>(GEMM_TILE) * wide));
    k.fma(to, at.warp, k.constant(4.0f * wide));
    k.fma(to, at.half, k.constant(2.0f * wide));

    k.copy_into(from, k.constant(static_cast<float>(a.a_offset)));
    k.fma(from, k.add(m0, at.row), k.constant(static_cast<float>(a.k) * wide));
    k.fma(from, k0, k.constant(wide));
    k.fma(from, at.warp, k.constant(4.0f * wide));
    k.fma(from, at.half, k.constant(2.0f * wide));

    for (uint32_t i = 0; i < a_words(a); ++i) {
        if (asynchronous) {
            k.cp_async(to, from, static_cast<float>(i * sizeof(float)));
        } else {
            k.store_shared(to, k.load(from, static_cast<float>(i * sizeof(float))));
        }
        if (i + 1 < a_words(a)) {
            k.fma(to, one, four);
        }
    }

    // B: columns 16*warp + 8*half + {0 .. 7} of its row, in the slab after A.
    //
    // shared = buffer + a_elements * wide + (row * 64 + col) * wide
    // global = b_offset + ((k0 + row) * n + n0 + col) * wide
    k.copy_into(to, buffer);
    k.fma(to, one, k.constant(static_cast<float>(GEMM_A_FLOATS) * wide));
    k.fma(to, at.row, k.constant(static_cast<float>(GEMM_TILE_N) * wide));
    k.fma(to, at.warp, k.constant(static_cast<float>(GEMM_TILE) * wide));
    k.fma(to, at.half, k.constant(8.0f * wide));

    k.copy_into(from, k.constant(static_cast<float>(a.b_offset)));
    k.fma(from, k.add(k0, at.row), k.constant(static_cast<float>(a.n) * wide));
    k.fma(from, n0, k.constant(wide));
    k.fma(from, at.warp, k.constant(static_cast<float>(GEMM_TILE) * wide));
    k.fma(from, at.half, k.constant(8.0f * wide));

    for (uint32_t i = 0; i < b_words(a); ++i) {
        if (asynchronous) {
            k.cp_async(to, from, static_cast<float>(i * sizeof(float)));
        } else {
            k.store_shared(to, k.load(from, static_cast<float>(i * sizeof(float))));
        }
        if (i + 1 < b_words(a)) {
            k.fma(to, one, four);
        }
    }
}

}  // namespace

Program build_gemm_program(void** args)
{
    const GemmArgs& a = *static_cast<const GemmArgs*>(args[0]);
    IRBuilder k;

    const Position at = lane_position(k);
    const Reg<Scalar> four = k.constant(4.0f);
    const Reg<Scalar> one = k.constant(1.0f);
    const Reg<Scalar> zero = k.constant(0.0f);
    const Reg<Scalar> tile = k.constant(static_cast<float>(GEMM_TILE));

    // Where this block's strip of C starts.
    const Reg<Scalar> m0 = k.mul(k.block_y(), tile);
    const Reg<Scalar> n0 =
        k.mul(k.block_x(), k.constant(static_cast<float>(GEMM_TILE_N)));

    // The accumulator, held in registers for the whole K loop. Eight elements a
    // lane, which is a fragment — the matrix instruction reads and writes it in
    // place, and the arithmetic route walks the same eight.
    const Reg<Frag> accumulator = k.fragment();
    for (uint32_t i = 0; i < MMA_FRAGMENT_REGISTERS; ++i) {
        k.set(accumulator.component(i), 0.0f);
    }

    const float wide = element_bytes(a);
    const uint32_t copies_a_warp = a_words(a) + b_words(a);
    const Reg<Scalar> buffer_bytes =
        k.constant(static_cast<float>(GEMM_STAGE_FLOATS) * wide);
    const Reg<Scalar> buffer = k.copy(zero);
    const Reg<Scalar> step = k.copy(zero);
    const Reg<Scalar> steps = k.constant(static_cast<float>(a.k / GEMM_TILE));
    const Reg<Scalar> k0 = k.copy(zero);

    const bool async = a.staging == TileStaging::AsyncDoubleBuffered;
    if (async) {
        emit_staging(k, a, at, buffer, k0, m0, n0, true);
    }

    const Label top = k.label();
    k.place(top);

    const Reg<Scalar> next_k0 = k.add(k0, tile);
    const Reg<Scalar> other = k.sub(buffer_bytes, buffer);
    if (async) {
        // The next k-step is issued before this one is multiplied, so the fetch
        // and the arithmetic overlap. Both arms wait: the loop is uniform across
        // the block, every thread taking the same one.
        k.if_else(
            k.lt(next_k0, k.constant(static_cast<float>(a.k))),
            [&] {
                emit_staging(k, a, at, other, next_k0, m0, n0, true);
                k.cp_async_wait(copies_a_warp);
            },
            [&] { k.cp_async_wait(0); });
    } else {
        emit_staging(k, a, at, buffer, k0, m0, n0, false);
    }
    k.barrier();

    // The fragments this lane holds: row `row`, columns 8*half + {0 .. 7}.
    // Scaled by the element width, like the staging above: at half precision the
    // same element sits at half the byte offset, and a fragment is half as far
    // from the one beside it.
    const Reg<Scalar> width = k.constant(wide);
    const Reg<Scalar> a_at =
        k.add(buffer,
              k.mul(k.add(k.mul(at.row, tile), k.mul(at.half, k.constant(8.0f))), width));
    const Reg<Scalar> b_at =
        k.add(k.add(buffer, k.constant(static_cast<float>(GEMM_A_FLOATS) * wide)),
              k.mul(k.add(k.mul(at.row, k.constant(static_cast<float>(GEMM_TILE_N))),
                          k.add(k.mul(at.warp, tile), k.mul(at.half, k.constant(8.0f)))),
                    width));

    if (a.matrix_unit) {
        // Sixteen loads and one instruction, or two loads and one instruction.
        // What the wide one saves is the waiting: this machine issues in order,
        // so eight round trips to shared memory are eight waits with nothing
        // between them.
        if (a.half_inputs) {
            // Four registers an operand instead of eight, and an instruction
            // priced at half. The accumulator stays single precision, which is
            // the arrangement every tensor core makes.
            k.mma(accumulator, k.load_shared_half_fragment(a_at),
                  k.load_shared_half_fragment(b_at));
        } else if (a.wide_fragments) {
            k.mma(accumulator, k.load_shared_fragment(a_at),
                  k.load_shared_fragment(b_at));
        } else {
            const Reg<Frag> a_frag = k.fragment();
            const Reg<Frag> b_frag = k.fragment();
            for (uint32_t i = 0; i < MMA_FRAGMENT_REGISTERS; ++i) {
                k.load_shared_into(a_frag.component(i), a_at,
                                   static_cast<float>(i * sizeof(float)));
                k.load_shared_into(b_frag.component(i), b_at,
                                   static_cast<float>(i * sizeof(float)));
            }
            k.mma(accumulator, a_frag, b_frag);
        }
    } else {
        // The same product one multiply-add at a time. A lane's eight outputs
        // each need a whole row of A against a column of B, so the inner loop is
        // over k rather than unrolled: emitting 144 loads would allocate 144
        // registers, and the file holds 250.
        const Reg<Scalar> kk = k.copy(zero);
        const Reg<Scalar> a_row = k.add(buffer, k.mul(k.mul(at.row, tile), four));
        const Reg<Scalar> b_col = k.add(
            k.add(buffer, k.constant(static_cast<float>(GEMM_A_FLOATS * sizeof(float)))),
            k.mul(k.add(k.mul(at.warp, tile), k.mul(at.half, k.constant(8.0f))), four));

        const Label inner = k.label();
        k.place(inner);
        const Reg<Scalar> a_v = k.load_shared(k.add(a_row, k.mul(kk, four)));
        const Reg<Scalar> b_row = k.add(
            b_col, k.mul(k.mul(kk, k.constant(static_cast<float>(GEMM_TILE_N))), four));
        for (uint32_t i = 0; i < MMA_FRAGMENT_REGISTERS; ++i) {
            k.fma(accumulator.component(i),
                  k.load_shared(b_row, static_cast<float>(i * sizeof(float))), a_v);
        }
        k.fma(kk, one, one);
        k.branch_to(inner, k.lt(kk, tile));
    }

    // Before the buffer this step used is filled again.
    k.barrier();

    k.copy_into(buffer, other);
    k.copy_into(k0, next_k0);
    k.fma(step, one, one);
    k.branch_to(top, k.lt(step, steps));

    // C[(m0 + row) * n + n0 + 16*warp + 8*half + i]
    const Reg<Scalar> c_at = k.add(
        k.constant(static_cast<float>(a.c_offset)),
        k.mul(k.add(k.mul(k.add(m0, at.row), k.constant(static_cast<float>(a.n))),
                    k.add(n0,
                          k.add(k.mul(at.warp, tile), k.mul(at.half, k.constant(8.0f))))),
              four));
    for (uint32_t i = 0; i < MMA_FRAGMENT_REGISTERS; ++i) {
        k.store(c_at, accumulator.component(i), static_cast<float>(i * sizeof(float)));
    }

    return k.build();
}

void run_gemm(MyGPURuntime& rt, const GemmArgs& args)
{
    if (args.m % GEMM_TILE != 0 || args.k % GEMM_TILE != 0 || args.n % GEMM_TILE_N != 0) {
        throw std::runtime_error("run_gemm: m and k must be multiples of " +
                                 std::to_string(GEMM_TILE) + " and n a multiple of " +
                                 std::to_string(GEMM_TILE_N) + ", and this is " +
                                 std::to_string(args.m) + "x" + std::to_string(args.n) +
                                 "x" + std::to_string(args.k));
    }

    const dim3 block{2, GEMM_TILE, GEMM_WARPS};
    const dim3 grid{args.n / GEMM_TILE_N, args.m / GEMM_TILE, 1};

    LaunchConfig config{grid, block};
    config.shared_bytes = (args.staging == TileStaging::Synchronous ? 1 : 2) *
                          GEMM_STAGE_FLOATS * (args.half_inputs ? 2 : 4);

    void* raw[] = {const_cast<GemmArgs*>(&args)};
    rt.myrt_launch(build_gemm_program, config, raw);
}

std::vector<float> pack_halves(const std::vector<float>& values)
{
    std::vector<float> packed(values.size() / 2);
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = pack_f16x2(f32_to_f16(values[i * 2]), f32_to_f16(values[i * 2 + 1]));
    }
    return packed;
}

std::vector<float> rounded_to_half(const std::vector<float>& values)
{
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = f16_to_f32(f32_to_f16(values[i]));
    }
    return out;
}

std::vector<float> gemm_reference(const std::vector<float>& a,
                                  const std::vector<float>& b, uint32_t m, uint32_t n,
                                  uint32_t k)
{
    std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < k; ++i) {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    return c;
}
