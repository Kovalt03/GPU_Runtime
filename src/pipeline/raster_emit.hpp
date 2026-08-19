#pragma once

#include "ir_builder.hpp"
#include "pipeline/types.hpp"

// Shared by the three raster kernels, which is the whole reason this is a
// header and not an anonymous namespace: build_raster_program, the tiled one
// and the shared-memory one emit the same coverage test and the same shade.
//
// Private to src/pipeline. IRBuilder is an implementation concern of this
// layer — no public header names it — so these do not belong in include/.

inline Reg<Scalar> emit_edge(IRBuilder& k, Reg<Scalar> ax, Reg<Scalar> ay, Reg<Scalar> bx,
                             Reg<Scalar> by, Reg<Scalar> px, Reg<Scalar> py)
{
    return k.sub(k.mul(k.sub(px, ax), k.sub(by, ay)),
                 k.mul(k.sub(py, ay), k.sub(bx, ax)));
}

// The affine weights, corrected for perspective. The device counterpart of
// perspective_correct, minus its guard against a zero denominator: inside a
// covered pixel the affine weights sum to one and every 1/w is positive, so the
// sum cannot reach zero.
//
// Named rather than inlined into the shade below because a caller's shader wants
// these as numbers where the built-in mode wants them as a colour. One
// arithmetic, so a varying and the debug colouring cannot disagree about what a
// weight is.
struct Corrected {
    Reg<Scalar> w0, w1, w2;
};

inline Corrected emit_correct(IRBuilder& k, Reg<Scalar> w0, Reg<Scalar> w1,
                              Reg<Scalar> w2, Reg<Scalar> iw0, Reg<Scalar> iw1,
                              Reg<Scalar> iw2)
{
    const Reg<Scalar> a0 = k.mul(w0, iw0);
    const Reg<Scalar> a1 = k.mul(w1, iw1);
    const Reg<Scalar> a2 = k.mul(w2, iw2);
    const Reg<Scalar> inv_total = k.rcp(k.add(k.add(a0, a1), a2));
    return {k.mul(a0, inv_total), k.mul(a1, inv_total), k.mul(a2, inv_total)};
}

// The two modes every raster route can serve: the debug colouring, and a
// caller's shader. Diffuse is not here because it needs buffers only the walk
// has — a world position and a face normal — so the walk tests for it first.
//
// The fragment comes in by reference already holding whatever varyings the route
// could interpolate, which is the one thing the routes differ on: the walk fills
// three, a tile has nowhere to keep one. Everything below it is the same on all
// three, and this is the single place that decides it.
inline void emit_covered_pixel(IRBuilder& k, const Shading& shading, Reg<Vec3> dst,
                               Fragment& fragment, const Corrected& c, Reg<Scalar> cx,
                               Reg<Scalar> cy, Reg<Scalar> depth)
{
    if (shading.mode != ShadingMode::Custom) {
        k.copy_into(dst.component(0), c.w1);
        k.copy_into(dst.component(1), c.w2);
        k.copy_into(dst.component(2), c.w0);
        return;
    }
    fragment.out = dst;
    fragment.x = cx;
    fragment.y = cy;
    fragment.depth = depth;
    fragment.w0 = c.w0;
    fragment.w1 = c.w1;
    fragment.w2 = c.w2;
    shading.shade(k, fragment);
}

// Keeping a triangle that covered the pixel and beat the running best — the one
// block every raster kernel ends its loop with, and the only place they differ
// in how they treat a warp whose lanes disagree.
//
// Branching, the lanes a triangle covers run the shade while the rest wait.
// Predicated, every lane shades and the result is kept arithmetically:
//
//   kept = take * new + (1 - take) * old
//
// take is exactly 1.0 or 0.0 out of V_CMP_F32, so one term survives whole and
// the other is multiplied away. No lane disagrees, and the coverage test stops
// contributing to the divergence rate — the shared kernel still reports a
// little, from the cooperative fill this has no bearing on.
//
// old + take * (new - old) is the same select an instruction cheaper, and is
// not exact: subtracting old and adding it back rounds, so a taken blend lands
// near new rather than on it. Sixty-four triangles of that drift changed the
// image, and a variant built to be compared against the branch has to agree
// with it to the bit.
//
// The branch wins by about 2% on every scene in render_bench, tiled and untiled
// alike, and predication's cost barely moves between them — every lane shading
// every triangle is the same work whatever the scene contains. Removing
// divergence and going faster turn out to be different things: what decides it
// is how much of the loop sits inside the branch, and here most of the cost
// (three edges, the weights, the depth) sits outside it.
template <typename Shade>
inline void emit_keep(IRBuilder& k, bool predicated, Reg<Scalar> take, Reg<Scalar> best_z,
                      Reg<Vec3> best, Reg<Scalar> depth, Reg<Scalar> one,
                      const Shade& shade)
{
    // shade is a callable rather than the six weights it used to take, because
    // what colours a pixel is no longer one expression: the walk can light what
    // it draws, and lighting reads buffers the tiled routes' tile lists do not
    // carry. The branch-or-blend decision is the same either way, which is what
    // this function is for.
    if (!predicated) {
        // Depth from the affine weights, colour from the corrected ones: NDC z
        // is linear in screen space and an attribute is not.
        k.if_(take, [&] {
            k.copy_into(best_z, depth);
            shade(best);
        });
        return;
    }

    // Into a scratch range rather than into best: the shade overwrites what it
    // is handed, and the blend below still needs the old value.
    const Reg<Vec3> shaded = k.vec3();
    shade(shaded);

    const Reg<Scalar> keep = k.sub(one, take);
    const auto blend = [&](Reg<Scalar> dst, Reg<Scalar> src) {
        k.copy_into(dst, k.mul(dst, keep));
        k.fma(dst, take, src);
    };
    blend(best_z, depth);
    blend(best.component(0), shaded.component(0));
    blend(best.component(1), shaded.component(1));
    blend(best.component(2), shaded.component(2));
}
