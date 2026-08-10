#pragma once

#include "ir_builder.hpp"

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

// Writes the corrected weights into dst as (w1, w2, w0), the order the
// framebuffer takes. The device counterpart of perspective_correct, minus its
// guard against a zero denominator: inside a covered pixel the affine weights
// sum to one and every 1/w is positive, so the sum cannot reach zero.
inline void emit_shade(IRBuilder& k, Reg<Vec3> dst, Reg<Scalar> w0, Reg<Scalar> w1,
                       Reg<Scalar> w2, Reg<Scalar> iw0, Reg<Scalar> iw1, Reg<Scalar> iw2)
{
    const Reg<Scalar> a0 = k.mul(w0, iw0);
    const Reg<Scalar> a1 = k.mul(w1, iw1);
    const Reg<Scalar> a2 = k.mul(w2, iw2);
    const Reg<Scalar> inv_total = k.rcp(k.add(k.add(a0, a1), a2));

    k.copy_into(dst.component(0), k.mul(a1, inv_total));
    k.copy_into(dst.component(1), k.mul(a2, inv_total));
    k.copy_into(dst.component(2), k.mul(a0, inv_total));
}
