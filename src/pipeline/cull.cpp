#include <cmath>
#include <stdexcept>
#include <string>

#include "ir_builder.hpp"
#include "pipeline/cull.hpp"
#include "thread.hpp"  // WARP_SIZE

namespace {

// The corner of a box furthest along a plane's normal.
//
// Testing that one is enough: if the most positive corner is behind the plane
// then every corner is, and if it is in front the box is not wholly outside.
// Eight corners would give the same answer for eight times the work.
Float3 furthest_along(const Float4& plane, const Box& box)
{
    return Float3{plane.x >= 0.0f ? box.hi.x : box.lo.x,
                  plane.y >= 0.0f ? box.hi.y : box.lo.y,
                  plane.z >= 0.0f ? box.hi.z : box.lo.z};
}

}  // namespace

Frustum frustum_of(const Float4x4& m)
{
    // Gribb-Hartmann: row 3 +/- row i. A clip coordinate is inside when
    // -w <= x <= w on each axis, and each of those six inequalities rearranges
    // into one of these planes.
    const auto row = [&](uint32_t r) {
        return Float4{m.at(r, 0), m.at(r, 1), m.at(r, 2), m.at(r, 3)};
    };
    const auto combine = [](const Float4& a, const Float4& b, float sign) {
        Float4 p{a.x + sign * b.x, a.y + sign * b.y, a.z + sign * b.z, a.w + sign * b.w};

        // Normalised so that the plane equation gives a distance rather than a
        // number proportional to one. Nothing here needs the distance, but a
        // caller comparing against a margin would, and an unnormalised plane
        // makes that silently wrong.
        const float length = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (length > 0.0f) {
            p = Float4{p.x / length, p.y / length, p.z / length, p.w / length};
        }
        return p;
    };

    const Float4 w = row(3);
    Frustum frustum;
    frustum.plane[0] = combine(w, row(0), 1.0f);   // left
    frustum.plane[1] = combine(w, row(0), -1.0f);  // right
    frustum.plane[2] = combine(w, row(1), 1.0f);   // bottom
    frustum.plane[3] = combine(w, row(1), -1.0f);  // top
    frustum.plane[4] = combine(w, row(2), 1.0f);   // near
    frustum.plane[5] = combine(w, row(2), -1.0f);  // far
    return frustum;
}

bool outside_frustum(const Frustum& frustum, const Box& box)
{
    for (const Float4& plane : frustum.plane) {
        const Float3 corner = furthest_along(plane, box);
        if (plane.x * corner.x + plane.y * corner.y + plane.z * corner.z + plane.w <
            0.0f) {
            return true;
        }
    }
    return false;
}

Program build_cull_program(void** args)
{
    const CullStageArgs& a = *static_cast<const CullStageArgs*>(args[0]);
    IRBuilder k;

    const Reg<Scalar> id = k.thread_x();
    const Reg<Scalar> live = k.lt(id, k.constant(static_cast<float>(a.instance_count)));

    k.if_(live, [&] {
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> one = k.constant(1.0f);

        const Reg<Scalar> box_addr =
            k.add(k.constant(static_cast<float>(a.boxes_offset)),
                  k.mul(id, k.constant(static_cast<float>(6 * sizeof(float)))));
        const Reg<Vec3> lo = k.load_vec3(box_addr, 0.0f);
        const Reg<Vec3> hi = k.load_vec3(box_addr, 12.0f);

        // Inside every plane, accumulated as a product of flags rather than as a
        // chain of branches. Six early exits would split the warp six times over
        // a test that is thirty instructions long — the trade emit_keep makes for
        // coverage, made the same way here.
        const Reg<Scalar> inside = k.constant(1.0f);
        for (uint32_t p = 0; p < 6; ++p) {
            const Float4& plane = a.frustum.plane[p];

            // The corner furthest along the normal, chosen without a branch: a
            // sign test yields 1.0 or 0.0, and that selects between lo and hi.
            const Reg<Scalar> distance = k.constant(plane.w);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float n = axis == 0 ? plane.x : (axis == 1 ? plane.y : plane.z);
                const Reg<Scalar> corner =
                    n >= 0.0f ? hi.component(axis) : lo.component(axis);
                k.fma(distance, corner, k.constant(n));
            }
            k.copy_into(inside, k.mul(inside, k.ge(distance, zero)));
        }

        // The survivors, each taking a slot nobody else got. grid.y is the
        // counter: the launch after this reads its grid from the same three
        // words, so there is nothing to publish afterwards.
        k.if_(inside, [&] {
            const Reg<Scalar> slot =
                k.atomic_add(k.constant(static_cast<float>(a.grid_offset)), one, 4.0f);

            const Reg<Scalar> from = k.add(
                k.constant(static_cast<float>(a.matrices_offset)),
                k.mul(id,
                      k.constant(static_cast<float>(MAT4_REGISTERS * sizeof(float)))));
            const Reg<Scalar> to = k.add(
                k.constant(static_cast<float>(a.survivors_offset)),
                k.mul(slot,
                      k.constant(static_cast<float>(MAT4_REGISTERS * sizeof(float)))));

            // The matrix itself rather than its index, so that the pass after
            // this one is the uninstanced kernel with a different address. An
            // index would save fifteen stores here and cost a load a vertex
            // there, which is the wrong way round: there are more vertices than
            // instances or this whole thing would not be worth doing.
            for (uint32_t i = 0; i < MAT4_REGISTERS; ++i) {
                const float at = static_cast<float>(i * sizeof(float));
                k.store(to, k.load(from, at), at);
            }
        });
    });

    return k.build();
}

void run_cull_stage(MyGPURuntime& rt, const CullStageArgs& args, StreamId stream)
{
    if (args.instance_count == 0) {
        throw std::runtime_error("run_cull_stage: nothing to cull");
    }
    if (args.grid_offset == 0 || args.survivors_offset == 0) {
        throw std::runtime_error(
            "run_cull_stage: nowhere to put the survivors or the grid they decide");
    }

    // One thread an instance. The pass this replaces walked them with one thread
    // and a loop, which is what stream_bench measured at 810 cycles of 900 —
    // there was nothing to cull then, so there was no reason to widen it.
    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.instance_count + WARP_SIZE - 1) / WARP_SIZE, 1, 1};

    void* raw[] = {const_cast<CullStageArgs*>(&args)};
    LaunchConfig config{grid, block};
    rt.myrt_launch_async(build_cull_program, config, raw, stream);
}
