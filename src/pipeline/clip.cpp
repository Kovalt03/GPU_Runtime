#include <stdexcept>
#include <string>

#include "ir_builder.hpp"
#include "pipeline/clip.hpp"
#include "pipeline/types.hpp"

namespace {

// Registers a clipped triangle at the slot the atomic handed out. Nine floats,
// three vertices of three components, in the layout every other pass reads.
void emit_write(IRBuilder& k, Reg<Scalar> base, Reg<Scalar> slot, Reg<Vec3> a,
                Reg<Vec3> b, Reg<Vec3> c)
{
    const Reg<Scalar> at =
        k.add(base, k.mul(slot, k.constant(3.0f * WORLD_VERTEX_BYTES)));
    k.store_vec3(at, a, 0.0f);
    k.store_vec3(at, b, static_cast<float>(WORLD_VERTEX_BYTES));
    k.store_vec3(at, c, static_cast<float>(2 * WORLD_VERTEX_BYTES));
}

// Where the segment from `in` to `out` crosses the plane.
//
// t = s_in / (s_in - s_out), which is between 0 and 1 exactly when the two ends
// are on opposite sides — and they are, or this would not be called.
Reg<Vec3> emit_crossing(IRBuilder& k, Reg<Vec3> in, Reg<Scalar> s_in, Reg<Vec3> out,
                        Reg<Scalar> s_out)
{
    const Reg<Scalar> t = k.mul(s_in, k.rcp(k.sub(s_in, s_out)));
    return k.add(in, k.scale(k.sub(out, in), t));
}

}  // namespace

Float4 near_plane_of(const Camera& camera)
{
    // Facing the way the camera looks, and sitting near_z in front of the eye:
    // a point is in front of it when dot(n, p) + d >= 0.
    const Float3 forward = normalize(camera.target - camera.eye);
    const float d = -(dot(forward, camera.eye) + camera.near_z);
    return Float4{forward.x, forward.y, forward.z, d};
}

Program build_clip_program(void** args)
{
    const ClipStageArgs& a = *static_cast<const ClipStageArgs*>(args[0]);
    IRBuilder k;

    const Reg<Scalar> zero = k.constant(0.0f);
    const Reg<Scalar> one = k.constant(1.0f);
    const Reg<Scalar> id = k.thread_x();

    k.if_(k.lt(id, k.constant(static_cast<float>(a.triangle_count))), [&] {
        // The plane, from the constant window: one program, any camera.
        const Reg<Scalar> plane_at = k.const_base();
        const Reg<Vec3> normal = k.load_vec3(plane_at);
        const Reg<Scalar> plane_d = k.load_const(plane_at, 3.0f * sizeof(float));

        const Reg<Scalar> tri = k.mul(id, k.constant(3.0f * WORLD_VERTEX_BYTES));
        const Reg<Scalar> from =
            k.add(tri, k.constant(static_cast<float>(a.world_offset)));

        const Reg<Vec3> v0 = k.load_vec3(from, 0.0f);
        const Reg<Vec3> v1 = k.load_vec3(from, static_cast<float>(WORLD_VERTEX_BYTES));
        const Reg<Vec3> v2 =
            k.load_vec3(from, static_cast<float>(2 * WORLD_VERTEX_BYTES));

        // Signed distance to the plane. Positive is in front of it.
        const Reg<Scalar> s0 = k.add(k.dot(normal, v0), plane_d);
        const Reg<Scalar> s1 = k.add(k.dot(normal, v1), plane_d);
        const Reg<Scalar> s2 = k.add(k.dot(normal, v2), plane_d);

        const Reg<Scalar> f0 = k.ge(s0, zero);
        const Reg<Scalar> f1 = k.ge(s1, zero);
        const Reg<Scalar> f2 = k.ge(s2, zero);
        const Reg<Scalar> inside = k.add(k.add(f0, f1), f2);

        const Reg<Scalar> counter = k.constant(static_cast<float>(a.counter_offset));
        const Reg<Scalar> out_base = k.constant(static_cast<float>(a.output_offset));

        // The three edges cross the plane at three points, and every case below
        // uses two of them. Computed once, outside the branches: working them out
        // where they are used meant six copies of the arithmetic and about 170
        // registers of the 250 a thread has.
        //
        // An edge with both ends on the same side divides by zero here and
        // produces nothing usable. That is safe because the branch that would
        // read it is the one that is not taken — the values are computed
        // unconditionally and *written* only under a guard.
        const Reg<Vec3> x01 = emit_crossing(k, v0, s0, v1, s1);
        const Reg<Vec3> x12 = emit_crossing(k, v1, s1, v2, s2);
        const Reg<Vec3> x20 = emit_crossing(k, v2, s2, v0, s0);

        // Whole triangle in front: one slot, and the vertices unchanged.
        k.if_(k.ge(inside, k.constant(3.0f)),
              [&] { emit_write(k, out_base, k.atomic_add(counter, one), v0, v1, v2); });

        // One vertex in front: a smaller triangle, with the two crossings on the
        // edges that leave it. Three cases, one a vertex, each written out rather
        // than rotated into place — this ISA cannot index a register, so "the
        // inside one" has to be a branch rather than a lookup.
        const Reg<Scalar> exactly_one = k.compare(inside, one, CmpOp::EQ);
        const auto one_inside = [&](Reg<Scalar> flag, Reg<Vec3> in, Reg<Vec3> leaving,
                                    Reg<Vec3> entering) {
            k.if_(k.min(exactly_one, flag), [&] {
                emit_write(k, out_base, k.atomic_add(counter, one), in, leaving,
                           entering);
            });
        };
        one_inside(f0, v0, x01, x20);
        one_inside(f1, v1, x12, x01);
        one_inside(f2, v2, x20, x12);

        // Two in front: a quadrilateral, which is two triangles. The winding is
        // kept — the raster stage decides coverage from the sign of the area, and
        // a flipped triangle would vanish rather than draw wrongly.
        const Reg<Scalar> exactly_two = k.compare(inside, k.constant(2.0f), CmpOp::EQ);
        const auto two_inside = [&](Reg<Scalar> behind, Reg<Vec3> p, Reg<Vec3> q,
                                    Reg<Vec3> leaving, Reg<Vec3> entering) {
            k.if_(k.min(exactly_two, behind), [&] {
                const Reg<Scalar> slot = k.atomic_add(counter, k.constant(2.0f));
                emit_write(k, out_base, slot, p, q, leaving);
                emit_write(k, out_base, k.add(slot, one), p, leaving, entering);
            });
        };
        // The flag names the vertex that is behind, and the two crossings are the
        // ones on the edges that reach it.
        two_inside(k.sub(one, f2), v0, v1, x12, x20);
        two_inside(k.sub(one, f0), v1, v2, x20, x01);
        two_inside(k.sub(one, f1), v2, v0, x01, x12);

        // Nothing in front writes nothing, which is the case with no branch of
        // its own: every guard above fails.
    });

    return k.build();
}

void run_clip_stage(MyGPURuntime& rt, const ClipStageArgs& args)
{
    if (args.triangle_count == 0) {
        return;
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.triangle_count + WARP_SIZE - 1) / WARP_SIZE, 1, 1};

    LaunchConfig config{grid, block};
    config.const_offset = args.plane_offset;

    void* raw[] = {const_cast<ClipStageArgs*>(&args)};
    rt.myrt_launch(build_clip_program, config, raw);
}
