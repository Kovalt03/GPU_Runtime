#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ir_builder.hpp"
#include "math3d.hpp"
#include "pipeline/raytrace.hpp"
#include "thread.hpp"  // WARP_SIZE, the launch width

// ---------------------------------------------------------------------------
// Ray tracing
//
// The comparison the rasteriser exists to be measured against. Same scene, same
// image, and divergence from a different cause: the rasteriser splits a warp at
// a triangle's edge, the ray tracer wherever one lane's ray leaves early.
// ---------------------------------------------------------------------------

RayBasis ray_basis(const Camera& camera, float aspect)
{
    // Same handedness as look_at, and it has to be: the two cameras are
    // compared by rendering one scene through both.
    const Float3 forward = normalize(camera.target - camera.eye);
    const Float3 right = normalize(cross(forward, camera.up));
    const Float3 up = cross(right, forward);

    // Folding the field of view in here is what leaves the kernel with three
    // scales and two adds.
    const float half_extent = std::tan(radians(camera.fov_y_degrees) * 0.5f);

    return RayBasis{camera.eye, right * (half_extent * aspect), up * half_extent,
                    forward};
}

Program build_raytrace_program(void** args)
{
    const RaytraceStageArgs& a = *static_cast<const RaytraceStageArgs*>(args[0]);
    IRBuilder k;

    // Twelve moves, uniform across the launch — every lane ends up holding its
    // own copy of the same twelve numbers, as it does for the vertex stage's
    // matrix and for the tile table. The redundancy a scalar unit exists to
    // avoid, and the third place it turns up.
    const RayBasis& b = a.basis;
    const Reg<Vec3> world_origin = k.constant(b.origin.x, b.origin.y, b.origin.z);
    const Reg<Vec3> right = k.constant(b.right.x, b.right.y, b.right.z);
    const Reg<Vec3> up = k.constant(b.up.x, b.up.y, b.up.z);
    const Reg<Vec3> forward = k.constant(b.forward.x, b.forward.y, b.forward.z);
    const Reg<Scalar> px = k.thread_x();
    const Reg<Scalar> py = k.thread_y();
    const Reg<Scalar> in_image =
        k.min(k.lt(px, k.constant(static_cast<float>(a.width))),
              k.lt(py, k.constant(static_cast<float>(a.height))));

    k.if_(in_image, [&] {
        // The half-pixel offset and the scale fold into one constant each, as
        // apps/ray_triangle.cpp does: two multiplies and two adds rather
        // than six. sy runs backwards because rows count down and the frame up.
        const Reg<Scalar> sx =
            k.add(k.mul(px, k.constant(2.0f / static_cast<float>(a.width))),
                  k.constant(1.0f / static_cast<float>(a.width) - 1.0f));
        const Reg<Scalar> sy =
            k.add(k.mul(py, k.constant(-2.0f / static_cast<float>(a.height))),
                  k.constant(1.0f - 1.0f / static_cast<float>(a.height)));
        const Reg<Vec3> world_dir =
            k.add(k.scale(right, sx), k.add(k.scale(up, sy), forward));

        // What the lower level and the intersection test read. The same
        // registers as the world ray until there is an upper level to move out
        // of, and then the instance's own — written once at each of its leaves.
        const bool two_level = a.traversal == Traversal::Tlas;
        const Reg<Vec3> origin = two_level ? k.vec3() : world_origin;
        const Reg<Vec3> dir = two_level ? k.vec3() : world_dir;
        if (two_level) {
            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(origin.component(c), world_origin.component(c));
                k.copy_into(dir.component(c), world_dir.component(c));
            }
        }
        // The running best, as in the rasteriser. Infinity rather than the
        // rasteriser's 2.0f: that worked because NDC depth is bounded, and t is
        // a ray parameter with no ceiling.
        // Infinity is the natural ceiling for a ray parameter, and the blend
        // cannot use it: it multiplies the running best by (1 - take), and
        // inf * 0 is NaN. FLT_MAX is above every real t, so nothing that would
        // have won loses.
        const Reg<Scalar> best_t =
            k.constant(a.predicated ? std::numeric_limits<float>::max()
                                    : std::numeric_limits<float>::infinity());
        const Reg<Vec3> best = k.vec3();
        k.set(best.component(0), 0.0f);
        k.set(best.component(1), 0.0f);
        k.set(best.component(2), 0.0f);

        // What a deferred shade is worked out from. Zeroed rather than left
        // alone: a ray that meets nothing colours the background, and the shader
        // still runs on it.
        const bool deferred = a.shade_when != ShadeWhen::Inline;
        //
        // The barycentrics are the exception: a walk that bounces overwrites
        // `best` with what it accumulated and never reaches the shade below, so
        // they are neither read nor allocated there. Two registers, which is
        // what a shadow needs to fit.
        const bool keeps_bary = deferred && a.bounces == 1;
        const Reg<Scalar> best_u = keeps_bary ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Scalar> best_v = keeps_bary ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Scalar> best_material = k.constant(0.0f);

        // Whether anything was hit at all. Shading inline never needed one — a
        // ray that meets nothing simply never reaches the shade — but a deferred
        // one runs after the loop whatever happened, so without this the
        // background is coloured by whichever arm material zero selects.
        //
        // This is the distinction a miss shader draws in DXR, arrived at from
        // the other side: the flag has to exist before there is anywhere to hang
        // one.
        const Reg<Scalar> anything_hit = k.constant(0.0f);

        // What a bounce carries between turns. `accumulated` is the pixel so
        // far; `attenuation` is how much of the next surface still reaches it,
        // and a surface that reflects nothing sets it to zero.
        // Allocated only where they are read. A handful of registers is nothing
        // on a walk that fits and everything on one that does not — the
        // two-level traversal was already at the ceiling, and taking them
        // unconditionally put it over. Shadow rays found the same wall again.
        const bool bouncing = a.bounces > 1;
        const bool shadowing = bouncing && a.shadows;
        const Reg<Vec3> accumulated =
            bouncing ? k.constant(0.0f, 0.0f, 0.0f) : Reg<Vec3>{};
        const Reg<Vec3> attenuation =
            bouncing ? k.constant(1.0f, 1.0f, 1.0f) : Reg<Vec3>{};
        const Reg<Vec3> best_normal =
            bouncing ? k.constant(0.0f, 0.0f, 1.0f) : Reg<Vec3>{};
        const Reg<Scalar> bounce = bouncing ? k.constant(0.0f) : Reg<Scalar>{};
        // Only where a walk lights the hit it just found. One that asks about
        // the light keeps the term across a turn instead, in hit_lambert below.
        const Reg<Scalar> lambert =
            bouncing && !shadowing ? k.constant(0.0f) : Reg<Scalar>{};

        // With shadows the walk runs two rays a bounce: the one that carries
        // the picture, and the one that asks whether the light reaches where it
        // stopped. Both are the same traversal, so what separates them is a
        // register saying which turn this is — and that register holds the same
        // value in every lane of every warp, being the parity of a counter, so
        // branching on it splits nothing.
        //
        // What the surface turn found has to survive the shadow turn, which
        // overwrites everything the traversal writes. The lit term is kept
        // rather than the normal it came from: one register instead of three,
        // and nothing downstream wants the direction back.
        //
        // Declared here rather than beside their use, which is below the label
        // the loop returns to: an initial value is an instruction, and one
        // emitted inside the loop is one that runs every time round.
        const Reg<Scalar> shadow_turn = shadowing ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Scalar> hit_any = shadowing ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Scalar> hit_material = shadowing ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Scalar> hit_lambert = shadowing ? k.constant(0.0f) : Reg<Scalar>{};
        const Reg<Vec3> onward = shadowing ? k.constant(0.0f, 0.0f, 1.0f) : Reg<Vec3>{};
        const Label bounce_top = k.label();
        if (bouncing) {
            k.place(bounce_top);

            // Reset for this turn. The traversal below writes them as it goes
            // and a second turn has to start where the first did.
            k.copy_into(best_t, k.constant(std::numeric_limits<float>::max()));
            k.copy_into(anything_hit, k.constant(0.0f));
        }

        // What the triangle loop runs over. Linear, these are the whole scene and
        // never change; traversing a tree, a leaf writes both before entering.
        //
        // run_raytrace_stage refuses a count of zero, so the linear loop can test
        // at the bottom without a guard. A leaf cannot be empty either — the
        // builder only writes a leaf for triangles it holds.
        const Reg<Scalar> one = k.constant(1.0f);
        const Reg<Scalar> zero = k.constant(0.0f);
        const Reg<Scalar> eps = k.constant(INTERSECT_EPSILON);
        const Reg<Scalar> stride = k.constant(static_cast<float>(3 * WORLD_VERTEX_BYTES));
        const Reg<Scalar> count = k.constant(static_cast<float>(a.triangle_count));
        const Reg<Scalar> cursor = k.constant(static_cast<float>(a.triangles_offset));

        // Where this instance's lower-level tree and triangles begin.
        //
        // Baked constants until there was a second level: every instance then
        // walked the same tree at the same address, which made a hundred copies
        // of one mesh expressible and a cube beside a sphere not. Read from the
        // instance instead, and the traversal below is unchanged — an address is
        // an address whichever way it arrived.
        const Reg<Scalar> blas_nodes = k.constant(
            static_cast<float>(a.traversal == Traversal::Linear ? 0 : a.bvh_offset));
        const Reg<Scalar> blas_triangles =
            k.constant(static_cast<float>(a.triangles_offset));

        // What this instance is made of. Zero without a second level, and a
        // shader that never asks pays nothing for it either way.
        const Reg<Scalar> material = k.constant(0.0f);
        const Reg<Scalar> i = k.constant(0.0f);

        // Above the traversal, not merely above k.place(top): a leaf falls
        // through to the loop, so anything between the two is issued once a leaf
        // rather than once a launch. Emitted whatever the mode — six moves cost
        // less than the optional it would take to declare them conditionally.
        const Float3& lp = a.shading.light_position;
        const Float3& bc = a.shading.base_colour;
        const Reg<Vec3> light = k.constant(lp.x, lp.y, lp.z);
        // Unallocated where nothing reads it: a walk that bounces takes its
        // colour from the material table, and the shader that would have used
        // this never runs.
        const Reg<Vec3> base = bouncing ? Reg<Vec3>{} : k.constant(bc.x, bc.y, bc.z);

        // --- traversal ------------------------------------------------------
        // Emitted before the triangle loop so that the loop's bottom test can
        // branch back into it. Nothing here is issued when the mode is Linear.
        const bool bvh = a.traversal != Traversal::Linear;
        const Label traverse = k.label();
        const Label traverse_next = k.label();
        const Label leaf_found = k.label();
        const Label done = k.label();

        // The upper level's own loop and its own place to come back to. Where
        // the lower level goes when its stack empties is the only thing two
        // levels change about it: the next instance rather than the pixel.
        const Label tlas_traverse = k.label();
        const Label tlas_next = k.label();
        const Label tlas_leaf = k.label();
        const Label instance_top = k.label();
        const Label instance_next = k.label();
        const Label& blas_exit = two_level ? instance_next : done;

        // The stack lives in shared memory because it cannot live anywhere else:
        // an instruction names its registers in immediate fields, so there is no
        // way to index the register file with a running pointer. Hardware puts a
        // traversal stack in the same place for the same reason.
        //
        // The block is one warp wide, so a lane's slot is its x within the block.
        // A global thread_x cannot be reduced to it — there is no integer modulo
        // — which is what block_x is for.
        Reg<Scalar> stack_slot = zero;
        Reg<Scalar> sp = zero;
        Reg<Vec3> inv_dir = k.vec3();
        Reg<Vec3> world_inv = two_level ? k.vec3() : inv_dir;

        // Where the upper level's leaf left off. Declared out here because the
        // lower level's exit has to advance them, and that sits past the block
        // that fills them in.
        Reg<Scalar> inst_i = zero;
        Reg<Scalar> inst_count = zero;
        if (bvh) {
            // Which lane of the block this is, not of the warp: a block several
            // rows tall holds several warps, and they cannot share a stack.
            const Reg<Scalar> lane_x =
                k.sub(px, k.mul(k.block_x(), k.constant(static_cast<float>(WARP_SIZE))));
            const Reg<Scalar> lane =
                a.block_rows == 1
                    ? lane_x
                    : k.add(lane_x, k.mul(k.sub(py, k.mul(k.block_y(),
                                                          k.constant(static_cast<float>(
                                                              a.block_rows)))),
                                          k.constant(static_cast<float>(WARP_SIZE))));
            // Both levels: the upper one is still on the stack while the lower
            // is walked, so a lane's slice has to hold the sum. Sizing it by the
            // lower level alone put lane 0's BLAS stack on top of lane 1's TLAS
            // stack, which loses whole instances and leaves a frame that still
            // looks like one.
            const uint32_t slice = a.stack_depth + a.tlas_stack_depth;
            stack_slot =
                k.mul(lane, k.constant(static_cast<float>(slice * sizeof(float))));

            // Three reciprocals once rather than three divisions a node. A zero
            // component gives an infinite slab bound, which compares the way the
            // test wants: the ray never leaves that slab.
            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(world_inv.component(c), k.rcp(world_dir.component(c)));
                k.copy_into(inv_dir.component(c), world_inv.component(c));
            }

            sp = k.constant(0.0f);

            // The lower level's slice starts past the upper one's, which is
            // still occupied while this runs — the instance being walked has a
            // node of its own waiting on the stack above.
            const Reg<Scalar> blas_slot =
                two_level ? k.add(stack_slot, k.constant(static_cast<float>(
                                                  a.tlas_stack_depth * sizeof(float))))
                          : stack_slot;

            if (two_level) {
                // --- the upper level ----------------------------------------
                const Reg<Scalar> tlas_sp = k.constant(0.0f);
                k.store_shared(stack_slot, zero);
                k.fma(tlas_sp, one, one);

                k.place(tlas_traverse);
                k.fma(tlas_sp, one, k.constant(-1.0f));
                const Reg<Scalar> tnode =
                    k.load_shared(k.add(stack_slot, k.mul(tlas_sp, k.constant(4.0f))));
                const Reg<Scalar> tnode_addr =
                    k.add(k.constant(static_cast<float>(a.tlas_offset)),
                          k.mul(tnode, k.constant(static_cast<float>(BVH_NODE_BYTES))));

                // Tested against the world ray, this level being where the
                // instances sit rather than where their geometry does.
                const Reg<Scalar> tnear = k.constant(0.0f);
                const Reg<Scalar> tfar = k.constant(0.0f);
                const Reg<Scalar> tcount = k.constant(0.0f);
                {
                    // Everything here dies at the brace, which is what lets a
                    // second traversal fit at all.
                    IRBuilder::Scratch scope(k);
                    const Reg<Vec3> tlo = k.load_vec3(tnode_addr, 0.0f);
                    const Reg<Vec3> thi = k.load_vec3(tnode_addr, 12.0f);
                    Reg<Scalar> near_so_far = zero;
                    Reg<Scalar> far_so_far = best_t;
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        const Reg<Scalar> t0 = k.mul(
                            k.sub(tlo.component(axis), world_origin.component(axis)),
                            world_inv.component(axis));
                        const Reg<Scalar> t1 = k.mul(
                            k.sub(thi.component(axis), world_origin.component(axis)),
                            world_inv.component(axis));
                        near_so_far = k.max(near_so_far, k.min(t0, t1));
                        far_so_far = k.min(far_so_far, k.max(t0, t1));
                    }
                    k.copy_into(tnear, near_so_far);
                    k.copy_into(tfar, far_so_far);
                    k.copy_into(tcount, k.load(tnode_addr, 28.0f));
                }
                k.branch_to(tlas_next, k.gt(tnear, tfar));
                k.branch_to(tlas_leaf, k.gt(tcount, zero));

                const Reg<Scalar> tleft = k.load(tnode_addr, 24.0f);
                k.store_shared(k.add(stack_slot, k.mul(tlas_sp, k.constant(4.0f))),
                               k.add(tleft, one));
                k.fma(tlas_sp, one, one);
                k.store_shared(k.add(stack_slot, k.mul(tlas_sp, k.constant(4.0f))),
                               tleft);
                k.fma(tlas_sp, one, one);
                k.place(tlas_next);
                k.branch_to(tlas_traverse, k.gt(tlas_sp, zero));
                k.branch_to(done);

                // --- a leaf: one instance at a time --------------------------
                k.place(tlas_leaf);
                inst_i = k.constant(0.0f);
                inst_count = k.constant(0.0f);
                const Reg<Scalar> inst_first = k.constant(0.0f);
                k.copy_into(inst_count, tcount);
                k.copy_into(inst_first, k.load(tnode_addr, 24.0f));

                k.place(instance_top);
                const Reg<Scalar> inst_addr =
                    k.add(k.constant(static_cast<float>(a.instances_offset)),
                          k.mul(k.add(inst_first, inst_i),
                                k.constant(static_cast<float>(TLAS_INSTANCE_BYTES))));

                // Sixteen scalar loads, a lane at a time. The constant window is
                // no help here and the pricing says why: two lanes at the same
                // TLAS leaf may still be at different instances, so the address
                // is not warp-uniform. A wide global matrix load is the slot the
                // naming scheme reserves and does not have.
                IRBuilder::Scratch instance_scope(k);
                const Reg<Mat4> to_object = k.mat4();
                for (uint32_t i = 0; i < MAT4_REGISTERS; ++i) {
                    k.load_into(to_object.component(i), inst_addr,
                                static_cast<float>(i * sizeof(float)));
                }

                // The point moves with w = 1 and the direction with w = 0, which
                // is what leaves t meaning the same thing on both sides.
                const Reg<Vec4> op = k.vec4();
                const Reg<Vec4> od = k.vec4();
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(op.component(c), world_origin.component(c));
                    k.copy_into(od.component(c), world_dir.component(c));
                }
                k.set(op.component(3), 1.0f);
                k.set(od.component(3), 0.0f);

                const Reg<Vec4> moved_origin = k.transform(to_object, op);
                const Reg<Vec4> moved_dir = k.transform(to_object, od);
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(origin.component(c), moved_origin.component(c));
                    k.copy_into(dir.component(c), moved_dir.component(c));
                    k.copy_into(inv_dir.component(c), k.rcp(moved_dir.component(c)));
                }

                // The three past the matrix: this instance's own tree, its own
                // triangles, and what it is made of. Two loads to make a scene
                // out of what was a field of copies, and one for [m7].
                k.copy_into(blas_nodes, k.load(inst_addr, 64.0f));
                k.copy_into(blas_triangles, k.load(inst_addr, 68.0f));
                k.copy_into(material, k.load(inst_addr, 72.0f));

                // And into the lower level, which is the loop below unchanged.
                k.copy_into(sp, zero);
                k.store_shared(blas_slot, zero);
                k.fma(sp, one, one);
            } else {
                k.store_shared(stack_slot, zero);  // the root
                k.fma(sp, one, one);
            }

            k.place(traverse);
            k.fma(sp, one, k.constant(-1.0f));
            const Reg<Scalar> node =
                k.load_shared(k.add(blas_slot, k.mul(sp, k.constant(4.0f))));
            const Reg<Scalar> node_addr = k.add(
                blas_nodes, k.mul(node, k.constant(static_cast<float>(BVH_NODE_BYTES))));

            // The slab test. Two wide loads for six floats, and the same
            // arithmetic nodes_visited runs on the host — a box the two disagree
            // about would drop geometry from one and not the other.
            const Reg<Scalar> node_count = k.constant(0.0f);
            {
                IRBuilder::Scratch scope(k);
                const Reg<Vec3> lo = k.load_vec3(node_addr, 0.0f);
                const Reg<Vec3> hi = k.load_vec3(node_addr, 12.0f);

                // Clamped below at zero so a box behind the origin cannot be entered,
                // and above at the running best, which is the early exit an ordered
                // traversal would get for free: a node further away than a hit
                // already found has nothing to offer.
                Reg<Scalar> near = zero;
                Reg<Scalar> far = best_t;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const Reg<Scalar> t0 =
                        k.mul(k.sub(lo.component(axis), origin.component(axis)),
                              inv_dir.component(axis));
                    const Reg<Scalar> t1 =
                        k.mul(k.sub(hi.component(axis), origin.component(axis)),
                              inv_dir.component(axis));
                    near = k.max(near, k.min(t0, t1));
                    far = k.min(far, k.max(t0, t1));
                }
                k.copy_into(node_count, k.load(node_addr, 28.0f));
                k.branch_to(traverse_next, k.gt(near, far));
            }
            k.branch_to(leaf_found, k.gt(node_count, zero));

            // Interior: both children, which are adjacent. The second push pops
            // first, so whichever goes on last is the one entered next.
            const Reg<Scalar> left = k.load(node_addr, 24.0f);
            Reg<Scalar> first = left;
            Reg<Scalar> second = k.add(left, one);
            if (a.order == TraversalOrder::Nearest) {
                // Where each child's slab test starts, which is how near it can
                // possibly be. Four wide loads and two slab tests a node — the
                // price of the ordering, paid whether or not it prunes anything.
                const auto entry = [&](Reg<Scalar> index) {
                    const Reg<Scalar> addr = k.add(
                        blas_nodes,
                        k.mul(index, k.constant(static_cast<float>(BVH_NODE_BYTES))));
                    const Reg<Vec3> clo = k.load_vec3(addr, 0.0f);
                    const Reg<Vec3> chi = k.load_vec3(addr, 12.0f);
                    Reg<Scalar> entered = zero;
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        const Reg<Scalar> c0 =
                            k.mul(k.sub(clo.component(axis), origin.component(axis)),
                                  inv_dir.component(axis));
                        const Reg<Scalar> c1 =
                            k.mul(k.sub(chi.component(axis), origin.component(axis)),
                                  inv_dir.component(axis));
                        entered = k.max(entered, k.min(c0, c1));
                    }
                    return entered;
                };
                const Reg<Scalar> right = k.add(left, one);

                // 1.0 when the right child is nearer. Adding a flag to an index
                // picks a child without a branch, which matters here more than
                // anywhere: a warp diverging over which subtree to enter is the
                // cost this whole ordering exists to avoid.
                const Reg<Scalar> swap = k.lt(entry(right), entry(left));
                first = k.add(left, swap);
                second = k.sub(right, swap);
            }

            k.store_shared(k.add(blas_slot, k.mul(sp, k.constant(4.0f))), second);
            k.fma(sp, one, one);
            k.store_shared(k.add(blas_slot, k.mul(sp, k.constant(4.0f))), first);
            k.fma(sp, one, one);
            k.place(traverse_next);
            k.branch_to(traverse, k.gt(sp, zero));
            k.branch_to(blas_exit);

            // A leaf runs the loop below over its own range instead of the scene.
            k.place(leaf_found);
            k.copy_into(count, node_count);
            k.copy_into(cursor,
                        k.add(blas_triangles, k.mul(k.load(node_addr, 24.0f), stride)));
            k.copy_into(i, zero);
        }

        // Every miss lands on next, and it is placed before the advance below:
        // a miss means "on to the next triangle", not "out of the loop", so the
        // cursor still has to move or the lane meets the same one for ever.
        //
        // Predicated, nothing branches to next and nothing places it, which
        // build() allows — only a label branched to and never placed is an error.
        const Label top = k.label();
        const Label next = k.label();
        k.place(top);

        // A triangle is nine floats, so each vertex is one wide load. Every lane
        // of a warp is on the same triangle here, which makes the three of them
        // three transactions where the nine scalar loads this replaced were nine.
        const Reg<Vec3> v0 = k.load_vec3(cursor, 0.0f);
        const Reg<Vec3> v1 = k.load_vec3(cursor, 12.0f);
        const Reg<Vec3> v2 = k.load_vec3(cursor, 24.0f);

        // Möller-Trumbore, a line for each line of intersect().
        const Reg<Vec3> e1 = k.sub(v1, v0);
        const Reg<Vec3> e2 = k.sub(v2, v0);
        const Reg<Vec3> h = k.cross(dir, e2);
        const Reg<Scalar> a_dot = k.dot(e1, h);

        // Which material this triangle is made of.
        //
        // The two buffers are parallel — a triangle is nine floats and a
        // material index is one — so the offset into one scales onto the other
        // by a constant, and the loop's own cursor is enough. No triangle number
        // is kept anywhere and there is no integer division to recover one.
        const auto per_triangle_material = [&] {
            if (a.material_offset == 0) {
                return material;
            }
            const float scale =
                static_cast<float>(sizeof(float)) / (3 * WORLD_VERTEX_BYTES);
            return k.load(
                k.mul(k.sub(cursor, k.constant(static_cast<float>(a.triangles_offset))),
                      k.constant(scale)),
                static_cast<float>(a.material_offset));
        };

        // The four exits, each written twice — the condition to leave on, and
        // the condition to have survived. Emitted through one call so the pair
        // stays in view: they are not each other's negation once NaN is in play,
        // and a flag that disagreed with its branch would make the two kernels
        // incomparable rather than merely wrong.
        std::vector<Reg<Scalar>> survived;
        const auto gate = [&](auto passed, auto failed) {
            if (a.predicated) {
                survived.push_back(passed());
            } else {
                k.branch_to(next, failed());
            }
        };

        // a < eps rather than |a| < eps rejects a back face along with a
        // parallel ray, the ISA having no absolute value.
        //
        // First of the four exits. Six conditions, but any() folds each pair
        // into one flag, so a warp splits four times rather than six — and
        // min() of two flags is their AND, which is that same fold predicated.
        gate([&] { return k.ge(a_dot, eps); }, [&] { return k.lt(a_dot, eps); });

        // Branching, the exit above is what makes this reciprocal safe: a ray
        // parallel to the triangle has already left. Predicated nothing has, so
        // every lane divides, and rcp(0) is infinity — which reaches u, v and t
        // and then meets a blend that multiplies discarded terms by zero, where
        // 0 * inf is NaN. A surviving lane has a_dot >= eps and max() hands it
        // back untouched, which is what keeps the two forms bit-identical; a
        // rejected one gets 1/eps, wrong but finite.
        const Reg<Scalar> f = k.rcp(a.predicated ? k.max(a_dot, eps) : a_dot);
        const Reg<Vec3> s = k.sub(origin, v0);

        const Reg<Scalar> u = k.mul(f, k.dot(s, h));
        gate([&] { return k.min(k.ge(u, zero), k.le(u, one)); },
             [&] { return k.any(k.lt(u, zero), k.gt(u, one)); });

        const Reg<Vec3> q = k.cross(s, e1);
        const Reg<Scalar> v = k.mul(f, k.dot(dir, q));
        gate([&] { return k.min(k.ge(v, zero), k.le(k.add(u, v), one)); },
             [&] { return k.any(k.lt(v, zero), k.gt(k.add(u, v), one)); });

        const Reg<Scalar> t = k.mul(f, k.dot(e2, q));
        gate([&] { return k.ge(t, eps); }, [&] { return k.lt(t, eps); });

        // Nearest wins. No depth buffer and no atomics: the thread owns its
        // pixel, so the running best is a register.
        //
        // Read at build time, not by the device: a KernelFunc runs once per
        // launch, so only one shading arm reaches the instruction stream.
        //
        // Shaded inside the loop, so a pixel that meets three triangles shades
        // three times. Carrying a normal and a point out to the end would cost
        // six registers to save work the twelve loads above already dominate.
        //
        // e1 and e2 are the intersection test's, not rebuilt from the vertices:
        // a normal wound off different edges than a_dot was would cull one face
        // and shade the other. dir is not normalised and does not need to be —
        // t is measured in units of it, so the two cancel.
        const auto shade = [&](Reg<Vec3> dst) {
            if (a.shading.mode == ShadingMode::Custom) {
                // The same Fragment the raster routes hand over, with two of its
                // fields meaning what this route can mean by them.
                //
                // No varyings: there is no vertex stage here to carry an
                // attribute, the triangles being read in world space. A shader
                // that wants per-vertex data reads it from a buffer with the
                // weights below, which is what they are for.
                //
                // depth is the ray parameter rather than an NDC depth. Both are
                // "how far", both order hits the same way, and neither route has
                // the other's number to give — a shader that only compares
                // depths is portable, one that expects [0, 1] is not.
                Fragment fragment;
                fragment.out = dst;
                fragment.x = px;
                fragment.y = py;
                fragment.depth = t;

                // Möller-Trumbore's u and v are already the weights of v1 and v2,
                // and they are exact rather than corrected: a ray meets the
                // triangle in world space, so there is no projection to undo.
                // Naming them w1 and w2 rather than w0 and w1 is what makes one
                // shader draw the same picture down this route and the walk.
                fragment.w0 = k.sub(k.sub(one, u), v);
                fragment.w1 = u;
                fragment.w2 = v;
                fragment.material = material;
                a.shading.shade(k, fragment);
                return;
            }
            if (a.shading.mode == ShadingMode::Diffuse) {
                const Reg<Vec3> normal = k.normalize(k.cross(e1, e2));
                const Reg<Vec3> point = k.add(world_origin, k.scale(world_dir, t));
                const Reg<Vec3> to_light = k.normalize(k.sub(light, point));
                const Reg<Scalar> diffuse = k.max(zero, k.dot(normal, to_light));
                const Reg<Vec3> lit = k.scale(base, diffuse);
                k.copy_into(dst.component(0), lit.component(0));
                k.copy_into(dst.component(1), lit.component(1));
                k.copy_into(dst.component(2), lit.component(2));
            } else {
                k.copy_into(dst.component(0), u);
                k.copy_into(dst.component(1), v);
                k.copy_into(dst.component(2), k.sub(k.sub(one, u), v));
            }
        };

        if (!a.predicated) {
            k.if_(k.lt(t, best_t), [&] {
                k.copy_into(best_t, t);
                if (deferred) {
                    // A handful of scalars instead of a colour. Everything a
                    // shader that asks nothing of the geometry needs, which is
                    // why deferring costs registers rather than a second pass
                    // over anything.
                    if (keeps_bary) {
                        k.copy_into(best_u, u);
                        k.copy_into(best_v, v);
                    }
                    k.copy_into(best_material, per_triangle_material());
                    k.copy_into(anything_hit, one);
                    if (bouncing) {
                        // The edges the intersection already built, crossed. Not
                        // normalised here — the bounce below does it once for
                        // the winner rather than once per candidate.
                        const Reg<Vec3> n = k.cross(e1, e2);
                        for (uint32_t c = 0; c < 3; ++c) {
                            k.copy_into(best_normal.component(c), n.component(c));
                        }
                    }
                } else {
                    shade(best);
                }
            });
            k.place(next);
        } else {
            // Four exits and the depth test collapse to one number, min() of two
            // flags being their AND. Nothing splits the warp.
            const Reg<Scalar> hit =
                k.min(k.min(survived[0], survived[1]), k.min(survived[2], survived[3]));
            const Reg<Scalar> take = k.min(hit, k.lt(t, best_t));

            // Into a scratch range rather than into best, the blend still
            // needing the old value — emit_keep does the same for coverage.
            const Reg<Vec3> shaded = k.vec3();
            if (!deferred) {
                shade(shaded);
            }

            // take*new + (1 - take)*old, not the cheaper old + take*(new - old):
            // that one rounds, and the raster variant drifted on 69,715 pixels
            // of 200,000 before it was changed.
            const Reg<Scalar> keep = k.sub(one, take);
            const auto blend = [&](Reg<Scalar> dst, Reg<Scalar> src) {
                k.copy_into(dst, k.mul(dst, keep));
                k.fma(dst, take, src);
            };
            blend(best_t, t);
            if (deferred) {
                if (keeps_bary) {
                    blend(best_u, u);
                    blend(best_v, v);
                }
                blend(best_material, per_triangle_material());
                blend(anything_hit, one);
                if (bouncing) {
                    const Reg<Vec3> n = k.cross(e1, e2);
                    for (uint32_t c = 0; c < 3; ++c) {
                        blend(best_normal.component(c), n.component(c));
                    }
                }
            } else {
                blend(best.component(0), shaded.component(0));
                blend(best.component(1), shaded.component(1));
                blend(best.component(2), shaded.component(2));
            }
        }
        k.fma(cursor, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

        // A leaf is one node's worth of triangles, so finishing it means going
        // back for the next node rather than to the pixel.
        if (bvh) {
            k.branch_to(traverse, k.gt(sp, zero));

            // The lower level is finished. Two levels, that means the next
            // instance in this leaf and then the next node above it; one level,
            // it means the pixel.
            if (two_level) {
                k.place(instance_next);
                k.fma(inst_i, one, one);
                k.branch_to(instance_top, k.lt(inst_i, inst_count));
                k.branch_to(tlas_next);
            }
            k.place(done);
        }

        // One turn of the walk is over. Add what this surface returns, aim at
        // where the next one would be, and go round again.
        if (bouncing) {
            // A miss leaves best_t at the ceiling it was reset to, and a point
            // that far along the ray overflows. The attenuation is about to be
            // zeroed either way, but zero times an infinity is a NaN and that
            // spreads into the pixel — so the distance is clamped rather than
            // the arithmetic being skipped, which would split the warp.
            const Reg<Scalar> reach = k.min(best_t, k.constant(1.0e6f));

            // How much of the light a surface facing this way takes.
            //
            // Two-sided, because a mesh built by concatenating boxes has no
            // winding worth trusting and a face turned away should be dim
            // rather than black. Directional, not a point light: the position
            // is read as a direction and the hit point never enters the
            // arithmetic. A point light needs one, and a ray that met nothing
            // has none — only the ceiling best_t was reset to. Taking a point
            // that far out and normalising towards a light beside the scene
            // loses every digit and returns a NaN, which survives being
            // multiplied by an attenuation of zero and reaches the pixel. A
            // walk that bounces wants a light at infinity anyway.
            const auto lambert_of = [&](Reg<Vec3> normal, Reg<Scalar> dst) {
                IRBuilder::Scratch scope(k);
                const Reg<Scalar> raw = k.dot(k.normalize(normal), k.normalize(light));
                k.copy_into(dst, k.max(raw, k.sub(zero, raw)));
            };

            // Adds what a surface returns to the pixel and takes the rest of the
            // walk down by what it kept. `blocked` is 1 where something stands
            // between the surface and the light, and 0 on a walk that never
            // asked.
            //
            // Every register it takes is inside the scope, so what it costs is
            // a peak rather than a residue: the direction below is worked out
            // after this has given its registers back.
            const auto absorb = [&](Reg<Scalar> hit, Reg<Scalar> mat, Reg<Scalar> lambert,
                                    Reg<Scalar> blocked, Reg<Scalar> terminal) {
                IRBuilder::Scratch scope(k);
                const Reg<Scalar> entry = k.mul(
                    mat, k.constant(static_cast<float>(MATERIAL_FLOATS * sizeof(float))));
                const Reg<Vec3> colour =
                    k.load_vec3(entry, static_cast<float>(a.material_table_offset));

                // What the surface passes on, and nothing on the last turn.
                //
                // A walk that simply stops is still travelling when it does,
                // and the weight it was still carrying has nowhere to go: drop
                // it and the deepest thing the ray reached comes back black,
                // which in a room of facing mirrors is a hole at the end of the
                // tunnel rather than a tunnel. Reading the last surface as
                // matte spends that weight where the ray actually was, so the
                // reflections run out into the colour of the last mirror
                // rather than into nothing.
                const Reg<Scalar> carried = k.mul(
                    k.add(
                        k.load(entry, static_cast<float>(a.material_table_offset + 12)),
                        k.load(entry, static_cast<float>(a.material_table_offset + 16))),
                    k.sub(one, terminal));

                // Ambient and the lit term, so nothing is wholly black. A
                // blocked surface keeps the ambient and loses the rest, which is
                // what makes a shadow dark rather than absent.
                const Reg<Scalar> lit = k.constant(0.3f);
                k.fma(lit, k.mul(lambert, k.sub(one, blocked)), k.constant(0.7f));

                // A miss returns the sky and ends the walk: nothing beyond it
                // can add anything, which zeroing the attenuation says without a
                // branch.
                const Reg<Scalar> missed = k.sub(one, hit);
                for (uint32_t c = 0; c < 3; ++c) {
                    const float sky = c == 2 ? 0.35f : (c == 1 ? 0.25f : 0.18f);

                    // hit  : attenuation * colour * lit * (1 - carried)
                    // miss : attenuation * sky
                    const Reg<Scalar> surface =
                        k.mul(k.mul(colour.component(c), lit), k.sub(one, carried));
                    const Reg<Scalar> gives =
                        k.add(k.mul(hit, surface), k.mul(missed, k.constant(sky)));
                    k.fma(accumulated.component(c), attenuation.component(c), gives);

                    // What the next turn may still contribute. A miss leaves
                    // nothing.
                    k.copy_into(attenuation.component(c),
                                k.mul(attenuation.component(c), k.mul(hit, carried)));
                }
            };

            // Where the ray goes on from a surface, written into `out`, and the
            // point it leaves from, written into `origin`. A ray that met
            // nothing keeps the ray it had — blended by `hit` rather than
            // branched on, because aiming a finished walk from where a miss
            // "ended" would put the origin a million units out and trace from
            // there. The intersection arithmetic loses every digit it has at
            // that distance and hands back NaN, which then survives being
            // multiplied by an attenuation of zero.
            //
            // `out` may be `dir` itself, so nothing is written until the last
            // loop and `point` is taken from the incoming ray before then.
            const auto leave = [&](Reg<Scalar> hit, Reg<Scalar> mat, Reg<Vec3> normal,
                                   Reg<Vec3> out) {
                IRBuilder::Scratch scope(k);
                const Reg<Scalar> entry = k.mul(
                    mat, k.constant(static_cast<float>(MATERIAL_FLOATS * sizeof(float))));
                const Reg<Scalar> transmission =
                    k.load(entry, static_cast<float>(a.material_table_offset + 16));
                const Reg<Vec3> point = k.add(origin, k.scale(dir, reach));
                const Reg<Vec3> d = k.normalize(dir);
                const Reg<Vec3> n = k.normalize(normal);

                // The normal as the ray meets it. A scene built by concatenating
                // boxes has no winding worth trusting, and both the lift and
                // Snell's law want the side the ray is on rather than the side
                // the triangle claims.
                const Reg<Scalar> inside = k.gt(k.dot(d, n), zero);
                const Reg<Scalar> flip = k.sub(one, k.add(inside, inside));
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(n.component(c), k.mul(n.component(c), flip));
                }
                const Reg<Scalar> cosi = k.sub(zero, k.dot(d, n));

                const Reg<Vec3> chosen = k.vec3();
                {
                    IRBuilder::Scratch mirror(k);
                    const Reg<Scalar> twice = k.sub(zero, k.add(cosi, cosi));
                    for (uint32_t c = 0; c < 3; ++c) {
                        k.copy_into(chosen.component(c),
                                    k.sub(d.component(c), k.mul(n.component(c), twice)));
                    }
                }
                {
                    // Snell's law, with the ratio picked by which side the ray
                    // is on: entering divides by the index, leaving multiplies
                    // by it. A negative discriminant is total internal
                    // reflection — at this angle the ray has no way out — and is
                    // clamped before the root so the arithmetic stays a number,
                    // the direction then being left at the mirror one.
                    //
                    // One direction, not a blend of the two: a blend points at
                    // neither surface. Splitting into both is what recursion
                    // buys and a loop keeping no stack does not have.
                    IRBuilder::Scratch bend(k);
                    const Reg<Scalar> inv = k.rcp(
                        k.load(entry, static_cast<float>(a.material_table_offset + 20)));
                    const Reg<Scalar> eta =
                        k.add(inv, k.mul(inside, k.sub(k.rcp(inv), inv)));
                    const Reg<Scalar> disc =
                        k.sub(one, k.mul(k.mul(eta, eta), k.sub(one, k.mul(cosi, cosi))));
                    const Reg<Scalar> bends =
                        k.mul(k.gt(transmission, zero), k.ge(disc, zero));
                    const Reg<Scalar> coef =
                        k.sub(k.mul(eta, cosi), k.sqrt(k.max(disc, zero)));
                    for (uint32_t c = 0; c < 3; ++c) {
                        const Reg<Scalar> bent = k.add(k.mul(eta, d.component(c)),
                                                       k.mul(coef, n.component(c)));
                        k.copy_into(
                            chosen.component(c),
                            k.add(chosen.component(c),
                                  k.mul(bends, k.sub(bent, chosen.component(c)))));
                    }
                }
                {
                    // Off the surface along the way out rather than along the
                    // normal: a grazing turn leaves at a shallow angle, and the
                    // normal would push it through a neighbouring face.
                    IRBuilder::Scratch lift(k);
                    for (uint32_t c = 0; c < 3; ++c) {
                        const Reg<Scalar> lifted =
                            k.add(point.component(c),
                                  k.mul(chosen.component(c), k.constant(1e-3f)));
                        k.copy_into(
                            origin.component(c),
                            k.add(origin.component(c),
                                  k.mul(hit, k.sub(lifted, origin.component(c)))));
                    }
                }
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(
                        out.component(c),
                        k.add(dir.component(c),
                              k.mul(hit, k.sub(chosen.component(c), dir.component(c)))));
                }
            };

            // Republishes the ray to everything that reads one, `dir` having
            // just been set. Two levels keep a second copy, the world one, which
            // an instance's leaf transforms into the first.
            const auto republish = [&] {
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(inv_dir.component(c), k.rcp(dir.component(c)));
                    k.copy_into(world_inv.component(c), inv_dir.component(c));
                    k.copy_into(world_origin.component(c), origin.component(c));
                    k.copy_into(world_dir.component(c), dir.component(c));
                }
            };

            // Whether this is the last turn the walk has. One comparison a turn,
            // and the same in every lane — a loop counter is.
            const Reg<Scalar> terminal = k.ge(
                bounce,
                k.constant(static_cast<float>(a.bounces * (a.shadows ? 2u : 1u)) - 1.0f));

            if (!shadowing) {
                lambert_of(best_normal, lambert);
                absorb(anything_hit, best_material, lambert, zero, terminal);
                leave(anything_hit, best_material, best_normal, dir);
                republish();
            } else {
                const Label surface_turn = k.label();
                const Label turn_end = k.label();
                k.branch_to(surface_turn, k.sub(one, shadow_turn));

                // The shadow turn. What the traversal just found is not a
                // surface but an answer: something stood between the last one
                // and the light. Colour that surface, then go where it sent the
                // ray.
                absorb(hit_any, hit_material, hit_lambert, anything_hit, terminal);
                for (uint32_t c = 0; c < 3; ++c) {
                    k.copy_into(dir.component(c), onward.component(c));
                }
                republish();
                k.copy_into(shadow_turn, zero);
                k.branch_to(turn_end);

                // The surface turn. Put this hit where the next traversal will
                // not overwrite it, work out where the ray goes on from here,
                // and then spend the traversal asking about the light instead.
                k.place(surface_turn);
                k.copy_into(hit_any, anything_hit);
                k.copy_into(hit_material, best_material);
                lambert_of(best_normal, hit_lambert);
                leave(anything_hit, best_material, best_normal, onward);
                {
                    IRBuilder::Scratch scope(k);
                    const Reg<Vec3> towards = k.normalize(light);
                    for (uint32_t c = 0; c < 3; ++c) {
                        k.copy_into(dir.component(c), towards.component(c));
                    }
                }
                republish();
                k.copy_into(shadow_turn, one);
                k.place(turn_end);
            }

            k.fma(bounce, one, one);
            k.branch_to(
                bounce_top,
                k.lt(bounce,
                     k.constant(static_cast<float>(a.bounces * (a.shadows ? 2u : 1u)))));

            for (uint32_t c = 0; c < 3; ++c) {
                k.copy_into(best.component(c), accumulated.component(c));
            }
        }

        // The shading, once the traversal is over and everything a shader wants
        // is in four registers.
        if (deferred && !bouncing) {
            if (a.shade_when == ShadeWhen::DeferredReordered) {
                // A rendezvous, so it belongs exactly here: every thread of the
                // block has finished traversing and none has started shading.
                // Registers and pc travel with a thread, so the frame is
                // unchanged and only which lane runs which arm moves.
                k.reorder(best_material);
            }

            // Guarded, so a ray that met nothing keeps the cleared frame. The
            // guard is inside the reorder rather than around it: REORDER is a
            // rendezvous, and a block whose misses skipped it would wait for
            // threads that had gone.
            k.if_(anything_hit, [&] {
                if (a.shading.mode == ShadingMode::Custom) {
                    Fragment fragment;
                    fragment.out = best;
                    fragment.x = px;
                    fragment.y = py;
                    fragment.depth = best_t;
                    fragment.w0 = k.sub(k.sub(one, best_u), best_v);
                    fragment.w1 = best_u;
                    fragment.w2 = best_v;
                    fragment.material = best_material;
                    a.shading.shade(k, fragment);
                } else {
                    k.copy_into(best.component(0), best_u);
                    k.copy_into(best.component(1), best_v);
                    k.copy_into(best.component(2), k.sub(k.sub(one, best_u), best_v));
                }
            });
        }

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

void run_raytrace_stage(MyGPURuntime& rt, const RaytraceStageArgs& args)
{
    if (args.width == 0 || args.height == 0) {
        throw std::runtime_error("run_raytrace_stage: an image with no pixels");
    }
    if (args.triangle_count == 0) {
        throw std::runtime_error("run_raytrace_stage: nothing to trace");
    }
    if (args.block_rows == 0) {
        throw std::runtime_error("run_raytrace_stage: a block covering no rows");
    }
    if (args.shade_when == ShadeWhen::DeferredReordered && args.block_rows < 2) {
        throw std::runtime_error(
            "run_raytrace_stage: reordering moves threads between the warps of a "
            "block, and a block one row tall is one warp — raise block_rows");
    }
    if (args.bounces == 0) {
        throw std::runtime_error("run_raytrace_stage: a walk of no bounces");
    }
    if (args.shadows && args.bounces < 2) {
        throw std::runtime_error(
            "run_raytrace_stage: a shadow is a second traversal, so the walk has to "
            "have room for one — raise bounces");
    }
    if (args.bounces > 1) {
        if (args.shade_when != ShadeWhen::Deferred) {
            throw std::runtime_error(
                "run_raytrace_stage: a bounce carries the hit past the traversal, "
                "which is what ShadeWhen::Deferred means — set it");
        }
        if (args.material_offset == 0 || args.material_table_offset == 0) {
            throw std::runtime_error(
                "run_raytrace_stage: bouncing needs a material a triangle and a "
                "table to look it up in");
        }
        if (args.traversal == Traversal::Tlas) {
            throw std::runtime_error(
                "run_raytrace_stage: a second level and a second bounce do not "
                "fit the register file together — one tree at a time");
        }
    }
    if (args.shade_when != ShadeWhen::Inline &&
        args.shading.mode == ShadingMode::Diffuse) {
        throw std::runtime_error(
            "run_raytrace_stage: a point light wants the winning triangle's "
            "edges, and a deferred shade keeps four scalars — shade it inline "
            "or colour it with something that asks the geometry nothing");
    }
    if (args.traversal == Traversal::Tlas) {
        if (args.tlas_stack_depth == 0 || args.tlas_offset == 0 ||
            args.instances_offset == 0) {
            throw std::runtime_error(
                "run_raytrace_stage: a second level with no tree, instances or "
                "stack to walk it with");
        }
        if (args.shading.mode == ShadingMode::Diffuse) {
            throw std::runtime_error(
                "run_raytrace_stage: a point light needs a world-space normal, "
                "and an instance's edges are in its own space — the "
                "inverse-transpose that would carry one across is not built");
        }
    }
    if (args.traversal != Traversal::Linear && args.stack_depth == 0) {
        throw std::runtime_error(
            "run_raytrace_stage: a tree to walk and no stack to walk it with — "
            "set stack_depth from Bvh::max_depth");
    }
    if (args.traversal != Traversal::Linear) {
        // One slice a lane, and a block is one warp wide. A stack that did not
        // fit would silently write into the next lane's slice, which shows up as
        // geometry appearing in the wrong pixel rather than as a failure.
        //
        // Both levels at once: the upper one is still on the stack while the
        // lower is walked, which is what makes a two-level slice the sum.
        const size_t floats =
            static_cast<size_t>(args.stack_depth + args.tlas_stack_depth) * WARP_SIZE *
            args.block_rows;
        if (floats > SHARED_MEM_FLOATS) {
            throw std::runtime_error(
                "run_raytrace_stage: a stack " +
                std::to_string(args.stack_depth + args.tlas_stack_depth) + " deep for " +
                std::to_string(WARP_SIZE) + " lanes is " + std::to_string(floats) +
                " floats, and a block holds " + std::to_string(SHARED_MEM_FLOATS));
        }
    }
    if (args.shading.mode == ShadingMode::Custom && !args.shading.shade) {
        throw std::runtime_error(
            "run_raytrace_stage: Custom shading with nothing to emit — set "
            "Shading::shade");
    }

    const dim3 block{WARP_SIZE, args.block_rows, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE,
                    (args.height + args.block_rows - 1) / args.block_rows, 1};

    void* raw[] = {const_cast<RaytraceStageArgs*>(&args)};
    rt.myrt_launch(build_raytrace_program, grid, block, raw);
}
