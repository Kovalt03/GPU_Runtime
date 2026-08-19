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
    const Reg<Vec3> origin = k.constant(b.origin.x, b.origin.y, b.origin.z);
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
        // kernels/ray_triangle.cpp does: two multiplies and two adds rather
        // than six. sy runs backwards because rows count down and the frame up.
        const Reg<Scalar> sx =
            k.add(k.mul(px, k.constant(2.0f / static_cast<float>(a.width))),
                  k.constant(1.0f / static_cast<float>(a.width) - 1.0f));
        const Reg<Scalar> sy =
            k.add(k.mul(py, k.constant(-2.0f / static_cast<float>(a.height))),
                  k.constant(1.0f - 1.0f / static_cast<float>(a.height)));
        const Reg<Vec3> dir = k.add(k.scale(right, sx), k.add(k.scale(up, sy), forward));
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
        const Reg<Scalar> i = k.constant(0.0f);

        // Above the traversal, not merely above k.place(top): a leaf falls
        // through to the loop, so anything between the two is issued once a leaf
        // rather than once a launch. Emitted whatever the mode — six moves cost
        // less than the optional it would take to declare them conditionally.
        const Float3& lp = a.shading.light_position;
        const Float3& bc = a.shading.base_colour;
        const Reg<Vec3> light = k.constant(lp.x, lp.y, lp.z);
        const Reg<Vec3> base = k.constant(bc.x, bc.y, bc.z);

        // --- traversal ------------------------------------------------------
        // Emitted before the triangle loop so that the loop's bottom test can
        // branch back into it. Nothing here is issued when the mode is Linear.
        const bool bvh = a.traversal == Traversal::Bvh;
        const Label traverse = k.label();
        const Label traverse_next = k.label();
        const Label leaf_found = k.label();
        const Label done = k.label();

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
        if (bvh) {
            const Reg<Scalar> lane =
                k.sub(px, k.mul(k.block_x(), k.constant(static_cast<float>(WARP_SIZE))));
            stack_slot = k.mul(
                lane, k.constant(static_cast<float>(a.stack_depth * sizeof(float))));

            // Three reciprocals once rather than three divisions a node. A zero
            // component gives an infinite slab bound, which compares the way the
            // test wants: the ray never leaves that slab.
            k.copy_into(inv_dir.component(0), k.rcp(dir.component(0)));
            k.copy_into(inv_dir.component(1), k.rcp(dir.component(1)));
            k.copy_into(inv_dir.component(2), k.rcp(dir.component(2)));

            sp = k.constant(0.0f);
            k.store_shared(stack_slot, zero);  // the root
            k.fma(sp, one, one);

            k.place(traverse);
            k.fma(sp, one, k.constant(-1.0f));
            const Reg<Scalar> node =
                k.load_shared(k.add(stack_slot, k.mul(sp, k.constant(4.0f))));
            const Reg<Scalar> node_addr =
                k.add(k.constant(static_cast<float>(a.bvh_offset)),
                      k.mul(node, k.constant(static_cast<float>(BVH_NODE_BYTES))));

            // The slab test. Two wide loads for six floats, and the same
            // arithmetic nodes_visited runs on the host — a box the two disagree
            // about would drop geometry from one and not the other.
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

            const Reg<Scalar> node_count = k.load(node_addr, 28.0f);
            k.branch_to(traverse_next, k.gt(near, far));
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
                        k.constant(static_cast<float>(a.bvh_offset)),
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

            k.store_shared(k.add(stack_slot, k.mul(sp, k.constant(4.0f))), second);
            k.fma(sp, one, one);
            k.store_shared(k.add(stack_slot, k.mul(sp, k.constant(4.0f))), first);
            k.fma(sp, one, one);
            k.place(traverse_next);
            k.branch_to(traverse, k.gt(sp, zero));
            k.branch_to(done);

            // A leaf runs the loop below over its own range instead of the scene.
            k.place(leaf_found);
            k.copy_into(count, node_count);
            k.copy_into(cursor, k.add(k.constant(static_cast<float>(a.triangles_offset)),
                                      k.mul(k.load(node_addr, 24.0f), stride)));
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
                a.shading.shade(k, fragment);
                return;
            }
            if (a.shading.mode == ShadingMode::Diffuse) {
                const Reg<Vec3> normal = k.normalize(k.cross(e1, e2));
                const Reg<Vec3> point = k.add(origin, k.scale(dir, t));
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
                shade(best);
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
            shade(shaded);

            // take*new + (1 - take)*old, not the cheaper old + take*(new - old):
            // that one rounds, and the raster variant drifted on 69,715 pixels
            // of 200,000 before it was changed.
            const Reg<Scalar> keep = k.sub(one, take);
            const auto blend = [&](Reg<Scalar> dst, Reg<Scalar> src) {
                k.copy_into(dst, k.mul(dst, keep));
                k.fma(dst, take, src);
            };
            blend(best_t, t);
            blend(best.component(0), shaded.component(0));
            blend(best.component(1), shaded.component(1));
            blend(best.component(2), shaded.component(2));
        }
        k.fma(cursor, stride, one);
        k.fma(i, one, one);
        k.branch_to(top, k.lt(i, count));

        // A leaf is one node's worth of triangles, so finishing it means going
        // back for the next node rather than to the pixel.
        if (bvh) {
            k.branch_to(traverse, k.gt(sp, zero));
            k.place(done);
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
    if (args.traversal == Traversal::Bvh && args.stack_depth == 0) {
        throw std::runtime_error(
            "run_raytrace_stage: a tree to walk and no stack to walk it with — "
            "set stack_depth from Bvh::max_depth");
    }
    if (args.traversal == Traversal::Bvh) {
        // One slice a lane, and a block is one warp wide. A stack that did not
        // fit would silently write into the next lane's slice, which shows up as
        // geometry appearing in the wrong pixel rather than as a failure.
        const size_t floats = static_cast<size_t>(args.stack_depth) * WARP_SIZE;
        if (floats > SHARED_MEM_FLOATS) {
            throw std::runtime_error(
                "run_raytrace_stage: a stack " + std::to_string(args.stack_depth) +
                " deep for " + std::to_string(WARP_SIZE) + " lanes is " +
                std::to_string(floats) + " floats, and a block holds " +
                std::to_string(SHARED_MEM_FLOATS));
        }
    }
    if (args.shading.mode == ShadingMode::Custom && !args.shading.shade) {
        throw std::runtime_error(
            "run_raytrace_stage: Custom shading with nothing to emit — set "
            "Shading::shade");
    }

    const dim3 block{WARP_SIZE, 1, 1};
    const dim3 grid{(args.width + WARP_SIZE - 1) / WARP_SIZE, args.height, 1};

    void* raw[] = {const_cast<RaytraceStageArgs*>(&args)};
    rt.myrt_launch(build_raytrace_program, grid, block, raw);
}
