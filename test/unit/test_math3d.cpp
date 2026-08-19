#include <stdexcept>

#include <gtest/gtest.h>

#include "math3d.hpp"

namespace {

// Matrix entries accumulate a few multiplies, so exact equality is the wrong
// bar. Everything here is far above float noise.
constexpr float EPS = 1e-5f;

void expect_near(Float3 got, Float3 want, const char* what)
{
    EXPECT_NEAR(got.x, want.x, EPS) << what << " .x";
    EXPECT_NEAR(got.y, want.y, EPS) << what << " .y";
    EXPECT_NEAR(got.z, want.z, EPS) << what << " .z";
}

// The perspective divide, which every screen-space expectation goes through.
Float3 to_ndc(Float4 clip)
{
    return Float3{clip.x / clip.w, clip.y / clip.w, clip.z / clip.w};
}

}  // namespace

// ---------------------------------------------------------------------------
// Float3
// ---------------------------------------------------------------------------

TEST(Math3D, CrossIsRightHanded)
{
    // x cross y is +z. The mirrored convention produces a picture that looks
    // reasonable and is backwards, so it is worth pinning here rather than
    // discovering it in a render.
    const Float3 x{1.0f, 0.0f, 0.0f};
    const Float3 y{0.0f, 1.0f, 0.0f};
    expect_near(cross(x, y), Float3{0.0f, 0.0f, 1.0f}, "x cross y");
    expect_near(cross(y, x), Float3{0.0f, 0.0f, -1.0f}, "y cross x");
}

TEST(Math3D, DotAndLengthAgree)
{
    const Float3 v{3.0f, 4.0f, 0.0f};
    EXPECT_NEAR(dot(v, v), 25.0f, EPS);
    EXPECT_NEAR(length(v), 5.0f, EPS);
}

TEST(Math3D, NormalizeGivesUnitLength)
{
    const Float3 v{0.0f, 3.0f, -4.0f};
    const Float3 n = normalize(v);
    EXPECT_NEAR(length(n), 1.0f, EPS);
    expect_near(n, Float3{0.0f, 0.6f, -0.8f}, "normalized");
}

TEST(Math3D, NormalizeRejectsAZeroVector)
{
    // NaNs survive every later multiply and surface as a blank frame, which is
    // indistinguishable from a camera pointed at nothing.
    EXPECT_THROW(normalize(Float3{0.0f, 0.0f, 0.0f}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Worked examples
//
// The tests above pin properties — right-handedness, unit length — and a wrong
// formula can satisfy a property by accident. These give each operation one
// case with a known answer instead.
//
// Every component gets a different value and none of them is zero, which is
// what makes a term reading the wrong operand impossible to hide: with {1,1,1}
// or a basis vector, most of the products are equal or vanish.
// ---------------------------------------------------------------------------

TEST(Math3D, AddAndSubtractActOnEveryComponent)
{
    const Float3 a{1.0f, 2.0f, 3.0f};
    const Float3 b{10.0f, 20.0f, 30.0f};

    expect_near(a + b, Float3{11.0f, 22.0f, 33.0f}, "a + b");
    expect_near(a - b, Float3{-9.0f, -18.0f, -27.0f}, "a - b");
}

TEST(Math3D, ScaleMultipliesEveryComponent)
{
    expect_near(Float3{1.0f, -2.0f, 3.0f} * 2.5f, Float3{2.5f, -5.0f, 7.5f}, "v * 2.5");
}

TEST(Math3D, DotOfTwoDifferentVectors)
{
    // dot(v, v) alone cannot tell a * b from a * a.
    EXPECT_NEAR(dot(Float3{1.0f, 2.0f, 3.0f}, Float3{4.0f, 5.0f, 6.0f}), 32.0f, EPS);
}

TEST(Math3D, CrossOfTwoGeneralVectors)
{
    const Float3 a{1.0f, 2.0f, 3.0f};
    const Float3 b{4.0f, 5.0f, 6.0f};

    expect_near(cross(a, b), Float3{-3.0f, 6.0f, -3.0f}, "a cross b");

    // Anti-commutative, which pins the sign of each of the six products rather
    // than only their difference.
    expect_near(cross(b, a), Float3{3.0f, -6.0f, 3.0f}, "b cross a");
}

// ---------------------------------------------------------------------------
// Float4x4 — the layout V_MATVEC_MAT4_F32 depends on
// ---------------------------------------------------------------------------

TEST(Math3D, StorageIsRowMajor)
{
    // The opcode reads reg[src0 + row * 4 + col]. If the host disagrees, every
    // transform comes out transposed, so this is checked directly rather than
    // inferred from a result.
    Float4x4 a = Float4x4::identity();
    a.at(0, 3) = 7.0f;
    EXPECT_FLOAT_EQ(a.m[3], 7.0f) << "row 0, column 3 is index 3";

    a.at(3, 0) = 9.0f;
    EXPECT_FLOAT_EQ(a.m[12], 9.0f) << "row 3, column 0 is index 12";
}

TEST(Math3D, IdentityLeavesAMatrixAlone)
{
    Float4x4 a = Float4x4::identity();
    a.at(1, 2) = 5.0f;
    a.at(0, 0) = 2.0f;

    const Float4x4 left = Float4x4::identity() * a;
    const Float4x4 right = a * Float4x4::identity();
    for (uint32_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(left.m[i], a.m[i]) << "index " << i;
        EXPECT_FLOAT_EQ(right.m[i], a.m[i]) << "index " << i;
    }
}

TEST(Math3D, TranslationMovesAPoint)
{
    Float4x4 t = Float4x4::identity();
    t.at(0, 3) = 1.0f;
    t.at(1, 3) = 2.0f;
    t.at(2, 3) = 3.0f;

    const Float4 moved = transform(t, Float3{0.0f, 0.0f, 0.0f}, 1.0f);
    expect_near(Float3{moved.x, moved.y, moved.z}, Float3{1.0f, 2.0f, 3.0f}, "point");

    // w = 0 marks a direction, which a translation must leave alone.
    const Float4 direction = transform(t, Float3{1.0f, 0.0f, 0.0f}, 0.0f);
    expect_near(Float3{direction.x, direction.y, direction.z}, Float3{1.0f, 0.0f, 0.0f},
                "direction");
}

TEST(Math3D, ProductAppliesTheRightFactorFirst)
{
    // perspective * view means view runs first. Two translations make the order
    // visible without any trigonometry in the way.
    Float4x4 first = Float4x4::identity();
    first.at(0, 3) = 1.0f;

    Float4x4 second = Float4x4::identity();
    second.at(0, 3) = 10.0f;
    second.at(0, 0) = 2.0f;

    const Float4 got = transform(second * first, Float3{0.0f, 0.0f, 0.0f}, 1.0f);
    EXPECT_NEAR(got.x, 12.0f, EPS) << "translate by 1, then scale by 2 and add 10";
}

// ---------------------------------------------------------------------------
// Camera matrices
// ---------------------------------------------------------------------------

TEST(Math3D, LookAtPutsTheTargetDownNegativeZ)
{
    // A camera 3 units along +z looking at the origin sees the origin 3 units
    // in front of it, and the camera looks down -z.
    const Float4x4 view =
        look_at(Float3{0.0f, 0.0f, 3.0f}, Float3{0.0f, 0.0f, 0.0f}, Float3{0, 1, 0});

    const Float4 origin = transform(view, Float3{0.0f, 0.0f, 0.0f}, 1.0f);
    expect_near(Float3{origin.x, origin.y, origin.z}, Float3{0.0f, 0.0f, -3.0f},
                "target in view space");

    // And the camera itself lands at the view-space origin.
    const Float4 eye = transform(view, Float3{0.0f, 0.0f, 3.0f}, 1.0f);
    expect_near(Float3{eye.x, eye.y, eye.z}, Float3{0.0f, 0.0f, 0.0f}, "eye");
}

TEST(Math3D, LookAtKeepsUpUpright)
{
    const Float4x4 view =
        look_at(Float3{0.0f, 0.0f, 3.0f}, Float3{0.0f, 0.0f, 0.0f}, Float3{0, 1, 0});

    // A point above the target has to stay above it in view space.
    const Float4 above = transform(view, Float3{0.0f, 1.0f, 0.0f}, 1.0f);
    EXPECT_GT(above.y, 0.0f);
}

TEST(Math3D, LookAtRejectsADegenerateBasis)
{
    // Eye on target leaves forward undefined; up parallel to forward leaves
    // right undefined. Both come out of normalize as a throw.
    EXPECT_THROW(look_at(Float3{1, 1, 1}, Float3{1, 1, 1}, Float3{0, 1, 0}),
                 std::runtime_error);
    EXPECT_THROW(look_at(Float3{0, 0, 3}, Float3{0, 0, 0}, Float3{0, 0, 1}),
                 std::runtime_error);
}

TEST(Math3D, PerspectivePutsDistanceIntoW)
{
    // The divide by w is a divide by distance, which is the whole mechanism of
    // foreshortening. A point 5 units down -z must arrive with w = 5.
    const Float4x4 proj = perspective(60.0f, 1.0f, 0.1f, 100.0f);
    const Float4 clip = transform(proj, Float3{0.0f, 0.0f, -5.0f}, 1.0f);
    EXPECT_NEAR(clip.w, 5.0f, EPS);
}

TEST(Math3D, PerspectiveMapsTheFrustumEdgeToOne)
{
    // At 90 degrees and square aspect, the frustum is exactly as wide as it is
    // deep, so a point at (d, 0, -d) sits on the right edge: ndc.x == 1.
    const Float4x4 proj = perspective(90.0f, 1.0f, 0.1f, 100.0f);
    const Float3 ndc = to_ndc(transform(proj, Float3{4.0f, 0.0f, -4.0f}, 1.0f));
    EXPECT_NEAR(ndc.x, 1.0f, EPS);
    EXPECT_NEAR(ndc.y, 0.0f, EPS);
}

TEST(Math3D, PerspectiveMapsNearAndFarToTheClipCube)
{
    const float near_z = 0.5f;
    const float far_z = 20.0f;
    const Float4x4 proj = perspective(60.0f, 1.5f, near_z, far_z);

    const Float3 at_near = to_ndc(transform(proj, Float3{0.0f, 0.0f, -near_z}, 1.0f));
    const Float3 at_far = to_ndc(transform(proj, Float3{0.0f, 0.0f, -far_z}, 1.0f));

    EXPECT_NEAR(at_near.z, -1.0f, EPS);
    EXPECT_NEAR(at_far.z, 1.0f, EPS);
}

TEST(Math3D, PerspectiveRejectsAZeroNearPlane)
{
    // A zero near plane makes the depth row infinite and every later comparison
    // a comparison of NaNs.
    EXPECT_THROW(perspective(60.0f, 1.0f, 0.0f, 100.0f), std::runtime_error);
    EXPECT_THROW(perspective(60.0f, 1.0f, 1.0f, 1.0f), std::runtime_error);
}

TEST(Math3D, ViewProjectionPutsTheTargetAtTheCentre)
{
    // The composition of the two, which is the single matrix a vertex kernel
    // receives. Whatever the camera looks at lands in the middle of the image.
    const Camera cam;
    const Float4x4 vp = cam.view_projection(16.0f / 9.0f);

    const Float3 ndc = to_ndc(transform(vp, cam.target, 1.0f));
    EXPECT_NEAR(ndc.x, 0.0f, EPS);
    EXPECT_NEAR(ndc.y, 0.0f, EPS);
    EXPECT_GT(ndc.z, -1.0f) << "the target sits between the near and far planes";
    EXPECT_LT(ndc.z, 1.0f);
}
