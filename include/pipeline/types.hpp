#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include "ir_builder.hpp"
#include "math3d.hpp"

// The graphics layer. Everything below is a thin wrapper over myrt_launch: the
// runtime stays general-purpose the way CUDA is, and the knowledge that these
// numbers are triangles lives only here.
//
// Rasterisation runs in two passes, because the two halves are indexed
// differently. The MVP transform is per vertex and coverage is per pixel, and a
// single kernel would have to pick one and waste threads on the other. Real
// hardware splits them for the same reason, into a vertex stage and a fragment
// stage, and this runtime can already launch twice.
//
//   pass 1  vertex_program   1 thread = 1 vertex   world -> screen
//   pass 2  raster_program   1 thread = 1 pixel    coverage, nearest wins
//
// Each stage is a header of its own. This one holds only what more than one of
// them needs: the strides they address device memory by, and the triangle the
// vertex stage hands the raster stages.

// --- device buffer layout ---------------------------------------------------
// Kernels address device memory by byte offset, so the strides are named once
// here rather than spelled out at every load.

// x, y, z in world space.
inline constexpr uint32_t WORLD_VERTEX_FLOATS = 3;
inline constexpr uint32_t WORLD_VERTEX_BYTES = WORLD_VERTEX_FLOATS * sizeof(float);

// x, y in pixels, the NDC depth pass 2 compares, and 1/w.
//
// w is kept because the divide is exactly what makes screen-space interpolation
// wrong for anything but depth. Barycentric weights taken from projected
// vertices are affine; an attribute carried across a perspective-projected
// triangle needs them weighted by 1/w and renormalised. Depth does not, being
// linear in screen space by construction — which is the whole reason a depth
// buffer stores NDC z.
//
// Costs a float per vertex and a reciprocal per pixel, and without it the
// rasteriser and the ray tracer only agree on a scene whose triangles all sit
// at one depth.
inline constexpr uint32_t SCREEN_VERTEX_FLOATS = 4;
inline constexpr uint32_t SCREEN_VERTEX_BYTES = SCREEN_VERTEX_FLOATS * sizeof(float);

// --- varyings ---------------------------------------------------------------
// Values a vertex carries to the pixels its triangles cover, interpolated on the
// way. GLSL declares them `out` in one stage and `in` in the next; here a launch
// says how many floats a vertex adds and they ride in the slots after the four
// above.
//
// Zero is the default and is what every figure taken before they existed used:
// a screen vertex is then exactly the four floats it always was, and no stride
// anywhere moves. A kernel pays for the varyings it declares and no others —
// there is no dead-code elimination here to drop an unread one.
//
// The cap is a register budget rather than a format: interpolating one costs
// three loads and three multiply-adds a pixel a triangle, and a raster kernel
// already spends around sixty registers before it starts.
inline constexpr uint32_t MAX_VARYINGS = 8;

// How wide a screen vertex is when a launch carries varyings. The stride reaches
// the kernels through their args rather than as a constant, which is what lets
// the zero case emit exactly the instructions it emitted before.
inline constexpr uint32_t screen_vertex_floats(uint32_t varyings)
{
    return SCREEN_VERTEX_FLOATS + varyings;
}

inline constexpr uint32_t screen_vertex_bytes(uint32_t varyings)
{
    return screen_vertex_floats(varyings) * sizeof(float);
}

// A triangle as pass 1 leaves it: x and y in pixels, z the NDC depth.
struct ScreenTriangle {
    Float3 v0;
    Float3 v1;
    Float3 v2;

    // One per vertex, in that order. Named rather than packed into a Float3,
    // which would read as a vector and is three unrelated scalars.
    float inv_w0 = 1.0f;
    float inv_w1 = 1.0f;
    float inv_w2 = 1.0f;
};

// RGB per pixel, the layout kernels/ray_triangle.cpp already writes, so the two
// renderers can be compared image against image rather than by description.
inline constexpr uint32_t PIXEL_FLOATS = 3;
inline constexpr uint32_t PIXEL_BYTES = PIXEL_FLOATS * sizeof(float);

// --- instancing --------------------------------------------------------------
// Where an instance's model matrix meets the view-projection.
//
// A flag because it is a crossing rather than a winner, and the cost model says
// where: MATMUL is four MATVECs, so folding once an instance beats an extra
// MATVEC at every vertex exactly when an instance has more than four vertices.
// Both arms deliver one matrix through the constant window, so that cost cancels
// and what is left is the arithmetic.
enum class InstanceTransform {
    // Two transforms at every vertex: by the model matrix, then by the
    // view-projection. No composition anywhere, and nothing to schedule.
    PerVertex,

    // A pass ahead of pass 1 folds the two, one thread an instance. Pass 1 then
    // does the single MATVEC it always did, reading a different matrix.
    ComposePass,
};

// --- what a vertex shader is handed ------------------------------------------
// The vertex a pass-1 thread is working on, before anything has been done to it.
//
// The fragment stage got a callback first, and the reason pass 1 did not was
// that it is short enough to write out: fifty-eight lines, half of them stores,
// against two hundred of coverage and depth. What changes that is a stage which
// has to *compute* a varying rather than carry one — a normal turned by a model
// matrix, a texture coordinate generated, a position blended from a bone
// palette. None of those is expressible by copying.
struct Vertex {
    // Write the position here. Object space, since pass 1 applies the matrices
    // afterwards — a shader that wanted to skip them would have nothing to
    // write, and pass 2 reads screen coordinates either way.
    Reg<Vec3> out;

    // Where the vertex was, as the buffer holds it.
    Reg<Vec3> position;

    // Which vertex and which instance, for a shader indexing a buffer of its
    // own. The instance is zero on a launch that has none.
    Reg<Scalar> index;
    Reg<Scalar> instance;

    // The attributes this launch declared, loaded and not yet stored. A shader
    // may overwrite them, which is the difference between carrying a varying and
    // producing one.
    std::array<Reg<Scalar>, MAX_VARYINGS> varyings{};
    uint32_t varying_count = 0;
};

// Emits the instructions that place a vertex. Called once as the kernel is
// built, like everything else here.
using VertexFn = std::function<void(IRBuilder&, const Vertex&)>;

// --- shading -----------------------------------------------------------------
// How a hit is coloured, shared by both renderers so that a frame from one can
// be held against a frame from the other.
//
// Barycentric is the debug colouring, and was for a long time the only one the
// rasteriser could produce: pass 1 keeps screen x, y, depth and 1/w, so the
// world position and normal a light needs are gone by the time pass 2 runs. What
// changed is that the geometry now stays on the device — the world vertices pass
// 1 read are still there, and a face normal a triangle is a small buffer beside
// them.
enum class ShadingMode {
    Barycentric,
    Diffuse,

    // What the caller supplies. Shading::shade holds a function that emits the
    // instructions, and it is handed the fragment below.
    Custom,
};

struct Fragment;

// Emits the instructions that colour a fragment. Called once when the kernel is
// built, not once a pixel — like everything else here, what it writes is a
// program rather than a value.
//
// It runs for fragments that go on to lose. Under `predicated` no lane is
// masked, so the colour is computed and then blended away; the ray tracer shades
// inside its triangle loop, so a pixel that meets three triangles shades three
// times. Writing `out` is therefore safe and any other effect is not — a store
// or an atomic in here fires for candidates the frame never shows.
using ShadeFn = std::function<void(IRBuilder&, const Fragment&)>;

struct Shading {
    ShadingMode mode = ShadingMode::Barycentric;

    // World space. Only read in Diffuse.
    Float3 light_position{2.0f, 3.0f, 1.0f};
    Float3 base_colour{1.0f, 1.0f, 1.0f};

    // Emits the colour, when mode is Custom. It rides inside Shading rather than
    // beside it because every route that can shade already takes a Shading: the
    // walk and the ray tracer both pick up a caller's shader without a signature
    // of their own, and a mode that named a function living somewhere else could
    // reach a route that never looked for it.
    ShadeFn shade;
};

// --- what a fragment shader is handed ---------------------------------------
// The pixel a raster kernel has decided is covered, and everything about it the
// kernel already worked out. A caller's shading function writes `out` and may
// read anything else here, or any buffer it knows the offset of.
//
// This is the fragment stage's interface, and it is deliberately the same set
// GLSL gives one: interpolated values, the pixel's position, and whatever the
// launch put in memory. What it adds is that the instruction set is not fenced
// off — a shading function may store to global memory or take an atomic, which
// GLSL allows only through an image or a storage buffer.
struct Fragment {
    // Write the colour here. Three registers, and what the pixel becomes if this
    // fragment wins the depth test.
    Reg<Vec3> out;

    // The varyings this launch declared, already interpolated with perspective
    // correction. varyings[i] is the i-th float a vertex carried.
    std::array<Reg<Scalar>, MAX_VARYINGS> varyings{};
    uint32_t varying_count = 0;

    // Where the pixel is and how deep the fragment is, in the same NDC the depth
    // buffer holds.
    Reg<Scalar> x;
    Reg<Scalar> y;
    Reg<Scalar> depth;

    // The perspective-corrected barycentric weights, for a shader that wants to
    // interpolate something the varyings do not carry — a vertex attribute it
    // reads out of its own buffer, say.
    Reg<Scalar> w0;
    Reg<Scalar> w1;
    Reg<Scalar> w2;

    // What the instance this hit belongs to is made of, and zero on every route
    // that has no instances.
    //
    // The only field here that differs between lanes for a reason the geometry
    // did not choose: two neighbouring pixels can hit different objects, so a
    // shader that branches on this splits the warp along the scene's own seams.
    // That is what makes it the input SER wants, and why it is a number the
    // runtime carries and never interprets.
    Reg<Scalar> material;
};

// One unit normal a triangle, which is what the raster routes read rather than
// deriving it per pixel. The ray tracer takes the cross product it already has
// the edges for; the rasteriser would have to load three world vertices to do
// the same, and does load them — for the point a point light needs.
inline constexpr uint32_t FACE_NORMAL_FLOATS = 3;
inline constexpr uint32_t FACE_NORMAL_BYTES = FACE_NORMAL_FLOATS * sizeof(float);

// --- depth -------------------------------------------------------------------
// What a raster launch does with the depth buffer.
//
// A depth prepass is two launches over the same geometry: the first keeps the
// nearest depth a pixel and colours nothing, the second colours only the
// triangle that depth names. What it buys is that a pixel is shaded once instead
// of once per covering triangle; what it costs is a second walk of every
// triangle, which is not cheap here — coverage is most of the loop.
enum class DepthUse {
    // No depth buffer. The running best in a register is all a single-pass walk
    // needs, one thread owning one pixel outright.
    None,

    // Write the nearest depth and shade nothing.
    Prepass,

    // Read it, and shade only the triangle that owns the pixel.
    EarlyZ,

    // Start from what the buffer holds and write back what wins, colour and
    // depth together. The standard depth-tested draw, and the only mode where a
    // second draw can land in a frame the first one already wrote.
    //
    // A thread no longer owns its pixel outright, which is the whole difference:
    // the three modes above start from an empty pixel and this one does not.
    Test,
};

// One float a pixel. Separate from the colour buffer rather than a fourth
// channel of it, because the prepass writes only this and the frame is read back
// as RGB.
inline constexpr uint32_t DEPTH_BYTES = sizeof(float);
