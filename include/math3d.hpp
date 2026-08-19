#pragma once

#include <cstdint>

// Host-side geometry. Nothing here runs on the simulated GPU: these are the
// values a launch works out ahead of time and hands to a kernel as constants,
// the way an engine sets a uniform before a draw call.
//
// --- why Float3 and not Vec3 -------------------------------------------------
// ir_builder.hpp already spends the names Vec3 and Mat4 on the shape tags that
// describe a *register range*. Any kernel builder includes both headers, so one
// of the two pairs had to give way, and the ISA-facing names belong on the ISA
// side. Borrowing HLSL's float3 / float4x4 for the host half keeps the split
// visible at every use site: Float3 is a value the host owns, Reg<Vec3> is
// three registers on a device that cannot see host memory at all. The same
// distinction hostMem and deviceMem already keep for storage.

struct Float3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Float3 operator+(Float3 a, Float3 b);
Float3 operator-(Float3 a, Float3 b);
Float3 operator*(Float3 v, float s);

float dot(Float3 a, Float3 b);
Float3 cross(Float3 a, Float3 b);
float length(Float3 v);

// Throws std::runtime_error on a zero-length vector rather than returning a
// vector of NaNs — a degenerate camera basis otherwise turns into a frame that
// is uniformly blank, with nothing to point at.
Float3 normalize(Float3 v);

// Clip space, where w survives the transform. Kept as its own type because the
// perspective divide is the one step that must not be skipped, and a Float3
// return would let it be.
struct Float4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

// Row-major across 16 floats: m[row * 4 + col]. The same order
// V_MATVEC_MAT4_F32 reads its 16 registers in, so a matrix reaches the device
// as a straight copy with no transpose in between — one fewer convention to get
// backwards.
struct Float4x4 {
    float m[16] = {};

    static Float4x4 identity();

    float& at(uint32_t row, uint32_t col);
    float at(uint32_t row, uint32_t col) const;
};

Float4x4 operator*(const Float4x4& a, const Float4x4& b);

// Applies the matrix to (v, w), returning all four clip-space components.
Float4 transform(const Float4x4& a, Float3 v, float w);

// A camera is described in degrees and std::tan takes radians. Both callers of
// this conversion would get it wrong the same way if each wrote its own, so it
// lives here once.
float radians(float degrees);

// Right-handed, looking down -z: the OpenGL and GLM convention, and the one the
// step 6 ray tracer already assumes for its camera.
Float4x4 look_at(Float3 eye, Float3 target, Float3 up);

// The inverse, by Gauss-Jordan with partial pivoting.
//
// General rather than the cheap affine formula, because nothing here promises a
// matrix is affine — a caller composes what it likes, and a wrong answer from a
// shortcut that did not hold would show up as geometry in the wrong place rather
// than as a failure.
//
// Throws std::runtime_error on a singular matrix, which for a transform means an
// instance flattened to nothing. Silently handing back garbage would put its
// triangles anywhere.
Float4x4 inverse(const Float4x4& m);

// fov_y is in degrees, aspect is width / height. Maps the frustum onto the
// [-1, 1] clip cube. near_z and far_z are positive distances in front of the
// camera, despite the camera looking down -z.
Float4x4 perspective(float fov_y_degrees, float aspect, float near_z, float far_z);

struct Camera {
    Float3 eye{0.0f, 0.0f, 3.0f};
    Float3 target{0.0f, 0.0f, 0.0f};
    Float3 up{0.0f, 1.0f, 0.0f};
    float fov_y_degrees = 60.0f;
    float near_z = 0.1f;
    float far_z = 100.0f;

    // projection * view, the single matrix a vertex kernel needs.
    //
    // Multiplied on the host rather than on the device because the product is
    // the same for every vertex in the launch: doing it per-vertex is precisely
    // the work V_MATMUL_MAT4_F32 would exist to do, and this is why that opcode
    // stays unbuilt.
    Float4x4 view_projection(float aspect) const;
};
