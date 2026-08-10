#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "math3d.hpp"

namespace {

// Rows and columns of a Float4x4, named so the index arithmetic below reads as
// the row-major rule rather than as a bare 4.
constexpr uint32_t DIMENSION = 4;

// Spelled out rather than taken from M_PI, which is a POSIX extension that
// <cmath> is not required to declare under -std=c++17.
constexpr float PI = 3.14159265358979323846f;

// The one place row-major is spelled out. Both at() overloads go through it, so
// a const and a non-const read cannot drift onto different layouts — and
// V_MATVEC_MAT4_F32 reads its 16 registers in exactly this order.
uint32_t index_of(uint32_t row, uint32_t col)
{
    if (row >= DIMENSION || col >= DIMENSION) {
        throw std::runtime_error("Float4x4::at: (" + std::to_string(row) + ", " +
                                 std::to_string(col) + ") is outside a " +
                                 std::to_string(DIMENSION) + "x" +
                                 std::to_string(DIMENSION) + " matrix");
    }
    return row * DIMENSION + col;
}

}  // namespace

// ---------------------------------------------------------------------------
// Float3
//
// Kept as named, separately tested functions rather than inlined into the camera
// matrices below: a sign error in cross() is far easier to find at this size
// than inside a basis it helped build.
// ---------------------------------------------------------------------------

Float3 operator+(Float3 a, Float3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Float3 operator-(Float3 a, Float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 operator*(Float3 v, float s)
{
    return {v.x * s, v.y * s, v.z * s};
}

float dot(Float3 a, Float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Float3 cross(Float3 a, Float3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float length(Float3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Float3 normalize(Float3 v)
{
    // Throwing rather than returning NaNs: a camera whose eye sits on its
    // target would otherwise carry them through every later multiply and arrive
    // as an empty frame, with nothing along the way reporting a failure.
    const float len = length(v);
    if (len == 0.0f) {
        throw std::runtime_error("normalize: zero-length vector has no direction");
    }
    return v * (1.0f / len);
}

// ---------------------------------------------------------------------------
// Float4x4
//
// Row-major, m[row * 4 + col], matching how V_MATVEC_MAT4_F32 reads registers.
// ---------------------------------------------------------------------------

Float4x4 Float4x4::identity()
{
    Float4x4 mat;
    for (uint32_t i = 0; i < DIMENSION; ++i) {
        mat.at(i, i) = 1.0f;
    }
    return mat;
}

// Both bounds-check through index_of: an out-of-range index would otherwise
// read a neighbouring row and produce a matrix that is wrong but not obviously
// so — a transform that still renders something plausible.
float& Float4x4::at(uint32_t row, uint32_t col)
{
    return m[index_of(row, col)];
}

float Float4x4::at(uint32_t row, uint32_t col) const
{
    return m[index_of(row, col)];
}

Float4x4 operator*(const Float4x4& a, const Float4x4& b)
{
    Float4x4 result;
    for (uint32_t i = 0; i < DIMENSION; ++i) {
        for (uint32_t j = 0; j < DIMENSION; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < DIMENSION; ++k) {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

Float4 transform(const Float4x4& a, Float3 v, float w)
{
    const float in[4] = {v.x, v.y, v.z, w};
    float out[4] = {};
    for (uint32_t row = 0; row < DIMENSION; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < DIMENSION; ++col) {
            sum += a.at(row, col) * in[col];
        }
        out[row] = sum;
    }
    return Float4{out[0], out[1], out[2], out[3]};
}

// ---------------------------------------------------------------------------
// Camera matrices
// ---------------------------------------------------------------------------

float radians(float degrees)
{
    return degrees * (PI / 180.0f);
}

Float4x4 look_at(Float3 eye, Float3 target, Float3 up)
{
    // Right-handed, looking down -z. normalize does the rejecting: an eye
    // sitting on its target leaves forward undefined, and an up parallel to
    // forward leaves right undefined.
    const Float3 forward = normalize(target - eye);
    const Float3 right = normalize(cross(forward, up));
    const Float3 true_up = cross(right, forward);

    // The rotation is the basis written as rows, and the last column undoes the
    // camera's position. The third row is negated because the camera looks
    // along -z while its basis vector points along +z.
    //
    // Laid out as the matrix it is; clang-format would otherwise pack the rows
    // into as few lines as fit.
    // clang-format off
    const Float4x4 view = {
        right.x,    right.y,    right.z,    -dot(right, eye),
        true_up.x,  true_up.y,  true_up.z,  -dot(true_up, eye),
        -forward.x, -forward.y, -forward.z,  dot(forward, eye),
        0.0f,       0.0f,       0.0f,        1.0f,
    };
    // clang-format on
    return view;
}

Float4x4 perspective(float fov_y_degrees, float aspect, float near_z, float far_z)
{
    // A zero near plane makes the depth row infinite, and an empty depth range
    // divides by zero; either way every later depth comparison compares NaNs
    // and the frame comes out blank with nothing having reported a failure.
    if (near_z <= 0.0f || far_z <= near_z) {
        throw std::runtime_error("perspective: need 0 < near_z < far_z, got near " +
                                 std::to_string(near_z) + " and far " +
                                 std::to_string(far_z));
    }
    if (aspect <= 0.0f) {
        throw std::runtime_error("perspective: aspect must be positive, got " +
                                 std::to_string(aspect));
    }

    // std::tan takes radians. Passing degrees still returns a number, and only
    // the two f entries below depend on it — so the mistake survives as a frame
    // with the wrong field of view and nothing else visibly wrong.
    const float half_fov = radians(fov_y_degrees) * 0.5f;
    const float f = 1.0f / std::tan(half_fov);

    // Named so the matrix below fits its own shape. These two map [near, far]
    // onto [-1, 1] once the divide by w has happened.
    const float depth_scale = (far_z + near_z) / (near_z - far_z);
    const float depth_bias = 2.0f * far_z * near_z / (near_z - far_z);

    // The -1 in the last row is what puts -z into w, making the divide by w a
    // divide by distance — which is the whole mechanism of foreshortening.
    // clang-format off
    const Float4x4 projection = {
        f / aspect, 0.0f, 0.0f,        0.0f,
        0.0f,       f,    0.0f,        0.0f,
        0.0f,       0.0f, depth_scale, depth_bias,
        0.0f,       0.0f, -1.0f,       0.0f,
    };
    // clang-format on
    return projection;
}

Float4x4 Camera::view_projection(float aspect) const
{
    // projection * view, so that view is applied first: a vertex is moved into
    // camera space before the frustum is imposed on it. The other order would
    // project world coordinates and then move the result, which means nothing.
    return perspective(fov_y_degrees, aspect, near_z, far_z) * look_at(eye, target, up);
}
