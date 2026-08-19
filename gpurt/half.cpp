#include <cstring>

#include "half.hpp"

// The conversions, written out rather than left to a compiler intrinsic: this
// project's rule is that a number it reports can be traced to something in the
// repository, and _Float16 support varies by target.

uint16_t f32_to_f16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Infinity keeps its sign; a NaN stays a NaN rather than becoming one.
        return static_cast<uint16_t>(sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // saturates
    }
    if (exponent <= 0) {
        // Subnormal, or too small to be one. The shift brings the implicit bit
        // back in before it is lost.
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        const uint32_t rounded =
            (mantissa + (1u << (shift - 1)) - 1 + ((mantissa >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }

    // Round to nearest, ties to even: add half of the last kept bit, and one
    // less than that when the kept bit is even.
    const uint32_t kept = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1FFFu;
    uint32_t out = static_cast<uint32_t>(exponent) << 10 | kept;
    if (remainder > 0x1000u || (remainder == 0x1000u && (kept & 1u) != 0u)) {
        ++out;  // carrying into the exponent is what it should do
    }
    return static_cast<uint16_t>(sign | out);
}

float f16_to_f32(uint16_t half)
{
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            // Subnormal: normalise it, which f32 has room for.
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        } else {
            bits = sign;
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
