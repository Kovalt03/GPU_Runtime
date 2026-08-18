#pragma once

#include <cstdint>
#include <cstring>

// Half precision, on the host side.
//
// The device never converts: a kernel that multiplies halves reads them already
// packed, as a kernel on hardware does — the narrow type is what is in memory,
// and putting it there is the caller's job. So there is no V_CVT opcode here and
// these are the only two functions in the project that know what an f16 looks
// like.
//
// Two of them share a register, which is the whole point beside the precision:
// a fragment of 256 halves is four registers a lane where a fragment of floats
// is eight, and a tile in memory is half the bytes.

// Round to nearest, ties to even, as hardware does. Values outside the range
// saturate to infinity rather than wrapping — a matrix multiply that overflows
// should produce something a test can see.
uint16_t f32_to_f16(float value);
float f16_to_f32(uint16_t half);

// Two halves in the bits of one float register. The result is a bit pattern
// rather than a number: nothing may do arithmetic on it, and the only
// instructions that read it are the ones whose names end in F16.
inline float pack_f16x2(uint16_t low, uint16_t high)
{
    const uint32_t bits =
        static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline void unpack_f16x2(float packed, uint16_t& low, uint16_t& high)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &packed, sizeof(bits));
    low = static_cast<uint16_t>(bits & 0xFFFFu);
    high = static_cast<uint16_t>(bits >> 16);
}
