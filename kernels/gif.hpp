#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "math3d.hpp"

// An animated GIF, written by hand.
//
// A still frame proves a renderer draws; it cannot prove anything moves. The
// project has written PPM since its first kernel because any viewer opens one
// and a diff reads it, and neither of those is true of a format that carries a
// sequence — so this is the one place a real encoder is worth the lines.
//
// Written out rather than linked because gpurt has no third-party dependency and
// an animation viewer is a poor reason to acquire the first. LZW is about a
// hundred lines and the format around it is a header, a palette and a block
// stream.

// Six levels a channel, which is the 216-colour cube every early web palette
// used. A rendered frame here is smooth shading over a black ground, so nearest
// on a fixed cube costs nothing a quantiser would recover — and a quantiser is a
// second algorithm to be wrong in.
inline constexpr uint32_t GIF_LEVELS = 6;
inline constexpr uint32_t GIF_COLOURS = GIF_LEVELS * GIF_LEVELS * GIF_LEVELS;

namespace gif_detail {

inline uint8_t level_of(float channel)
{
    const float clamped = channel < 0.0f ? 0.0f : (channel > 1.0f ? 1.0f : channel);
    return static_cast<uint8_t>(clamped * (GIF_LEVELS - 1) + 0.5f);
}

// Codes are written least significant bit first and run across byte boundaries,
// which is the part of LZW that a stream of bytes cannot express directly.
struct BitStream {
    std::vector<uint8_t> bytes;
    uint32_t partial = 0;
    uint32_t bits = 0;

    void write(uint32_t code, uint32_t width)
    {
        partial |= code << bits;
        bits += width;
        while (bits >= 8) {
            bytes.push_back(static_cast<uint8_t>(partial & 0xFF));
            partial >>= 8;
            bits -= 8;
        }
    }

    void flush()
    {
        if (bits > 0) {
            bytes.push_back(static_cast<uint8_t>(partial & 0xFF));
            partial = 0;
            bits = 0;
        }
    }
};

// GIF's variable-code LZW. The dictionary is keyed on (prefix, next), which for
// 216 colours and a 12-bit ceiling fits a flat table — a map would allocate per
// pixel for no gain at this size.
inline std::vector<uint8_t> compress(const std::vector<uint8_t>& indices,
                                     uint32_t code_bits)
{
    const uint32_t clear = 1u << code_bits;
    const uint32_t end = clear + 1;

    BitStream out;
    std::vector<int32_t> table((1u << 12) * GIF_COLOURS, -1);
    uint32_t next = end + 1;
    uint32_t width = code_bits + 1;

    out.write(clear, width);
    if (indices.empty()) {
        out.write(end, width);
        out.flush();
        return out.bytes;
    }

    uint32_t prefix = indices[0];
    for (size_t i = 1; i < indices.size(); ++i) {
        const uint32_t next_index = indices[i];
        const size_t slot = static_cast<size_t>(prefix) * GIF_COLOURS + next_index;
        if (table[slot] >= 0) {
            prefix = static_cast<uint32_t>(table[slot]);
            continue;
        }

        out.write(prefix, width);
        if (next < (1u << 12)) {
            table[slot] = static_cast<int32_t>(next++);
            if (next > (1u << width) && width < 12) {
                ++width;
            }
        } else {
            // The dictionary is full. Clearing is what keeps a long animation
            // compressing rather than emitting twelve bits a pixel for ever.
            out.write(clear, width);
            std::fill(table.begin(), table.end(), -1);
            next = end + 1;
            width = code_bits + 1;
        }
        prefix = next_index;
    }

    out.write(prefix, width);
    out.write(end, width);
    out.flush();
    return out.bytes;
}

inline void put16(std::ostream& out, uint16_t v)
{
    out.put(static_cast<char>(v & 0xFF));
    out.put(static_cast<char>(v >> 8));
}

// Sub-blocks of at most 255 bytes, each preceded by its length, terminated by a
// zero. The format's only framing.
inline void write_blocks(std::ostream& out, const std::vector<uint8_t>& data)
{
    size_t at = 0;
    while (at < data.size()) {
        const size_t n = std::min<size_t>(255, data.size() - at);
        out.put(static_cast<char>(n));
        out.write(reinterpret_cast<const char*>(&data[at]), static_cast<long>(n));
        at += n;
    }
    out.put(0);
}

}  // namespace gif_detail

// Writes an animation. `frames` holds width * height pixels each, and
// `delay_centiseconds` is what the format counts in — 3 is about thirty a
// second, which is the rate a motion file usually carries.
inline void write_gif(const std::string& path,
                      const std::vector<std::vector<Float3>>& frames, uint32_t width,
                      uint32_t height, uint32_t delay_centiseconds = 3)
{
    if (frames.empty()) {
        throw std::runtime_error("write_gif: an animation with no frames");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_gif: cannot open " + path + " for writing");
    }

    out.write("GIF89a", 6);
    gif_detail::put16(out, static_cast<uint16_t>(width));
    gif_detail::put16(out, static_cast<uint16_t>(height));

    // A global table of 256 entries, of which 216 are the cube and the rest are
    // black: the size field is a power of two and 216 is not.
    out.put(static_cast<char>(0xF7));  // global table, 8 bits a pixel
    out.put(0);                        // background is entry zero
    out.put(0);                        // no aspect ratio
    for (uint32_t i = 0; i < 256; ++i) {
        const uint32_t r = i / (GIF_LEVELS * GIF_LEVELS);
        const uint32_t g = (i / GIF_LEVELS) % GIF_LEVELS;
        const uint32_t b = i % GIF_LEVELS;
        const bool real = i < GIF_COLOURS;
        out.put(static_cast<char>(real ? r * 255 / (GIF_LEVELS - 1) : 0));
        out.put(static_cast<char>(real ? g * 255 / (GIF_LEVELS - 1) : 0));
        out.put(static_cast<char>(real ? b * 255 / (GIF_LEVELS - 1) : 0));
    }

    // Netscape's looping extension, which is how a GIF says "for ever".
    out.write("\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00", 19);

    for (const std::vector<Float3>& frame : frames) {
        if (frame.size() != static_cast<size_t>(width) * height) {
            throw std::runtime_error("write_gif: a frame of the wrong size");
        }

        out.write("\x21\xF9\x04\x00", 4);  // graphic control, no disposal
        gif_detail::put16(out, static_cast<uint16_t>(delay_centiseconds));
        out.put(0);  // no transparent colour
        out.put(0);

        out.put(0x2C);  // image descriptor
        gif_detail::put16(out, 0);
        gif_detail::put16(out, 0);
        gif_detail::put16(out, static_cast<uint16_t>(width));
        gif_detail::put16(out, static_cast<uint16_t>(height));
        out.put(0);  // no local table, not interlaced

        std::vector<uint8_t> indices;
        indices.reserve(frame.size());
        for (const Float3& pixel : frame) {
            const uint32_t r = gif_detail::level_of(pixel.x);
            const uint32_t g = gif_detail::level_of(pixel.y);
            const uint32_t b = gif_detail::level_of(pixel.z);
            indices.push_back(
                static_cast<uint8_t>(r * GIF_LEVELS * GIF_LEVELS + g * GIF_LEVELS + b));
        }

        out.put(8);  // code size: the table is 256 entries
        gif_detail::write_blocks(out, gif_detail::compress(indices, 8));
    }

    out.put(0x3B);  // trailer
}
