#pragma once

#include <algorithm>
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

// The palette is built from the frames rather than fixed.
//
// A 216-colour cube was tried first, on the reasoning that a quantiser is a
// second algorithm to be wrong in. It is wrong about exactly this content: six
// levels a channel turns a base colour of (0.85, 0.78, 0.70) into (4, 4, 4),
// which is grey. Smooth shading over one hue is the case a fixed cube handles
// worst, because it spends its levels on hues that are not there.
//
// Median cut, which is the standard answer and about sixty lines: split the box
// holding every colour along its longest axis at the median, repeat on whichever
// box holds the most, and average each box at the end.
inline constexpr uint32_t GIF_COLOURS = 256;

namespace gif_detail {

struct Rgb {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

inline uint8_t byte_of(float channel)
{
    const float clamped = channel < 0.0f ? 0.0f : (channel > 1.0f ? 1.0f : channel);
    return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
}

// One box of the median cut: a run of the colour list, and where it reaches.
struct Box {
    size_t begin = 0;
    size_t end = 0;
    Rgb lo;
    Rgb hi;

    size_t count() const
    {
        return end - begin;
    }

    uint32_t longest_axis() const
    {
        const uint32_t r = static_cast<uint32_t>(hi.r - lo.r);
        const uint32_t g = static_cast<uint32_t>(hi.g - lo.g);
        const uint32_t b = static_cast<uint32_t>(hi.b - lo.b);
        return r >= g && r >= b ? 0u : (g >= b ? 1u : 2u);
    }

    uint32_t spread() const
    {
        const uint32_t r = static_cast<uint32_t>(hi.r - lo.r);
        const uint32_t g = static_cast<uint32_t>(hi.g - lo.g);
        const uint32_t b = static_cast<uint32_t>(hi.b - lo.b);
        return std::max(r, std::max(g, b));
    }
};

inline uint8_t channel_of(const Rgb& c, uint32_t axis)
{
    return axis == 0 ? c.r : (axis == 1 ? c.g : c.b);
}

inline void fit(Box& box, const std::vector<Rgb>& colours)
{
    box.lo = Rgb{255, 255, 255};
    box.hi = Rgb{0, 0, 0};
    for (size_t i = box.begin; i < box.end; ++i) {
        const Rgb& c = colours[i];
        box.lo = Rgb{std::min(box.lo.r, c.r), std::min(box.lo.g, c.g),
                     std::min(box.lo.b, c.b)};
        box.hi = Rgb{std::max(box.hi.r, c.r), std::max(box.hi.g, c.g),
                     std::max(box.hi.b, c.b)};
    }
}

// Splits the widest box until there are as many as asked for, then averages
// each. Splitting the widest rather than the most populous is what keeps a small
// bright highlight from being merged into the mass of mid-tones around it.
inline std::vector<Rgb> median_cut(std::vector<Rgb> colours, uint32_t wanted)
{
    if (colours.empty()) {
        return {Rgb{}};
    }

    std::vector<Box> boxes;
    Box all{0, colours.size(), {}, {}};
    fit(all, colours);
    boxes.push_back(all);

    while (boxes.size() < wanted) {
        size_t widest = boxes.size();
        uint32_t best = 0;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (boxes[i].count() > 1 && boxes[i].spread() > best) {
                best = boxes[i].spread();
                widest = i;
            }
        }
        if (widest == boxes.size()) {
            break;  // every box is one colour
        }

        Box& box = boxes[widest];
        const uint32_t axis = box.longest_axis();
        const size_t middle = box.begin + box.count() / 2;
        std::nth_element(colours.begin() + static_cast<long>(box.begin),
                         colours.begin() + static_cast<long>(middle),
                         colours.begin() + static_cast<long>(box.end),
                         [axis](const Rgb& a, const Rgb& b) {
                             return channel_of(a, axis) < channel_of(b, axis);
                         });

        Box upper{middle, box.end, {}, {}};
        box.end = middle;
        fit(box, colours);
        fit(upper, colours);
        boxes.push_back(upper);
    }

    std::vector<Rgb> palette;
    palette.reserve(boxes.size());
    for (const Box& box : boxes) {
        uint64_t r = 0;
        uint64_t g = 0;
        uint64_t b = 0;
        for (size_t i = box.begin; i < box.end; ++i) {
            r += colours[i].r;
            g += colours[i].g;
            b += colours[i].b;
        }
        const uint64_t n = std::max<uint64_t>(1, box.count());
        palette.push_back(Rgb{static_cast<uint8_t>(r / n), static_cast<uint8_t>(g / n),
                              static_cast<uint8_t>(b / n)});
    }
    return palette;
}

// Nearest by squared distance. Linear over 256 entries a pixel, which is the
// cost of not building an octree — an animation here is a few hundred thousand
// pixels and this is not what takes the time.
inline uint8_t nearest(const std::vector<Rgb>& palette, const Rgb& want)
{
    uint32_t best = 0;
    int32_t closest = 1 << 30;
    for (uint32_t i = 0; i < palette.size(); ++i) {
        const int32_t dr = static_cast<int32_t>(palette[i].r) - want.r;
        const int32_t dg = static_cast<int32_t>(palette[i].g) - want.g;
        const int32_t db = static_cast<int32_t>(palette[i].b) - want.b;
        const int32_t d = dr * dr + dg * dg + db * db;
        if (d < closest) {
            closest = d;
            best = i;
        }
    }
    return static_cast<uint8_t>(best);
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

    // Every colour in the animation, sampled if there are a great many: median
    // cut reads the list rather than a histogram, and a hundred thousand entries
    // says the same thing as ten thousand about where the boxes go.
    const size_t total = frames.size() * frames.front().size();
    const size_t stride = std::max<size_t>(1, total / 60000);
    std::vector<gif_detail::Rgb> sample;
    sample.reserve(total / stride + 1);
    for (size_t f = 0; f < frames.size(); ++f) {
        for (size_t i = 0; i < frames[f].size(); i += stride) {
            const Float3& p = frames[f][i];
            sample.push_back(gif_detail::Rgb{gif_detail::byte_of(p.x),
                                             gif_detail::byte_of(p.y),
                                             gif_detail::byte_of(p.z)});
        }
    }
    const std::vector<gif_detail::Rgb> palette =
        gif_detail::median_cut(std::move(sample), GIF_COLOURS);

    out.write("GIF89a", 6);
    gif_detail::put16(out, static_cast<uint16_t>(width));
    gif_detail::put16(out, static_cast<uint16_t>(height));

    out.put(static_cast<char>(0xF7));  // global table of 256, 8 bits a pixel
    out.put(0);                        // background is entry zero
    out.put(0);                        // no aspect ratio
    for (uint32_t i = 0; i < 256; ++i) {
        const gif_detail::Rgb c = i < palette.size() ? palette[i] : gif_detail::Rgb{};
        out.put(static_cast<char>(c.r));
        out.put(static_cast<char>(c.g));
        out.put(static_cast<char>(c.b));
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
            indices.push_back(gif_detail::nearest(
                palette, gif_detail::Rgb{gif_detail::byte_of(pixel.x),
                                         gif_detail::byte_of(pixel.y),
                                         gif_detail::byte_of(pixel.z)}));
        }

        out.put(8);  // code size: the table is 256 entries
        gif_detail::write_blocks(out, gif_detail::compress(indices, 8));
    }

    out.put(0x3B);  // trailer
}
