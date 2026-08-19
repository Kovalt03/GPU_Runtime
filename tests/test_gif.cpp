#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../kernels/gif.hpp"

namespace {

std::string scratch(const std::string& name)
{
    return std::string(GPURT_BUILD_DIR) + "/" + name;
}

std::vector<uint8_t> bytes_of(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>());
}

// The decoder the encoder is checked against. Written from the format rather
// than from the encoder, which is the only way agreeing means anything — an
// encoder held against its own inverse agrees with itself however wrong it is.
std::vector<std::vector<uint8_t>> decode(const std::vector<uint8_t>& data)
{
    if (data.size() < 13 || std::string(data.begin(), data.begin() + 6) != "GIF89a") {
        throw std::runtime_error("not a GIF89a");
    }
    size_t at = 13 + static_cast<size_t>(2 << (data[10] & 7)) * 3;

    std::vector<std::vector<uint8_t>> frames;
    while (at < data.size()) {
        if (data[at] == 0x21) {
            at += data[at + 1] == 0xF9 ? 8 : 2;
            if (data[at - 2] == 0xFF || data[at - 1] == 0xFF) {
                while (at < data.size() && data[at] != 0) {
                    at += 1 + data[at];
                }
                ++at;
            }
            continue;
        }
        if (data[at] != 0x2C) {
            break;
        }

        at += 10;
        const uint32_t code_bits = data[at++];
        std::vector<uint8_t> raw;
        while (at < data.size() && data[at] != 0) {
            const uint8_t n = data[at];
            raw.insert(raw.end(), data.begin() + static_cast<long>(at) + 1,
                       data.begin() + static_cast<long>(at) + 1 + n);
            at += 1 + n;
        }
        ++at;

        const uint32_t clear = 1u << code_bits;
        const uint32_t end = clear + 1;
        std::vector<std::vector<uint8_t>> dictionary;
        const auto reset = [&] {
            dictionary.assign(clear + 2, {});
            for (uint32_t i = 0; i < clear; ++i) {
                dictionary[i] = {static_cast<uint8_t>(i)};
            }
        };
        reset();

        std::vector<uint8_t> pixels;
        uint32_t width = code_bits + 1;
        uint32_t buffer = 0;
        uint32_t bits = 0;
        bool have_previous = false;
        uint32_t previous = 0;
        for (size_t i = 0; i < raw.size(); ++i) {
            buffer |= static_cast<uint32_t>(raw[i]) << bits;
            bits += 8;
            while (bits >= width) {
                const uint32_t code = buffer & ((1u << width) - 1);
                buffer >>= width;
                bits -= width;

                if (code == clear) {
                    reset();
                    width = code_bits + 1;
                    have_previous = false;
                    continue;
                }
                if (code == end) {
                    i = raw.size();
                    bits = 0;
                    break;
                }

                std::vector<uint8_t> entry;
                if (code < dictionary.size() && !dictionary[code].empty()) {
                    entry = dictionary[code];
                } else if (have_previous) {
                    entry = dictionary[previous];
                    entry.push_back(entry.front());
                } else {
                    throw std::runtime_error("a code with nothing behind it");
                }
                pixels.insert(pixels.end(), entry.begin(), entry.end());

                if (have_previous && dictionary.size() < 4096) {
                    std::vector<uint8_t> made = dictionary[previous];
                    made.push_back(entry.front());
                    dictionary.push_back(made);
                    if (dictionary.size() >= (1u << width) && width < 12) {
                        ++width;
                    }
                }
                previous = code;
                have_previous = true;
            }
        }
        frames.push_back(pixels);
    }
    return frames;
}

}  // namespace

TEST(Gif, AnAnimationSurvivesBeingWrittenAndReadBack)
{
    // The whole of what the encoder has to get right, and the only check worth
    // having: the pixels that come out are the pixels that went in. LZW is the
    // sort of code that produces a plausible file while being wrong.
    constexpr uint32_t WIDTH = 37;  // not a multiple of anything, on purpose
    constexpr uint32_t HEIGHT = 11;

    std::vector<std::vector<Float3>> animation;
    for (uint32_t frame = 0; frame < 3; ++frame) {
        std::vector<Float3> pixels;
        for (uint32_t y = 0; y < HEIGHT; ++y) {
            for (uint32_t x = 0; x < WIDTH; ++x) {
                // A run of one colour, then something that changes every pixel:
                // the first compresses and the second does not, and an encoder
                // can be wrong about either alone.
                const bool banded = x < WIDTH / 2;
                pixels.push_back(banded
                                     ? Float3{0.2f, 0.4f, 0.6f}
                                     : Float3{static_cast<float>((x + frame) % 6) / 5.0f,
                                              static_cast<float>(y % 6) / 5.0f,
                                              static_cast<float>((x * y) % 6) / 5.0f});
            }
        }
        animation.push_back(pixels);
    }

    const std::string path = scratch("round_trip.gif");
    write_gif(path, animation, WIDTH, HEIGHT);

    const std::vector<std::vector<uint8_t>> read = decode(bytes_of(path));
    ASSERT_EQ(read.size(), animation.size());
    for (size_t f = 0; f < animation.size(); ++f) {
        ASSERT_EQ(read[f].size(), static_cast<size_t>(WIDTH) * HEIGHT) << "frame " << f;
        for (size_t i = 0; i < read[f].size(); ++i) {
            const Float3& wanted = animation[f][i];
            const uint32_t expected =
                gif_detail::level_of(wanted.x) * GIF_LEVELS * GIF_LEVELS +
                gif_detail::level_of(wanted.y) * GIF_LEVELS +
                gif_detail::level_of(wanted.z);
            ASSERT_EQ(read[f][i], expected) << "frame " << f << ", pixel " << i;
        }
    }
}

TEST(Gif, ALongRunReachesTheDictionaryCeilingAndCarriesOn)
{
    // Twelve bits is four thousand entries, and an encoder that stopped adding
    // rather than clearing would emit twelve bits a pixel from there on — the
    // file would still decode, which is why this checks the pixels rather than
    // the size.
    constexpr uint32_t WIDTH = 200;
    constexpr uint32_t HEIGHT = 200;

    std::vector<Float3> pixels;
    for (uint32_t i = 0; i < WIDTH * HEIGHT; ++i) {
        // A sequence with no short repeats, so the dictionary fills.
        const uint32_t v = (i * 2654435761u) >> 24;
        pixels.push_back(Float3{static_cast<float>(v % 6) / 5.0f,
                                static_cast<float>((v / 6) % 6) / 5.0f,
                                static_cast<float>((v / 36) % 6) / 5.0f});
    }

    const std::string path = scratch("ceiling.gif");
    write_gif(path, {pixels}, WIDTH, HEIGHT);

    const std::vector<std::vector<uint8_t>> read = decode(bytes_of(path));
    ASSERT_EQ(read.size(), 1u);
    ASSERT_EQ(read[0].size(), static_cast<size_t>(WIDTH) * HEIGHT);
    for (size_t i = 0; i < read[0].size(); ++i) {
        const uint32_t expected =
            gif_detail::level_of(pixels[i].x) * GIF_LEVELS * GIF_LEVELS +
            gif_detail::level_of(pixels[i].y) * GIF_LEVELS +
            gif_detail::level_of(pixels[i].z);
        ASSERT_EQ(read[0][i], expected) << "pixel " << i;
    }
}

TEST(Gif, AnAnimationItCannotWriteIsRefused)
{
    EXPECT_THROW(write_gif(scratch("empty.gif"), {}, 4, 4), std::runtime_error);
    EXPECT_THROW(write_gif(scratch("short.gif"), {{Float3{}}}, 4, 4), std::runtime_error);
    EXPECT_THROW(write_gif("nowhere/at/all.gif", {{Float3{}}}, 1, 1), std::runtime_error);
}
