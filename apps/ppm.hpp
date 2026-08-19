#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Shared by the demos, which both end by writing a frame out. Plain ASCII P3:
// the point is that any viewer opens it and the file can be diffed, not that it
// is compact.
inline void write_ppm(const std::string& path, const std::vector<float>& rgb,
                      uint32_t width, uint32_t height)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open " + path + " for writing");
    }

    out << "P3\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = std::clamp(rgb[i * 3 + c], 0.0f, 1.0f);
            out << static_cast<int>(v * 255.0f + 0.5f) << (c == 2 ? '\n' : ' ');
        }
    }
}
