#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

uint32_t Mesh::vertex_count() const
{
    return static_cast<uint32_t>(vertices.size());
}

uint32_t Mesh::triangle_count() const
{
    return static_cast<uint32_t>(indices.size() / 3);
}

std::vector<Float3> Mesh::flattened() const
{
    std::vector<Float3> out;
    out.reserve(indices.size());
    for (uint32_t i : indices) {
        out.push_back(vertices[i]);
    }
    return out;
}

Mesh cube_mesh()
{
    Mesh mesh;
    mesh.vertices = {
        Float3{-0.5f, -0.5f, -0.5f}, Float3{0.5f, -0.5f, -0.5f},
        Float3{0.5f, 0.5f, -0.5f},   Float3{-0.5f, 0.5f, -0.5f},
        Float3{-0.5f, -0.5f, 0.5f},  Float3{0.5f, -0.5f, 0.5f},
        Float3{0.5f, 0.5f, 0.5f},    Float3{-0.5f, 0.5f, 0.5f},
    };

    // Each face as a quad fanned from its first corner, which is exactly what
    // load_obj does to the quads in assets/cube.obj — so the two agree triangle
    // for triangle and the round trip is a real comparison.
    // clang-format off
    const uint32_t faces[6][4] = {
        {4, 5, 6, 7},  // +z
        {1, 0, 3, 2},  // -z
        {5, 1, 2, 6},  // +x
        {0, 4, 7, 3},  // -x
        {3, 7, 6, 2},  // +y
        {0, 1, 5, 4},  // -y
    };
    // clang-format on

    for (const uint32_t* face : faces) {
        mesh.indices.insert(mesh.indices.end(), {face[0], face[1], face[2]});
        mesh.indices.insert(mesh.indices.end(), {face[0], face[2], face[3]});
    }
    return mesh;
}

namespace {

// One vertex of a face: `1`, `1/1`, `1//1` or `1/1/1`. Only the position is
// read; the texture and normal indices are the fields this loader skips.
//
// so_far is how many v lines have been seen, which a negative index counts back
// from — the format allows it and it is the one part of index handling a reader
// gets wrong silently.
uint32_t parse_face_index(const std::string& spec, size_t so_far, const std::string& line)
{
    const std::string first = spec.substr(0, spec.find('/'));
    if (first.empty()) {
        throw std::runtime_error("load_obj: face vertex with no position index: " + line);
    }

    long value = 0;
    try {
        size_t consumed = 0;
        value = std::stol(first, &consumed);
        if (consumed != first.size()) {
            throw std::invalid_argument("trailing");
        }
    } catch (const std::exception&) {
        throw std::runtime_error("load_obj: face index is not a number: " + line);
    }

    const long resolved = (value < 0) ? static_cast<long>(so_far) + value : value - 1;
    if (resolved < 0 || resolved >= static_cast<long>(so_far)) {
        throw std::runtime_error("load_obj: face index " + std::to_string(value) +
                                 " is outside the " + std::to_string(so_far) +
                                 " vertices seen so far: " + line);
    }
    return static_cast<uint32_t>(resolved);
}

}  // namespace

Mesh load_obj(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("load_obj: cannot open " + path);
    }

    Mesh mesh;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        std::string tag;
        if (!(in >> tag) || tag[0] == '#') {
            continue;
        }

        if (tag == "v") {
            Float3 v;
            if (!(in >> v.x >> v.y >> v.z)) {
                throw std::runtime_error("load_obj: vertex needs three numbers: " + line);
            }
            mesh.vertices.push_back(v);
            continue;
        }

        if (tag != "f") {
            // vt, vn, mtllib, usemtl, o, g, s: material and grouping, not shape.
            continue;
        }

        std::vector<uint32_t> face;
        std::string spec;
        while (in >> spec) {
            face.push_back(parse_face_index(spec, mesh.vertices.size(), line));
        }
        if (face.size() < 3) {
            throw std::runtime_error("load_obj: face with fewer than three vertices: " +
                                     line);
        }

        for (size_t i = 1; i + 1 < face.size(); ++i) {
            mesh.indices.insert(mesh.indices.end(), {face[0], face[i], face[i + 1]});
        }
    }

    if (mesh.indices.empty()) {
        throw std::runtime_error("load_obj: " + path + " holds no faces");
    }
    return mesh;
}

namespace {

// The size Forsyth's constants were tuned against. Kept apart from the
// cache_size a caller asks for: the scoring curve is his and does not follow a
// different ring size, which is a limitation of borrowing tuned numbers.
constexpr float FORSYTH_CACHE = 32.0f;

}  // namespace

uint32_t simulated_cache_misses(const Mesh& mesh, uint32_t cache_size)
{
    if (cache_size == 0) {
        throw std::runtime_error("simulated_cache_misses: a cache of no entries");
    }

    // A ring of the most recently transformed vertices, oldest overwritten
    // first. Linear search because the sizes real hardware uses are 16 to 32,
    // where scanning beats anything with an index behind it.
    std::vector<uint32_t> fifo;
    fifo.reserve(cache_size);
    size_t oldest = 0;

    uint32_t misses = 0;
    for (uint32_t index : mesh.indices) {
        if (std::find(fifo.begin(), fifo.end(), index) != fifo.end()) {
            continue;
        }
        ++misses;
        if (fifo.size() < cache_size) {
            fifo.push_back(index);
        } else {
            fifo[oldest] = index;
            oldest = (oldest + 1) % fifo.size();
        }
    }
    return misses;
}

Mesh shuffled(const Mesh& mesh, uint32_t seed)
{
    std::vector<uint32_t> order(mesh.triangle_count());
    for (uint32_t t = 0; t < order.size(); ++t) {
        order[t] = t;
    }

    // mt19937 rather than rand(), so the same seed gives the same order on any
    // machine — a benchmark that shuffles differently per run measures nothing.
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);

    Mesh out;
    out.vertices = mesh.vertices;
    out.indices.reserve(mesh.indices.size());
    for (uint32_t t : order) {
        out.indices.push_back(mesh.indices[t * 3 + 0]);
        out.indices.push_back(mesh.indices[t * 3 + 1]);
        out.indices.push_back(mesh.indices[t * 3 + 2]);
    }
    return out;
}

namespace {

// Forsyth's scoring, from "Linear-Speed Vertex Cache Optimisation" (2006).
//
// The constants are his: tuned rather than derived, and the shape matters more
// than the values. The first three entries are deliberately held equal — a
// steeper curve there would drag the next choice back onto the triangle just
// emitted instead of moving along the surface.
float vertex_score(int cache_position, uint32_t triangles_left)
{
    if (triangles_left == 0) {
        return -1.0f;
    }

    float score = 0.0f;
    if (cache_position >= 0) {
        if (cache_position < 3) {
            score = 0.75f;
        } else {
            const float fraction =
                1.0f - static_cast<float>(cache_position - 3) / (FORSYTH_CACHE - 3);
            score = std::pow(fraction, 1.5f);
        }
    }

    // Rarely-used vertices first, so the common ones stay resident for longer.
    return score + 2.0f * std::pow(static_cast<float>(triangles_left), -0.5f);
}

}  // namespace

Mesh optimised_for_cache(const Mesh& mesh, uint32_t cache_size)
{
    if (cache_size < 4) {
        throw std::runtime_error("optimised_for_cache: a cache of fewer than four");
    }
    const uint32_t triangles = mesh.triangle_count();
    if (triangles == 0) {
        return mesh;
    }

    // Which triangles each vertex belongs to, and how many of them are still
    // unemitted. Both are what the score reads.
    std::vector<std::vector<uint32_t>> using_vertex(mesh.vertex_count());
    for (uint32_t t = 0; t < triangles; ++t) {
        for (uint32_t k = 0; k < 3; ++k) {
            using_vertex[mesh.indices[t * 3 + k]].push_back(t);
        }
    }
    std::vector<uint32_t> remaining(mesh.vertex_count());
    for (uint32_t v = 0; v < mesh.vertex_count(); ++v) {
        remaining[v] = static_cast<uint32_t>(using_vertex[v].size());
    }

    // position[v] is where v sits in the cache, or -1. Kept alongside the ring
    // so scoring a triangle is three lookups rather than three searches.
    std::vector<int> position(mesh.vertex_count(), -1);
    std::vector<uint32_t> cache;
    cache.reserve(cache_size + 3);

    std::vector<bool> emitted(triangles, false);
    std::vector<uint32_t> order;
    order.reserve(triangles);

    for (uint32_t done = 0; done < triangles; ++done) {
        // Only triangles touching a cached vertex can score above a cold one,
        // so the search is over those rather than over everything left. That is
        // what makes the whole pass linear in the mesh.
        uint32_t best = UINT32_MAX;
        float best_score = -1.0f;
        for (uint32_t v : cache) {
            for (uint32_t t : using_vertex[v]) {
                if (emitted[t]) {
                    continue;
                }
                float score = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) {
                    const uint32_t vertex = mesh.indices[t * 3 + k];
                    score += vertex_score(position[vertex], remaining[vertex]);
                }
                if (score > best_score) {
                    best_score = score;
                    best = t;
                }
            }
        }

        // Nothing adjacent is left — the mesh has more than one shell, or the
        // cache has just been emptied. Starting from the lowest unemitted index
        // keeps the result the same on every run.
        if (best == UINT32_MAX) {
            for (uint32_t t = 0; t < triangles; ++t) {
                if (!emitted[t]) {
                    best = t;
                    break;
                }
            }
        }

        emitted[best] = true;
        order.push_back(best);

        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t v = mesh.indices[best * 3 + k];
            --remaining[v];

            // To the front, as a use does on any LRU-ish ring.
            const auto found = std::find(cache.begin(), cache.end(), v);
            if (found != cache.end()) {
                cache.erase(found);
            }
            cache.insert(cache.begin(), v);
        }
        while (cache.size() > cache_size) {
            position[cache.back()] = -1;
            cache.pop_back();
        }
        for (size_t i = 0; i < cache.size(); ++i) {
            position[cache[i]] = static_cast<int>(i);
        }
    }

    Mesh out;
    out.vertices = mesh.vertices;
    out.indices.reserve(mesh.indices.size());
    for (uint32_t t : order) {
        out.indices.push_back(mesh.indices[t * 3 + 0]);
        out.indices.push_back(mesh.indices[t * 3 + 1]);
        out.indices.push_back(mesh.indices[t * 3 + 2]);
    }
    return out;
}
