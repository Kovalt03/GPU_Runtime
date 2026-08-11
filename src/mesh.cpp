#include "mesh.hpp"

#include <cstdint>
#include <fstream>
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
