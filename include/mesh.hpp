#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "math3d.hpp"

// Geometry on its way in. Host code with no ISA in it, which is why it sits
// beside math3d.hpp rather than under pipeline/ — a mesh is an input to the
// graphics layer the way a camera is, not a stage of it.

// Unique positions, and three indices per triangle.
//
// The indexed form is what OBJ already carries and what the vertex stage wants:
// pass 1 runs a thread per vertex, so a corner shared by six triangles is six
// transforms if the triangles are flattened first and one if they are not.
// Modelling that saving is what the Input Assembler exists to do on real
// hardware.
struct Mesh {
    std::vector<Float3> vertices;
    std::vector<uint32_t> indices;

    uint32_t vertex_count() const;
    uint32_t triangle_count() const;

    // Three vertices per triangle, indices resolved — the form the draw routes
    // still take, and what bin_triangles would have built anyway.
    std::vector<Float3> flattened() const;
};

// A unit cube at the origin, eight vertices and twelve triangles, each face
// wound counter-clockwise as seen from outside.
//
// Built in code so the mesh path can be exercised with no parser in the way:
// anything downstream that goes wrong is then downstream's fault.
Mesh cube_mesh();

// Reads the v and f lines of a Wavefront OBJ.
//
// Everything else — vt, vn, mtllib, usemtl, o, g, s — is skipped rather than
// rejected, being about material and grouping rather than shape. All four face
// spellings are accepted: `1`, `1/1`, `1//1`, `1/1/1`. Negative indices count
// back from the end, as the format allows.
//
// A face of more than three vertices is fanned from its first vertex. That is
// right for the convex faces an OBJ normally carries and visibly wrong for a
// concave one, which is the trade every loader of this size makes.
//
// Throws std::runtime_error on a missing file, a malformed line, or an index
// outside the vertex list. A loader that dropped a face instead would render as
// a hole, and a hole reads as a bug in the rasteriser.
Mesh load_obj(const std::string& path);

// --- vertex cache ------------------------------------------------------------
// What a fixed-function Input Assembler would have paid for this mesh, and what
// reordering its triangles would have saved it.
//
// A fixed-function IA streams vertices past a FIFO of the last sixteen or
// thirty-two and re-transforms any that has been evicted before its next use,
// so the order triangles arrive in is the order it pays for. Pass 1 here
// materialises every unique vertex into the screen buffer and reads it back by
// index, so a vertex is transformed once whatever the order.
//
// That is why the functions below measure a machine we are not: the difference
// between what they report and what our counters report is what materialising
// bought. None of them changes a rendered frame — vertices and the set of
// triangles are untouched in all three, and only the order moves.

// How many vertex transforms a FIFO cache of this size would have needed.
//
// Divided by triangle_count() this is ACMR, the ratio the literature quotes.
// Its floor is vertex_count() — a cache larger than the mesh misses once per
// vertex, which is what materialising achieves unconditionally. Its ceiling is
// three per triangle, nothing being reused at all.
uint32_t simulated_cache_misses(const Mesh& mesh, uint32_t cache_size = 32);

// Triangles in a deliberately bad order, from a seed so that two runs agree.
Mesh shuffled(const Mesh& mesh, uint32_t seed);

// Triangles in a good one, by Forsyth's greedy score: a vertex is worth having
// while it is still in the cache and worth finishing with while few triangles
// still need it, and the triangle scoring highest on the sum of its three goes
// next.
//
// Worth a factor of three or four on a fixed-function IA. Here it moves the
// issued work by a thousandth of a percent, and even that much is depth
// ordering rather than vertex reuse — nearest-wins fires a different number of
// times when the triangles covering a pixel arrive in a different order. The
// tests hold both halves of that.
Mesh optimised_for_cache(const Mesh& mesh, uint32_t cache_size = 32);
