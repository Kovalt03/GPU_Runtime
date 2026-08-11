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
