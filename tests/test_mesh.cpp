#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "math3d.hpp"
#include "mesh.hpp"

namespace {

// Baked in at configure time: ctest runs the binaries from the build tree,
// where a path relative to the source would miss.
const std::string ASSETS = GPURT_ASSETS_DIR;

// Writes an OBJ so a parser case can be stated inline rather than as another
// committed file.
//
// Into the system temp directory, not the working one: ctest runs from the
// build tree but a developer running the binary by hand runs from wherever
// they are, and a relative path would leave a dozen .obj files there.
std::string write_temp_obj(const std::string& name, const std::string& body)
{
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream out(path);
    out << body;
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    return path;
}

// Every face of a closed convex solid around the origin points away from it, so
// this catches a triangle wound the wrong way without naming which one is which.
bool faces_outward(const Mesh& mesh)
{
    for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
        const Float3 a = mesh.vertices[mesh.indices[t * 3 + 0]];
        const Float3 b = mesh.vertices[mesh.indices[t * 3 + 1]];
        const Float3 c = mesh.vertices[mesh.indices[t * 3 + 2]];
        const Float3 normal = cross(b - a, c - a);
        const Float3 centre = (a + b + c) * (1.0f / 3.0f);
        if (dot(normal, centre) <= 0.0f) {
            return false;
        }
    }
    return true;
}

// Triangles as sorted vertex triples, so two meshes can be compared without
// caring which order the faces came out in.
std::vector<std::array<uint32_t, 3>> sorted_triangles(const Mesh& mesh)
{
    std::vector<std::array<uint32_t, 3>> out;
    for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
        std::array<uint32_t, 3> tri{mesh.indices[t * 3 + 0], mesh.indices[t * 3 + 1],
                                    mesh.indices[t * 3 + 2]};
        std::sort(tri.begin(), tri.end());
        out.push_back(tri);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

TEST(Mesh, CubeHasEightVerticesAndTwelveTriangles)
{
    const Mesh cube = cube_mesh();

    // The whole point of the indexed form: a cube has eight corners, and the
    // flattened list it replaces would have thirty-six.
    EXPECT_EQ(cube.vertex_count(), 8u);
    EXPECT_EQ(cube.triangle_count(), 12u);
    EXPECT_EQ(cube.indices.size(), 36u);
    EXPECT_EQ(cube.flattened().size(), 36u);
}

TEST(Mesh, CubeFacesAreWoundOutward)
{
    EXPECT_TRUE(faces_outward(cube_mesh()));
}

TEST(Mesh, FlatteningResolvesEveryIndex)
{
    const Mesh cube = cube_mesh();
    const std::vector<Float3> flat = cube.flattened();

    ASSERT_EQ(flat.size(), cube.indices.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        const Float3 want = cube.vertices[cube.indices[i]];
        EXPECT_FLOAT_EQ(flat[i].x, want.x) << "vertex " << i;
        EXPECT_FLOAT_EQ(flat[i].y, want.y) << "vertex " << i;
        EXPECT_FLOAT_EQ(flat[i].z, want.z) << "vertex " << i;
    }
}

TEST(Mesh, LoadsTheCubeAssetAsCubeMesh)
{
    // The asset stores quads. That they fan into the same twelve triangles is
    // what makes this a check on the loader rather than on itself.
    const Mesh loaded = load_obj(ASSETS + "/cube.obj");
    const Mesh built = cube_mesh();

    ASSERT_EQ(loaded.vertex_count(), built.vertex_count());
    for (uint32_t i = 0; i < built.vertex_count(); ++i) {
        EXPECT_FLOAT_EQ(loaded.vertices[i].x, built.vertices[i].x) << "vertex " << i;
        EXPECT_FLOAT_EQ(loaded.vertices[i].y, built.vertices[i].y) << "vertex " << i;
        EXPECT_FLOAT_EQ(loaded.vertices[i].z, built.vertices[i].z) << "vertex " << i;
    }
    EXPECT_EQ(loaded.indices, built.indices);
    EXPECT_TRUE(faces_outward(loaded));
}

TEST(Mesh, AcceptsAllFourFaceSpellings)
{
    const std::string vertices = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
    const std::string spellings[] = {"f 1 2 3\n", "f 1/1 2/2 3/3\n", "f 1//1 2//2 3//3\n",
                                     "f 1/1/1 2/2/2 3/3/3\n"};

    for (const std::string& face : spellings) {
        const Mesh mesh = load_obj(write_temp_obj("spelling.obj", vertices + face));
        EXPECT_EQ(mesh.triangle_count(), 1u) << face;
        EXPECT_EQ(mesh.indices, (std::vector<uint32_t>{0, 1, 2})) << face;
    }
}

TEST(Mesh, SkipsWhatIsNotShape)
{
    // Material and grouping lines are ignored rather than refused, or no OBJ
    // exported by a real tool would load.
    const std::string body =
        "# a comment\n"
        "mtllib thing.mtl\n"
        "o cube\n"
        "g group\n"
        "s off\n"
        "usemtl red\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvn 0 0 1\n"
        "f 1 2 3\n";

    const Mesh mesh = load_obj(write_temp_obj("skipped.obj", body));
    EXPECT_EQ(mesh.vertex_count(), 3u);
    EXPECT_EQ(mesh.triangle_count(), 1u);
}

TEST(Mesh, FansAFaceOfMoreThanThreeVertices)
{
    const std::string quad = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
    const Mesh mesh = load_obj(write_temp_obj("quad.obj", quad));

    EXPECT_EQ(mesh.triangle_count(), 2u);
    EXPECT_EQ(mesh.indices, (std::vector<uint32_t>{0, 1, 2, 0, 2, 3}));

    const std::string pentagon =
        "v 0 0 0\nv 1 0 0\nv 2 1 0\nv 1 2 0\nv 0 2 0\nf 1 2 3 4 5\n";
    EXPECT_EQ(load_obj(write_temp_obj("pentagon.obj", pentagon)).triangle_count(), 3u);
}

TEST(Mesh, CountsNegativeIndicesBackFromTheEnd)
{
    // -1 is the most recent vertex, so this names the same triangle as `f 1 2 3`.
    const std::string body = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
    const Mesh mesh = load_obj(write_temp_obj("negative.obj", body));

    EXPECT_EQ(mesh.indices, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(Mesh, RefusesAnIndexOutsideTheVertexList)
{
    const std::string body = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n";
    EXPECT_THROW(load_obj(write_temp_obj("range.obj", body)), std::runtime_error);

    const std::string negative = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n";
    EXPECT_THROW(load_obj(write_temp_obj("negrange.obj", negative)), std::runtime_error);
}

TEST(Mesh, RefusesAFaceThatIsNotATriangle)
{
    const std::string body = "v 0 0 0\nv 1 0 0\nf 1 2\n";
    EXPECT_THROW(load_obj(write_temp_obj("twogon.obj", body)), std::runtime_error);
}

TEST(Mesh, RefusesAMalformedLine)
{
    const std::string short_vertex = "v 0 0\nf 1 1 1\n";
    EXPECT_THROW(load_obj(write_temp_obj("shortv.obj", short_vertex)),
                 std::runtime_error);

    const std::string not_a_number = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 two 3\n";
    EXPECT_THROW(load_obj(write_temp_obj("nan.obj", not_a_number)), std::runtime_error);
}

TEST(Mesh, RefusesAMissingFileAndAnEmptyOne)
{
    EXPECT_THROW(load_obj(ASSETS + "/there-is-no-such-mesh.obj"), std::runtime_error);

    const std::string vertices_only = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
    EXPECT_THROW(load_obj(write_temp_obj("nofaces.obj", vertices_only)),
                 std::runtime_error);
}

TEST(Mesh, TriangleSetSurvivesTheAssetRoundTrip)
{
    // Compared as sorted triples, which is the check that survives a loader
    // choosing a different face order later.
    EXPECT_EQ(sorted_triangles(load_obj(ASSETS + "/cube.obj")),
              sorted_triangles(cube_mesh()));
}
