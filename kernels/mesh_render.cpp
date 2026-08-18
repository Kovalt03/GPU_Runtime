// Meshes through every route, and through both storage forms.
//
// The triangle demos stay as they are: each holds one shape hard-coded, which
// is what makes them a fixed reference between the two renderers. What they
// cannot show is a vertex used more than once, and every question about
// indexing needs exactly that.
//
// The meshes in assets/ are chosen to spread one ratio — vertices per triangle,
// which is how much of the geometry is shared:
//
//   tetrahedron   4 / 4     1.00   the least a closed surface can share
//   cube          8 / 12    0.67
//   grid         81 / 128   0.63
//   sphere      182 / 360   0.51   near the limit Euler gives a closed mesh
//
//   ./build/kernels/mesh_render                        every mesh in assets/
//   ./build/kernels/mesh_render assets/sphere.obj      one of them
//   ./build/kernels/mesh_render --size 256 --out dir   slower, four times over
//
// Two tables. The first asks which route is cheapest for a mesh; the second
// asks what storing it indexed is worth, which is a saving in pass 1 and a cost
// in pass 2 and does not come out the same way for every shape.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "app_run.hpp"
#include "mesh.hpp"
#include "pipeline/draw.hpp"
#include "ppm.hpp"
#include "runtime.hpp"

namespace {

const char* const DEFAULT_MESHES[] = {"assets/tetrahedron.obj", "assets/cube.obj",
                                      "assets/grid.obj", "assets/sphere.obj"};

// Off-axis so more than one face shows and they land at different depths, which
// is what makes nearest-wins do anything.
//
// Close enough that a unit shape fills most of the frame: a mesh that covered a
// third of it would leave two thirds of the pixels measuring the bounds check
// rather than the coverage test.
Camera mesh_camera()
{
    Camera cam;
    cam.eye = Float3{1.25f, 0.95f, 1.8f};
    cam.target = Float3{0.0f, 0.0f, 0.0f};
    cam.up = Float3{0.0f, 1.0f, 0.0f};
    cam.fov_y_degrees = 60.0f;
    return cam;
}

std::string stem(const std::string& path)
{
    const size_t slash = path.find_last_of('/');
    const size_t start = (slash == std::string::npos) ? 0 : slash + 1;
    const size_t dot = path.find_last_of('.');
    return path.substr(start, (dot == std::string::npos) ? dot : dot - start);
}

struct Reading {
    uint64_t weighted = 0;
    double divergence = 0.0;
    double seconds = 0.0;
    std::vector<Float3> frame;
};

// A runtime per reading, so each is a total rather than a difference of two.
template <typename Route>
Reading measure(size_t budget, Route&& route)
{
    MyGPURuntime rt(budget);
    const Stopwatch watch;
    Reading r;
    r.frame = route(rt);
    r.seconds = watch.seconds();
    r.weighted = rt.stats().weighted_lane_ops;
    r.divergence = rt.divergence_rate();
    return r;
}

void write_frame(const std::string& path, const std::vector<Float3>& frame,
                 uint32_t width, uint32_t height)
{
    std::vector<float> flat;
    flat.reserve(frame.size() * 3);
    for (const Float3& p : frame) {
        flat.push_back(p.x);
        flat.push_back(p.y);
        flat.push_back(p.z);
    }
    write_ppm(path, flat, width, height);
}

double change(uint64_t from, uint64_t to)
{
    return 100.0 * (static_cast<double>(to) - static_cast<double>(from)) /
           static_cast<double>(from);
}

struct Row {
    std::string name;
    Mesh mesh;
    Reading walk, tiled, shared, ray;
    uint64_t pass1_indexed = 0;
    uint64_t pass1_steps_baked = 0;
    uint64_t pass1_steps_window = 0;
    uint64_t pass1_lanes_baked = 0;
    uint64_t pass1_lanes_window = 0;
    uint64_t pass1_flat = 0;
    uint64_t pass2_indexed = 0;
    uint64_t pass2_flat = 0;
    bool routes_agree = true;
};

}  // namespace

int run(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    // Four meshes through five measurements each, and the walk route is
    // O(pixels x triangles) with no binning to save it — the sphere alone is
    // most of the wall clock, at nine seconds against a quarter for the tiled
    // route on the same frame. 512 is four times this again.
    const uint32_t size = args.flag("size", 256);

    std::vector<std::string> paths = args.positional;
    if (paths.empty()) {
        for (const char* p : DEFAULT_MESHES) {
            paths.emplace_back(p);
        }
    }

    const DrawTarget target{size, size, mesh_camera()};
    const std::string images = args.images_dir();
    std::vector<Row> rows;

    for (const std::string& path : paths) {
        Row row;
        row.name = stem(path);
        try {
            row.mesh = load_obj(path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "mesh_render: %s\n", e.what());
            return 1;
        }

        const Mesh& mesh = row.mesh;
        const std::vector<Float3> flat = mesh.flattened();

        // Asked for rather than worked out here. Sizing a device by hand is
        // what sent this program looking for three megabytes it had not
        // reserved, twice — once for a framebuffer and once for a binning that
        // copies a triangle into every tile it reaches.
        //
        // The flattened list is the larger of the two forms, so planning for it
        // covers the indexed one as well.
        BufferPlan plan;
        plan.world_vertices = mesh.triangle_count() * 3;
        plan.screen_vertices = plan.world_vertices;
        plan.indices = mesh.triangle_count() * 3;
        plan.width = size;
        plan.height = size;
        const size_t budget =
            plan.device_bytes() +
            BufferPlan::binned_bytes(size, size, mesh.triangle_count()) + (1u << 20);

        Shading lit;
        lit.mode = ShadingMode::Diffuse;
        lit.light_position = Float3{3.0f, 4.0f, 2.0f};
        lit.base_colour = Float3{0.85f, 0.6f, 0.35f};

        row.walk = measure(budget,
                           [&](MyGPURuntime& rt) { return draw_walk(rt, mesh, target); });
        row.tiled = measure(
            budget, [&](MyGPURuntime& rt) { return draw_tiled(rt, mesh, target); });
        row.shared = measure(
            budget, [&](MyGPURuntime& rt) { return draw_shared(rt, mesh, target); });
        row.ray = measure(budget, [&](MyGPURuntime& rt) {
            return draw_raytrace(rt, mesh, target, lit);
        });

        // Storing the same geometry the other way. Pass 1 is where indexing
        // pays and pass 2 is where it charges, so both have to be read.
        row.pass1_indexed = vertex_stage_cost(mesh.vertices, target).weighted_lane_ops;
        row.pass1_flat = vertex_stage_cost(flat, target).weighted_lane_ops;

        // And the same pass with the matrix read from the constant window rather
        // than baked into the program. The frame is identical; what moves is the
        // sixteen moves a baked matrix costs every thread.
        const SchedulerStats baked = vertex_stage_cost(mesh.vertices, target);
        const SchedulerStats window =
            vertex_stage_cost(mesh.vertices, target, Uniforms::Window);
        row.pass1_steps_baked = baked.warp_steps;
        row.pass1_steps_window = window.warp_steps;
        row.pass1_lanes_baked = baked.active_lane_ops;
        row.pass1_lanes_window = window.active_lane_ops;
        row.pass2_indexed = row.walk.weighted;
        row.pass2_flat = measure(budget, [&](MyGPURuntime& rt) {
                             return draw_walk(rt, flat, target);
                         }).weighted;

        // The three rasteriser routes have to agree pixel for pixel; the ray
        // tracer is lit and deliberately does not.
        for (const Reading* other : {&row.tiled, &row.shared}) {
            for (size_t p = 0; p < row.walk.frame.size(); ++p) {
                if (other->frame[p].x != row.walk.frame[p].x ||
                    other->frame[p].y != row.walk.frame[p].y ||
                    other->frame[p].z != row.walk.frame[p].z) {
                    row.routes_agree = false;
                }
            }
        }

        for (const auto& [suffix, reading] :
             {std::pair<const char*, const Reading*>{"walk", &row.walk},
              {"tiled", &row.tiled},
              {"shared", &row.shared},
              {"raytrace", &row.ray}}) {
            write_frame(images + row.name + "_" + suffix + ".ppm", reading->frame, size,
                        size);
        }
        rows.push_back(std::move(row));
    }

    std::printf("\n[MESH] %u meshes at %ux%u\n\n", static_cast<uint32_t>(rows.size()),
                size, size);

    std::printf("  %-12s %6s %6s %6s %14s %14s %14s %14s %7s\n", "mesh", "verts", "tris",
                "v/t", "walk", "tiled", "shared", "raytrace", "agree");
    for (const Row& r : rows) {
        std::printf("  %-12s %6u %6u %6.2f %14llu %14llu %14llu %14llu %7s\n",
                    r.name.c_str(), r.mesh.vertex_count(), r.mesh.triangle_count(),
                    double(r.mesh.vertex_count()) / r.mesh.triangle_count(),
                    static_cast<unsigned long long>(r.walk.weighted),
                    static_cast<unsigned long long>(r.tiled.weighted),
                    static_cast<unsigned long long>(r.shared.weighted),
                    static_cast<unsigned long long>(r.ray.weighted),
                    r.routes_agree ? "yes" : "NO");
    }

    std::printf("\n  %-12s %6s | %12s %12s %8s | %14s %14s %8s\n", "indexed vs", "v/t",
                "pass1 idx", "pass1 flat", "change", "pass2 idx", "pass2 flat", "change");
    for (const Row& r : rows) {
        std::printf("  %-12s %6.2f | %12llu %12llu %7.1f%% | %14llu %14llu %7.1f%%\n",
                    r.name.c_str(),
                    double(r.mesh.vertex_count()) / r.mesh.triangle_count(),
                    static_cast<unsigned long long>(r.pass1_indexed),
                    static_cast<unsigned long long>(r.pass1_flat),
                    change(r.pass1_flat, r.pass1_indexed),
                    static_cast<unsigned long long>(r.pass2_indexed),
                    static_cast<unsigned long long>(r.pass2_flat),
                    change(r.pass2_flat, r.pass2_indexed));
    }

    // Where the matrix comes from. The frame is the same either way; a baked one
    // is sixteen moves every thread runs, and a program that serves one matrix.
    std::printf("\n  %-12s | %12s %12s %8s | %12s %12s %8s\n", "uniform", "steps baked",
                "steps window", "change", "lanes baked", "lanes window", "change");
    for (const Row& r : rows) {
        std::printf("  %-12s | %12llu %12llu %7.1f%% | %12llu %12llu %7.1f%%\n",
                    r.name.c_str(), static_cast<unsigned long long>(r.pass1_steps_baked),
                    static_cast<unsigned long long>(r.pass1_steps_window),
                    change(r.pass1_steps_baked, r.pass1_steps_window),
                    static_cast<unsigned long long>(r.pass1_lanes_baked),
                    static_cast<unsigned long long>(r.pass1_lanes_window),
                    change(r.pass1_lanes_baked, r.pass1_lanes_window));
    }

    std::printf("\n  %-12s %10s %10s %10s %10s\n", "seconds", "walk", "tiled", "shared",
                "raytrace");
    for (const Row& r : rows) {
        std::printf("  %-12s %10.2f %10.2f %10.2f %10.2f\n", r.name.c_str(),
                    r.walk.seconds, r.tiled.seconds, r.shared.seconds, r.ray.seconds);
    }

    const std::string csv_path = args.out_dir + "mesh_render.csv";
    std::ofstream csv(csv_path);
    if (!csv) {
        std::fprintf(stderr, "mesh_render: cannot write %s\n", csv_path.c_str());
        return 1;
    }
    csv << "mesh,vertices,triangles,vertices_per_triangle,width,height,route,"
        << "weighted_lane_ops,divergence_rate,seconds\n";
    for (const Row& r : rows) {
        const std::pair<const char*, const Reading*> routes[] = {{"walk", &r.walk},
                                                                 {"tiled", &r.tiled},
                                                                 {"shared", &r.shared},
                                                                 {"raytrace", &r.ray}};
        for (const auto& [name, reading] : routes) {
            csv << r.name << ',' << r.mesh.vertex_count() << ','
                << r.mesh.triangle_count() << ','
                << double(r.mesh.vertex_count()) / r.mesh.triangle_count() << ',' << size
                << ',' << size << ',' << name << ',' << reading->weighted << ','
                << reading->divergence << ',' << reading->seconds << '\n';
        }
        csv << r.name << ',' << r.mesh.vertex_count() << ',' << r.mesh.triangle_count()
            << ',' << double(r.mesh.vertex_count()) / r.mesh.triangle_count() << ','
            << size << ',' << size << ",pass1_indexed," << r.pass1_indexed << ",,\n";
        csv << r.name << ',' << size << ',' << size << ",pass1_steps_baked,"
            << r.pass1_steps_baked << ",,\n";
        csv << r.name << ',' << size << ',' << size << ",pass1_steps_window,"
            << r.pass1_steps_window << ",,\n";
        csv << r.name << ',' << r.mesh.vertex_count() << ',' << r.mesh.triangle_count()
            << ',' << double(r.mesh.vertex_count()) / r.mesh.triangle_count() << ','
            << size << ',' << size << ",pass1_flattened," << r.pass1_flat << ",,\n";
        csv << r.name << ',' << r.mesh.vertex_count() << ',' << r.mesh.triangle_count()
            << ',' << double(r.mesh.vertex_count()) / r.mesh.triangle_count() << ','
            << size << ',' << size << ",pass2_flattened," << r.pass2_flat << ",,\n";
    }
    std::printf("\n  wrote %s and %zu images in %s\n\n", csv_path.c_str(),
                rows.size() * 4, images.c_str());

    for (const Row& r : rows) {
        if (!r.routes_agree) {
            return 1;
        }
    }
    return 0;
}

// Everything the runtime refuses arrives as an exception — a device allocation
// too large for the budget above, most of the time. Reported rather than left
// to terminate, so a resolution that will not fit says so.
int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mesh_render: %s\n", e.what());
        return 1;
    }
}
