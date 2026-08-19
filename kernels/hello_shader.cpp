// The shortest program that draws something of its own.
//
// Everything a caller needs is here: geometry, an attribute a vertex carries,
// a fragment shader, and a frame to put the result in. It exists to be read —
// this is the answer to "how do I write a rendering program against this
// runtime", and if it stops being short the answer has got worse.
//
//   ./build/kernels/hello_shader                  benchmarks/result/hello_shader.ppm
//   ./build/kernels/hello_shader 256 256          at another size

#include <cstdint>
#include <cstdio>
#include <vector>

#include "app_run.hpp"
#include "pipeline/draw.hpp"
#include "pipeline/raster.hpp"
#include "pipeline/swap_chain.hpp"
#include "pipeline/vertex.hpp"
#include "ppm.hpp"

int main(int argc, char** argv)
{
    const Args args = parse_args(argc, argv);
    const uint32_t width = args.number(0, 128);
    const uint32_t height = args.number(1, 96);

    // --- the scene ----------------------------------------------------------
    // Two triangles, and three floats a vertex to carry to the pixels: here a
    // colour, but a normal or a texture coordinate rides the same way.
    // clang-format off
    const std::vector<Float3> world{
        Float3{-1.5f, -1.0f, 0.0f}, Float3{ 1.0f, -1.1f, 0.0f}, Float3{-0.2f,  1.2f, 0.0f},
        Float3{-0.6f, -1.2f, 0.8f}, Float3{ 1.6f, -0.4f, 0.8f}, Float3{ 0.7f,  1.0f, 0.8f},
    };
    const std::vector<float> attributes{
        1.0f, 0.2f, 0.1f,   0.1f, 1.0f, 0.2f,   0.2f, 0.1f, 1.0f,
        0.9f, 0.9f, 0.2f,   0.2f, 0.9f, 0.9f,   0.9f, 0.2f, 0.9f,
    };
    // clang-format on
    constexpr uint32_t VARYINGS = 3;

    const DrawTarget target{width, height, Camera{}};

    // --- the device ---------------------------------------------------------
    MyGPURuntime rt(1u << 26);

    void* geometry = rt.myrt_malloc(world.size() * sizeof(Float3));
    void* screen = rt.myrt_malloc(world.size() * screen_vertex_bytes(VARYINGS));
    void* attribute_buffer = rt.myrt_malloc(attributes.size() * sizeof(float));
    rt.myrt_memcpy(geometry, world.data(), world.size() * sizeof(Float3),
                   Direction::HostToDevice);
    rt.myrt_memcpy(attribute_buffer, attributes.data(), attributes.size() * sizeof(float),
                   Direction::HostToDevice);

    SwapChain chain(rt, target);

    // --- pass 1: vertices to the screen -------------------------------------
    VertexStageArgs vertex;
    vertex.view_projection = target.camera.view_projection(target.aspect());
    vertex.world_offset = rt.myrt_device_offset(geometry);
    vertex.screen_offset = rt.myrt_device_offset(screen);
    vertex.vertex_count = static_cast<uint32_t>(world.size());
    vertex.width = width;
    vertex.height = height;
    vertex.attribute_offset = rt.myrt_device_offset(attribute_buffer);
    vertex.varying_count = VARYINGS;
    run_vertex_stage(rt, vertex);
    rt.myrt_sync(false);

    // --- pass 2: the fragment shader ----------------------------------------
    // A function that emits instructions, called once as the kernel is built.
    // Everything the instruction set can do is available: this one interpolates
    // the vertex colour, darkens it with depth, and lifts the blue.
    RasterStageArgs raster;
    raster.screen_offset = rt.myrt_device_offset(screen);
    raster.framebuffer_offset = rt.myrt_device_offset(chain.back().pixels);
    raster.depth_offset = rt.myrt_device_offset(chain.back().depth);
    raster.depth = DepthUse::Test;
    raster.width = width;
    raster.height = height;
    raster.triangle_count = static_cast<uint32_t>(world.size() / 3);
    raster.varying_count = VARYINGS;
    raster.shading.mode = ShadingMode::Custom;
    raster.shading.shade = [](IRBuilder& k, const Fragment& f) {
        // The smallest of the three weights is 0 on an edge and 1/3 dead centre,
        // so this darkens the rim of every triangle. Nothing about it is built in
        // — it is arithmetic on values the runtime handed over.
        const Reg<Scalar> rim = k.min(f.w0, k.min(f.w1, f.w2));
        const Reg<Scalar> gain = k.constant(0.4f);
        k.fma(gain, rim, k.constant(2.4f));
        for (uint32_t channel = 0; channel < 3; ++channel) {
            k.copy_into(f.out.component(channel), k.mul(f.varyings[channel], gain));
        }
    };
    run_raster_stage(rt, raster);

    // --- present ------------------------------------------------------------
    const std::vector<Float3> pixels = chain.present();

    // Float3 a pixel on the device, three floats a pixel in the file.
    std::vector<float> flat;
    flat.reserve(pixels.size() * 3);
    for (const Float3& pixel : pixels) {
        flat.push_back(pixel.x);
        flat.push_back(pixel.y);
        flat.push_back(pixel.z);
    }
    const std::string path = args.out_dir + "hello_shader.ppm";
    write_ppm(path, flat, width, height);

    rt.myrt_sync();  // [STATS] divergence: X.X%, throughput: X.X GIOPS
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
