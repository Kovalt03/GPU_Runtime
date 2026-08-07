// Ray-Triangle intersection, Möller-Trumbore, rendered to a PPM.
//
// This is the validation workload for the whole stack: one thread per pixel,
// each casting a ray at a single triangle. Divergence appears on its own at the
// triangle's edges, where some lanes of a warp hit and the rest miss — which is
// exactly what the benchmark in stage 7 goes on to measure.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <vector>

#include "isa.hpp"
#include "runtime.hpp"

namespace {

// --- scene ------------------------------------------------------------------
// Placed at z = -2 and sized so the triangle covers a good part of the frame:
// its edges then cross many warps, which is where divergence comes from.

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// Apex up, base along the bottom.
//
// Wound counter-clockwise as seen from the camera. The intersection test culls
// back faces — it compares a < eps rather than |a| < eps, the ISA having no
// absolute value — so the opposite winding renders nothing at all, and a blank
// frame with divergence at exactly 0% is what that looks like.
struct Scene {
    Vec3 v0{0.0f, 0.5f, -2.0f};    // apex
    Vec3 v1{-0.5f, -0.5f, -2.0f};  // bottom left
    Vec3 v2{0.5f, -0.5f, -2.0f};   // bottom right
    Vec3 origin{0.0f, 0.0f, 0.0f};
};

// --- register map -----------------------------------------------------------
// Follows DOC/01_virtual_isa.md, which worked the algorithm out against the ISA
// before any of it was written. Vectors occupy three consecutive registers, so
// each name below claims r, r+1, r+2 unless marked scalar.
//
// The launch reserves r253..r255 for thread coordinates, so allocation runs
// upward from r0 and has plenty of room.

constexpr uint8_t R_ORIGIN = 0;  // O
constexpr uint8_t R_DIR = 3;     // D
constexpr uint8_t R_V0 = 6;      // triangle
constexpr uint8_t R_V1 = 9;
constexpr uint8_t R_V2 = 12;
constexpr uint8_t R_E1 = 15;   // V1 - V0
constexpr uint8_t R_E2 = 18;   // V2 - V0
constexpr uint8_t R_H = 21;    // cross(D, E2)
constexpr uint8_t R_A = 24;    // scalar: dot(E1, h)
constexpr uint8_t R_F = 25;    // scalar: 1/a
constexpr uint8_t R_S = 26;    // O - V0
constexpr uint8_t R_U = 29;    // scalar: barycentric u
constexpr uint8_t R_Q = 30;    // cross(s, E1)
constexpr uint8_t R_V = 33;    // scalar: barycentric v
constexpr uint8_t R_T = 34;    // scalar: ray parameter
constexpr uint8_t R_EPS = 35;  // scalar constants
constexpr uint8_t R_ZERO = 36;
constexpr uint8_t R_ONE = 37;
constexpr uint8_t R_COND = 38;  // scalar: branch conditions
constexpr uint8_t R_COND_TMP = 39;
constexpr uint8_t R_COLOR = 40;  // rgb written to the framebuffer
constexpr uint8_t R_ADDR = 43;   // scalar: framebuffer byte offset
constexpr uint8_t R_TMP = 44;    // scalar scratch
constexpr uint8_t R_TMP2 = 45;

// --- kernel -----------------------------------------------------------------

struct KernelArgs {
    const Scene* scene = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t framebuffer_offset = 0;  // byte offset of the framebuffer in device memory
};

// Builds the instruction sequence. Called once per launch, so anything known at
// this point — the triangle, the resolution, the framebuffer address — can be
// baked in as an immediate rather than loaded per thread. On real hardware that
// is what constant memory and immediate operands are for: the values are
// uniform across the launch, and loading them 65536 times would swamp the
// arithmetic being measured.
Program build_ray_triangle_program(void** args)
{
    const KernelArgs& a = *static_cast<const KernelArgs*>(args[0]);
    Program p;

    // [1] Constants and the triangle, as immediates.
    const auto mov_vec3 = [&p](uint8_t reg, const Vec3& v) {
        p.push_back(make_v_mov_f32(reg + 0, v.x));
        p.push_back(make_v_mov_f32(reg + 1, v.y));
        p.push_back(make_v_mov_f32(reg + 2, v.z));
    };

    p.push_back(make_v_mov_f32(R_EPS, 1e-6f));
    p.push_back(make_v_mov_f32(R_ZERO, 0.0f));
    p.push_back(make_v_mov_f32(R_ONE, 1.0f));

    mov_vec3(R_ORIGIN, a.scene->origin);
    mov_vec3(R_V0, a.scene->v0);
    mov_vec3(R_V1, a.scene->v1);
    mov_vec3(R_V2, a.scene->v2);

    // [2] Ray direction: normalize(x - 0.5, 0.5 - y, -1), where x and y are the
    //     pixel *centre* in [0,1). REG_GLOBAL_ID_X/_Y arrive as integer pixel
    //     coordinates, and the ISA has no divide, so the reciprocal is baked in
    //     and the kernel multiplies.
    //
    //     Sampling the centre rather than the corner matters: an edge landing
    //     exactly on a corner leaves u + v at 1.0 give or take a rounding error,
    //     and neighbouring pixels along it then disagree, fraying the edge. The
    //     half-pixel folds into the constant, so it costs nothing:
    //       (px + 0.5) * inv_w - 0.5  ==  px * inv_w + (0.5 * inv_w - 0.5)
    const float inv_w = 1.0f / static_cast<float>(a.width);
    const float inv_h = 1.0f / static_cast<float>(a.height);

    p.push_back(make_v_mov_f32(R_TMP, inv_w));
    p.push_back(make_v_mul_f32(R_DIR + 0, REG_GLOBAL_ID_X, R_TMP));
    p.push_back(make_v_mov_f32(R_TMP2, 0.5f * inv_w - 0.5f));
    p.push_back(make_v_add_f32(R_DIR + 0, R_DIR + 0, R_TMP2));

    // Negated, because an image numbers its rows downward while the world has
    // y pointing up. Without this the triangle renders upside down.
    p.push_back(make_v_mov_f32(R_TMP, -inv_h));
    p.push_back(make_v_mul_f32(R_DIR + 1, REG_GLOBAL_ID_Y, R_TMP));
    p.push_back(make_v_mov_f32(R_TMP2, 0.5f - 0.5f * inv_h));
    p.push_back(make_v_add_f32(R_DIR + 1, R_DIR + 1, R_TMP2));

    p.push_back(make_v_mov_f32(R_DIR + 2, -1.0f));

    p.push_back(make_v_norm_vec3_f32(R_DIR, R_DIR));

    // [3] Möller-Trumbore, transcribed from the mapping in DOC/01 that was
    //     worked out against the ISA before any of this existed.
    //
    //     A branch target is not known while the tests are being emitted, the
    //     miss path not existing yet. Placeholders are recorded and patched once
    //     the label is placed — a forward reference, as any assembler handles it.
    std::vector<size_t> miss_branches;
    const auto branch_to_miss = [&p, &miss_branches](uint8_t cond) {
        miss_branches.push_back(p.size());
        p.push_back(make_bra_div(cond, 0));
    };

    p.push_back(make_v_sub_vec3_f32(R_E1, R_V1, R_V0));
    p.push_back(make_v_sub_vec3_f32(R_E2, R_V2, R_V0));
    p.push_back(make_v_cross_vec3_f32(R_H, R_DIR, R_E2));
    p.push_back(make_v_dot_vec3_f32(R_A, R_E1, R_H));

    // The ISA has no absolute value, so this tests a < eps rather than
    // |a| < eps. That rejects a back-facing triangle along with a parallel ray
    // — back-face culling, for free.
    p.push_back(make_v_cmp_f32(R_COND, R_A, R_EPS, CmpOp::LT));
    branch_to_miss(R_COND);

    p.push_back(make_v_rcp_f32(R_F, R_A));
    p.push_back(make_v_sub_vec3_f32(R_S, R_ORIGIN, R_V0));

    // u = f * dot(s, h)
    p.push_back(make_v_dot_vec3_f32(R_U, R_S, R_H));
    p.push_back(make_v_mul_f32(R_U, R_U, R_F));

    // Outside the triangle when u < 0 or u > 1. Adding the two flags is an OR,
    // since each is 1.0 or 0.0 — one branch instead of two, and so one
    // divergence point instead of two.
    p.push_back(make_v_cmp_f32(R_COND, R_U, R_ZERO, CmpOp::LT));
    p.push_back(make_v_cmp_f32(R_COND_TMP, R_U, R_ONE, CmpOp::GT));
    p.push_back(make_v_add_f32(R_COND, R_COND, R_COND_TMP));
    branch_to_miss(R_COND);

    // v = f * dot(D, q)
    p.push_back(make_v_cross_vec3_f32(R_Q, R_S, R_E1));
    p.push_back(make_v_dot_vec3_f32(R_V, R_DIR, R_Q));
    p.push_back(make_v_mul_f32(R_V, R_V, R_F));

    // Outside when v < 0 or u + v > 1.
    p.push_back(make_v_add_f32(R_TMP, R_U, R_V));
    p.push_back(make_v_cmp_f32(R_COND, R_V, R_ZERO, CmpOp::LT));
    p.push_back(make_v_cmp_f32(R_COND_TMP, R_TMP, R_ONE, CmpOp::GT));
    p.push_back(make_v_add_f32(R_COND, R_COND, R_COND_TMP));
    branch_to_miss(R_COND);

    // t = f * dot(E2, q). Behind the camera counts as a miss.
    p.push_back(make_v_dot_vec3_f32(R_T, R_E2, R_Q));
    p.push_back(make_v_mul_f32(R_T, R_T, R_F));
    p.push_back(make_v_cmp_f32(R_COND, R_T, R_EPS, CmpOp::LT));
    branch_to_miss(R_COND);

    // [4] Hit: barycentric coordinates as the colour. Each vertex comes out a
    //     pure primary, so wrong intersection maths shows as a wrong gradient —
    //     flat white would hide it.
    //
    //     V_MOV_F32 takes an immediate, so a register-to-register copy has to go
    //     through arithmetic; adding zero is the idiom.
    p.push_back(make_v_add_f32(R_COLOR + 0, R_U, R_ZERO));
    p.push_back(make_v_add_f32(R_COLOR + 1, R_V, R_ZERO));
    p.push_back(make_v_sub_f32(R_COLOR + 2, R_ONE, R_U));
    p.push_back(make_v_sub_f32(R_COLOR + 2, R_COLOR + 2, R_V));

    // Skip the miss path. Patched below for the same reason as the others.
    const size_t skip_miss = p.size();
    p.push_back(make_bra(0));

    // [5] Miss: background.
    const size_t miss_label = p.size();
    p.push_back(make_v_mov_f32(R_COLOR + 0, 0.0f));
    p.push_back(make_v_mov_f32(R_COLOR + 1, 0.0f));
    p.push_back(make_v_mov_f32(R_COLOR + 2, 0.0f));

    // Both paths converge here, which is where the warp reconverges too.
    const size_t write_label = p.size();
    for (size_t at : miss_branches) {
        p[at] = make_bra_div(p[at].src0, static_cast<int32_t>(miss_label - at));
    }
    p[skip_miss] = make_bra(static_cast<int32_t>(write_label - skip_miss));

    // [6] addr = framebuffer_offset + (py * width + px) * 3 * sizeof(float)
    //
    //     Rounding the grid up leaves lanes with px >= width whenever the width
    //     is not a multiple of the warp size. Their address lands on the next
    //     row, so they are branched over the stores entirely. The current scene
    //     hides this — those lanes write black onto a black margin — but the
    //     corruption is real for any triangle reaching the left edge.
    p.push_back(make_v_mov_f32(R_TMP, static_cast<float>(a.width)));
    p.push_back(make_v_cmp_f32(R_COND, REG_GLOBAL_ID_X, R_TMP, CmpOp::GE));
    const size_t skip_store = p.size();
    p.push_back(make_bra_div(R_COND, 0));

    p.push_back(make_v_mul_f32(R_ADDR, REG_GLOBAL_ID_Y, R_TMP));
    p.push_back(make_v_add_f32(R_ADDR, R_ADDR, REG_GLOBAL_ID_X));

    p.push_back(make_v_mov_f32(R_TMP, 3.0f * sizeof(float)));
    p.push_back(make_v_mul_f32(R_ADDR, R_ADDR, R_TMP));

    p.push_back(make_v_mov_f32(R_TMP, static_cast<float>(a.framebuffer_offset)));
    p.push_back(make_v_add_f32(R_ADDR, R_ADDR, R_TMP));

    // Store takes the address first and the value second, the reverse of load.
    p.push_back(make_v_st_global_f32(R_ADDR, R_COLOR + 0, 0.0f));
    p.push_back(make_v_st_global_f32(R_ADDR, R_COLOR + 1, 4.0f));
    p.push_back(make_v_st_global_f32(R_ADDR, R_COLOR + 2, 8.0f));

    p[skip_store] = make_bra_div(R_COND, static_cast<int32_t>(p.size() - skip_store));

    p.push_back(make_ret());
    return p;
}

// --- PPM --------------------------------------------------------------------

void write_ppm(const std::string& path, const std::vector<float>& rgb, uint32_t width,
               uint32_t height)
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

}  // namespace

int main(int argc, char** argv)
{
    const uint32_t width = (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : 256;
    const uint32_t height =
        (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : width;
    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t fb_bytes = pixels * 3 * sizeof(float);

    MyGPURuntime rt(fb_bytes + (1u << 20));

    void* fb = rt.myrt_malloc(fb_bytes);

    const Scene scene;
    KernelArgs kargs;
    kargs.scene = &scene;
    kargs.width = width;
    kargs.height = height;
    // Register values are byte offsets from the start of device memory, and the
    // framebuffer is the first allocation, so this is zero — but computing it
    // keeps the kernel correct if anything is ever allocated ahead of it.
    kargs.framebuffer_offset = 0;

    void* args[] = {&kargs};

    // A 2D launch, so that REG_GLOBAL_ID_X/_Y arrive as pixel coordinates
    // directly. A flat index would have to be split with an integer division
    // the ISA cannot express.
    //
    // 32 threads along x puts one warp on 32 horizontally adjacent pixels,
    // which is what makes a triangle edge split a warp — the divergence this
    // whole project exists to measure.
    const dim3 block{32, 1, 1};
    const dim3 grid{(width + block.x - 1) / block.x, height, 1};

    std::printf("rendering %ux%u — %zu pixels, %u blocks of %u threads\n", width, height,
                pixels, grid.volume(), block.volume());
    rt.myrt_launch(build_ray_triangle_program, grid, block, args);
    rt.myrt_sync();

    std::vector<float> host_fb(pixels * 3, 0.0f);
    rt.myrt_memcpy(host_fb.data(), fb, fb_bytes, Direction::DeviceToHost);

    const std::string path = "output/result.ppm";
    write_ppm(path, host_fb, width, height);
    std::printf("wrote %s\n", path.c_str());

    rt.myrt_free(fb);
    return 0;
}
