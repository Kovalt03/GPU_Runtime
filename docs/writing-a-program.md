# Writing a program against the runtime

A rendering program is host C++ that allocates buffers, describes two passes and
reads the frame back. What makes it a *program* rather than a configuration is
the fragment shader: a C++ callable that runs once, as the kernel is assembled,
and emits instructions.

`apps/hello_shader.cpp` is the whole of it in about eighty lines. The two
parts that are not bookkeeping:

```cpp
// Three floats a vertex, alongside the position, carried to the pixels.
vertex.attribute_offset = rt.myrt_device_offset(attribute_buffer);
vertex.varying_count    = 3;

// The shader. Called at build time; f holds registers, not values.
raster.shading.mode = ShadingMode::Custom;
raster.varying_count = 3;
raster.shading.shade = [](IRBuilder& k, const Fragment& f) {
    const Reg<Scalar> rim = k.min(f.w0, k.min(f.w1, f.w2));
    const Reg<Scalar> gain = k.constant(0.4f);
    k.fma(gain, rim, k.constant(2.4f));
    for (uint32_t channel = 0; channel < 3; ++channel) {
        k.copy_into(f.out.component(channel), k.mul(f.varyings[channel], gain));
    }
};
```

`Fragment` hands over what a GLSL fragment stage receives: the interpolated
varyings, the pixel's position and depth, the three barycentric weights — all
perspective-corrected, because that correction is what an attribute has to be
interpolated with and it should not be a caller's job to rediscover — and the
register the colour goes in. Uniforms arrive through the constant window
(`Uniforms::Window`), which is charged once a warp rather than once a thread.

Three things follow from the shader being a builder rather than a text:

- **The whole instruction set is available**, branches included. There is no
  shading language to gate what a caller may write, so anything the ISA can do a
  shader can do — the cost is that the shader is C++ and its errors are C++
  errors, not compile diagnostics against a shading grammar.
- **Loops over varyings unroll before the kernel exists.** The `for` above runs
  on the host; the device sees three straight-line multiplies. A shading language
  would need a compiler to reach the same place.
- **It costs what it emits.** The divergence and throughput a run reports include
  the shader, so a branch in it shows up in the same figures as a branch in the
  rasteriser.

The two built-in modes remain: `ShadingMode::Barycentric` fills with the weights
and `Diffuse` applies a point light. A `Custom` shader that interpolates a
per-vertex colour produces a frame **pixel-identical** to `Barycentric`, which is
how the varying path is tested — `Barycentric` was already checked against a host
reference, so the new path inherits that check rather than needing its own.

The shader rides inside `Shading`, which every route that can colour a pixel
already takes — so the ray tracer picks up the same function without a signature
of its own, and one shader draws the same picture down both. That is a claim
about naming as much as arithmetic: the walk interpolates corrected weights
across a projected triangle while the tracer reads Möller–Trumbore's `u` and `v`
out of a world-space solve, and a test holds the two frames against each other.

All four routes take a shader. What differs is what each can put in a fragment:

| | walk | tiled / shared | ray tracer |
|---|---|---|---|
| custom shader | yes | yes | yes |
| varyings | yes | refused — the tile format is fixed | none: no vertex stage |
| `Diffuse` | yes | refused — no world position | yes |
| `f.depth` | NDC, as the depth buffer holds it | NDC | the ray parameter |

The refusals are narrow on purpose. A tile carries three vertices of four floats,
fixed inside the kernel, so a varying has nowhere to go and a point light has no
world position to work from — but the weights, the pixel and the depth are all
there already, and a shader that asks the geometry no questions wants nothing
else. Refusing the shader outright would have been the wider claim, not the safer
one. What is refused throws rather than draws a plausible frame that is not the
one asked for.

A shader runs for fragments that lose: under `predicated` nothing is masked and
the colour is blended away, and the tracer shades once per candidate triangle.
Writing `out` is safe; a store or an atomic in a shader fires for candidates the
frame never shows.

---

[← back to the README](../README.md)
