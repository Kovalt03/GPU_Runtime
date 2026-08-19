# The Virtual ISA

45 opcodes, eight bytes each: an opcode, three register
indices and an immediate. `test/unit/test_isa.cpp` checks the inventory and
the naming scheme; `gpurt/isa.hpp` is the list itself.

## Naming

Opcodes follow a fixed scheme so that future extensions fill in a slot rather
than force a rename:

```
ALU        V_<OP>[_<SHAPE>]_<TYPE>                  V_ADD_F32, V_CROSS_VEC3_F32
Memory     V_<LD|ST|CP|ATOM_OP>_<SPACE>[_<SPACE>][_<SHAPE>]_<TYPE>
                                                    V_LD_GLOBAL_VEC3_F32
                                                    V_CP_ASYNC_SHARED_GLOBAL_F32
                                                    V_ATOM_ADD_GLOBAL_F32
Warp       S_<OP>                                   S_BALLOT, S_ANY, S_ALL
Control    <OP>                                     BRA, BRA_DIV, BARRIER, RET
```

- **`V_`** marks instructions whose result differs between lanes; **`S_`** those
  where every lane ends up with the same value, following AMD, where `s_`
  instructions write a scalar register file. This machine has none, so the line
  is drawn on the result rather than the file. Control flow mutates warp state
  only (`pc`, `activeMask`) and carries no prefix.
- A lane exchange gives every lane something different, so it is
  `V_SHUFFLE_F32` and not `S_SHUFFLE` — the rule decides it rather than
  intuition about which instructions feel collective.
- **`<SHAPE>`** is omitted for scalars. `VEC3` and `MAT4` are built —
  `V_MATVEC_MAT4_F32` arrived for the vertex stage, `V_MATMUL_MAT4_F32` for
  instancing and `V_LD_GLOBAL_VEC3_F32` once there was a memory model to show
  what a wide load saves — each filling a slot the scheme had reserved without
  renaming anything. `MAT3` is still open,
  and `VEC4` is what the raster routes would want: their screen vertex is four
  floats, so a VEC3 load leaves 1/w behind.
- **`<TYPE>`** always comes last. `F32` and `F16` are built; `F64` is open.
- **The memory verbs are four, not two.** `LD` and `ST` have a register at one
  end. `CP` has memory at both, so it names two spaces and puts the destination
  first as PTX does — `V_CP_ASYNC_SHARED_GLOBAL_F32`. `ATOM` reads and writes one
  address indivisibly and carries its operation in the name.

`F32` rather than `FP32` because that is the mnemonic standard — PTX `add.f32`,
AMD `v_add_f32`, WGSL / SPIR-V / Rust `f32`. The rule is machine-checked by
`OpcodeNamesFollowScheme` in `test/unit/test_isa.cpp`, so it cannot drift as
opcodes are added.

## The inventory

Grouped by what each group made possible, since that is why they exist.

| | Opcodes | What it bought |
|---|---|---|
| Scalar arithmetic | `V_MUL_F32` `V_ADD_F32` `V_SUB_F32` `V_RCP_F32` `V_SQRT_F32` `V_FMA_F32` `V_MIN_F32` `V_MAX_F32` `V_MOV_F32` | the intersection test |
| Vectors | `V_ADD_VEC3_F32` `V_SUB_VEC3_F32` `V_SCALE_VEC3_F32` `V_DOT_VEC3_F32` `V_CROSS_VEC3_F32` `V_NORM_VEC3_F32` | a kernel that reads like the maths |
| Matrices | `V_MATVEC_MAT4_F32` `V_MATMUL_MAT4_F32` | the vertex stage, then instancing — 16 multiply-adds against 64 |
| Matrix unit | `V_MMA_16X16X16_F32` `V_MMA_16X16X16_F16` | a warp computing 16x16x16 in one instruction; the first opcode whose operands belong to the warp rather than the lane |
| Wide loads | `V_LD_GLOBAL_VEC3_F32` `V_LD_GLOBAL_MAT4_F32` `V_LD_SHARED_16X16_F32` `V_LD_SHARED_16X16_F16` | a wide load is not several narrow ones — it asks about the lines a span crosses, once |
| Memory | `V_LD_GLOBAL_F32` `V_ST_GLOBAL_F32` `V_LD_SHARED_F32` `V_ST_SHARED_F32` | separate address spaces |
| Constant window | `V_LD_CONST_F32` `V_LD_CONST_MAT4_F32` | a load that *declares* the address is warp-uniform, so it is charged once a warp |
| Cluster | `V_LD_CLUSTER_F32` `BARRIER_CLUSTER` | a block reading its neighbour's shared memory (Hopper DSMEM) |
| Asynchronous copy | `V_CP_ASYNC_SHARED_GLOBAL_F32` `S_CP_ASYNC_WAIT` | global to shared without a register and without the warp waiting; visibility comes from the wait, not from time |
| Atomics | `V_ATOM_ADD_GLOBAL_F32` | the *return* is the point — 32 lanes adding one to a counter get 32 different answers, which is how a compaction pass hands out slots |
| Comparison | `V_CMP_F32` | 1.0 or 0.0 in a float register, so coverage can branch or blend |
| Warp level | `S_BALLOT` `S_ANY` `S_ALL` `S_SYNCWARP` `V_SHUFFLE_F32` | lanes cooperating rather than disagreeing; participants are declared in the immediate, never inferred |
| Synchronisation | `BARRIER` | shared-memory staging; reaching it diverged is refused |
| Reordering | `REORDER` | regrouping a block's threads by a key, so lanes about to do the same thing share a warp |
| Control flow | `BRA` `BRA_DIV` `RET` | |

`opcode_name()` has no `default` arm, so adding an opcode without naming it is a
compile error rather than a run-time surprise.

---

[← back to the README](../README.md)
