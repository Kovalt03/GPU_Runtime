# 1단계 — Virtual ISA

## 목표

GPU가 실행하는 저수준 명령어셋을 C++ enum + struct로 정의한다.
실제 NVIDIA PTX와 유사한 구조를 취하되, Ray-Triangle intersection 구현에
필요한 모든 연산을 완비한 23개 opcode 세트로 설계한다.

최종 목표가 "Ray Tracing을 지원하는 GPU Runtime"이므로, 향후 **행렬 변환
(mat3/mat4)과 벡터 단위 메모리 접근**이 추가될 것을 전제로 네이밍 체계를
먼저 확정한다. 확장 시 기존 이름을 고치지 않고 자리만 채울 수 있어야 한다.

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/isa.hpp` | Opcode 정의, Instruction 구조체, CmpOp enum, 팩토리 함수 선언 |
| `gpurt/isa.cpp` | `opcode_name()` 구현, 팩토리 함수 구현 |
| `test/unit/test_isa.cpp` | GoogleTest — 인코딩·필드 배치·네이밍 규칙 검증 |

> 코드 주석은 영어로 작성한다. 이 문서는 구현 과정의 한글 기록이다.

---

## 네이밍 규칙 (확장의 기준)

```
연산 명령어    V_<OP>[_<SHAPE>]_<TYPE>
메모리 명령어  V_<LD|ST>_<SPACE>[_<SHAPE>]_<TYPE>
제어 흐름      <OP>                          (접두사·타입 없음)
```

### `V_` 접두사 — 레인별 실행 표식

**레인별 레지스터 파일(`Thread::regs`)에 결과를 쓰는 명령어에만 붙인다.**
AMD GCN/RDNA의 `v_add_f32`(vector ALU) 관례를 따른다.

제어 흐름(`BRA`, `BRA_DIV`, `RET`)은 레지스터를 쓰지 않고 워프 단위 상태
(`pc`, `activeMask`)만 바꾸므로 접두사가 없다. `BRA_DIV`는 조건 레지스터를
**읽기만** 하고 쓰지 않으므로 여기 속한다.

이 구분은 장식이 아니라 4단계 Scheduler의 구현 구조와 1:1로 대응한다 —
`V_` 명령어는 `for (lane : activeMask)` 루프 안에서, 제어 흐름 명령어는
루프 **밖**에서 워프 상태를 갱신하며 처리된다.

### `<SHAPE>` — 피연산자 형상

생략하면 스칼라(레지스터 1개)다. 별도 벡터 레지스터 파일 없이 스칼라
레지스터 배열을 연속으로 묶어 표현한다 (PTX `v4` 관례와 동일).

| 토큰 | 의미 | 레지스터 | 정렬 | 상태 |
|------|------|----------|------|------|
| (없음) | 스칼라 | 1 | — | 구현됨 |
| `VEC3` | 3-component | 연속 3 | 없음 | 구현됨 |
| `VEC4` | 4-component | 연속 4 | 4의 배수 | 예약 |
| `MAT3` | 3×3 행렬 (row-major) | 연속 9 | 4의 배수 | 예약 |
| `MAT4` | 4×4 행렬 (row-major) | 연속 16 | 4의 배수 | 구현됨 (`V_MATVEC_MAT4_F32`) |

`VEC3`에 정렬 제약을 두지 않은 것은 Möller–Trumbore의 레지스터 할당이
3개 단위로 촘촘히 이어지기 때문이다(`r0..2`, `r3..5`, ...). 반면 `VEC4`
이상은 정렬을 강제해야 나중에 벡터 로드를 한 트랜잭션으로 모델링할 수 있다.

### `<SPACE>` — 주소 공간

`GLOBAL`(device memory) / `SHARED`(block 내 공유). 향후 `CONST`, `LOCAL` 예약.

### `<TYPE>` — 항상 마지막

`F32`. 향후 `F64`, `F16`.

> **왜 `FP32`가 아니라 `F32`인가**
> `FP32`는 스펙시트·성능 문서의 용어이고("FP32 throughput", "FP16 Tensor Core"),
> **명령어 니모닉에서는 `f32`가 표준**이다 — PTX `add.f32`, AMD `v_add_f32`,
> WGSL/SPIR-V/Rust `f32`, RISC-V `fadd.s`. 우리가 짓는 것은 니모닉이므로 `F32`.
> (IEEE 754 표준 용어는 `binary32`이나 실무에서 쓰이지 않는다.)

### 적용 예시

```
V_ADD_F32              스칼라 덧셈
V_ADD_VEC3_F32         3-component 덧셈
V_LD_GLOBAL_F32        global에서 스칼라 1개 로드
V_LD_GLOBAL_VEC3_F32   global에서 연속 3개 로드
V_MATVEC_MAT4_F32      mat4 × vec4
V_ADD_F64              배정밀도 스칼라 덧셈           (예약)
BRA_DIV                워프 마스크 분할 — 접두사 없음
```

---

## 명령어셋 (23 opcodes)

### 스칼라 산술 (9개)

```
┌────────────────────┬──────────────────────────────────────────────────┐
│ Opcode             │ 의미                                             │
├────────────────────┼──────────────────────────────────────────────────┤
│ V_MUL_F32          │ reg[dst] = reg[src0] * reg[src1]                 │
│ V_ADD_F32          │ reg[dst] = reg[src0] + reg[src1]                 │
│ V_SUB_F32          │ reg[dst] = reg[src0] - reg[src1]                 │
│ V_RCP_F32          │ reg[dst] = 1.0f / reg[src0]      (src1 unused)   │
│ V_SQRT_F32         │ reg[dst] = sqrt(reg[src0])       (src1 unused)   │
│ V_FMA_F32          │ reg[dst] += reg[src0] * reg[src1] (in-place acc) │
│ V_MIN_F32          │ reg[dst] = min(reg[src0], reg[src1])             │
│ V_MAX_F32          │ reg[dst] = max(reg[src0], reg[src1])             │
│ V_MOV_F32          │ reg[dst] = imm             (src0/src1 unused)    │
└────────────────────┴──────────────────────────────────────────────────┘
```

### 벡터 산술 — VEC3 (6개)

연속 레지스터 관례: `dst`/`src`가 벡터의 **첫 번째** 레지스터 인덱스.

```
┌────────────────────┬──────────────────────────────────────────────────┐
│ Opcode             │ 의미                                             │
├────────────────────┼──────────────────────────────────────────────────┤
│ V_ADD_VEC3_F32     │ reg[dst..+2] = reg[src0..+2] + reg[src1..+2]     │
│ V_SUB_VEC3_F32     │ reg[dst..+2] = reg[src0..+2] - reg[src1..+2]     │
│ V_SCALE_VEC3_F32   │ reg[dst..+2] = reg[src0..+2] * reg[src1]         │
│                    │   (src1은 스칼라 1개)                            │
│ V_DOT_VEC3_F32     │ reg[dst] = dot(reg[src0..+2], reg[src1..+2])     │
│                    │   (dst는 스칼라 1개)                             │
│ V_CROSS_VEC3_F32   │ reg[dst..+2] =                                   │
│                    │   cross(reg[src0..+2], reg[src1..+2])            │
│ V_NORM_VEC3_F32    │ reg[dst..+2] = normalize(reg[src0..+2])          │
│                    │   (src1 unused)                                  │
└────────────────────┴──────────────────────────────────────────────────┘
```

`SHAPE` 토큰은 **명령어가 다루는 주된 형상**을 가리킨다. `V_DOT_VEC3_F32`의
결과처럼 일부 피연산자가 스칼라인 경우는 위 표에 개별 명시한다.

### 비교 (1개)

```
┌────────────────────┬──────────────────────────────────────────────────┐
│ Opcode             │ 의미                                             │
├────────────────────┼──────────────────────────────────────────────────┤
│ V_CMP_F32          │ reg[dst] = (reg[src0] OP reg[src1]) ? 1.0f : 0.0f│
│                    │ OP = imm 필드를 uint32_t로 reinterpret → CmpOp   │
└────────────────────┴──────────────────────────────────────────────────┘

enum class CmpOp : uint32_t { LT=0, GT=1, EQ=2, NEQ=3, LE=4, GE=5 };
```

결과가 float 레지스터에 1.0f / 0.0f로 저장되므로 `BRA_DIV`와 조합해 사용.
OR 조합: `V_ADD_F32 r_cond, r_c0, r_c1` → 하나라도 1.0f면 != 0.0f.

### 메모리 (4개)

```
┌────────────────────┬──────────────────────────────────────────────────┐
│ Opcode             │ 의미                                             │
├────────────────────┼──────────────────────────────────────────────────┤
│ V_LD_GLOBAL_F32    │ reg[dst] = global[reg[src0] + imm]  (src1 unused)│
│ V_ST_GLOBAL_F32    │ global[reg[src0] + imm] = reg[src1] (dst unused) │
│ V_LD_SHARED_F32    │ reg[dst] = shared[reg[src0] + imm]  (src1 unused)│
│ V_ST_SHARED_F32    │ shared[reg[src0] + imm] = reg[src1] (dst unused) │
└────────────────────┴──────────────────────────────────────────────────┘
```

주소는 항상 `src0`, 저장할 값은 항상 `src1`이다. 다만 팩토리 인자 순서는
로드가 `(dst, addr)`, 스토어가 `(addr, src)`로 서로 반대이므로 주의.

### 제어 흐름 (3개)

```
┌────────────────────┬──────────────────────────────────────────────────┐
│ Opcode             │ 의미                                             │
├────────────────────┼──────────────────────────────────────────────────┤
│ BRA                │ pc += (int32_t)imm  (무조건 분기, PC-relative)   │
│ BRA_DIV            │ if (reg[src0] != 0.0f) pc += (int32_t)imm        │
│                    │   → activeMask 분할 (divergence 발생 지점)       │
│ RET                │ 스레드 종료 (모든 피연산자 unused)               │
└────────────────────┴──────────────────────────────────────────────────┘
```

---

## 데이터 구조

```cpp
// gpurt/isa.hpp

enum class CmpOp : uint32_t { LT=0, GT=1, EQ=2, NEQ=3, LE=4, GE=5 };

enum class Opcode : uint8_t {
    // Scalar ALU
    V_MUL_F32, V_ADD_F32, V_SUB_F32,
    V_RCP_F32, V_SQRT_F32, V_FMA_F32,
    V_MIN_F32, V_MAX_F32, V_MOV_F32,
    // Vector ALU (VEC3)
    V_ADD_VEC3_F32, V_SUB_VEC3_F32, V_SCALE_VEC3_F32,
    V_DOT_VEC3_F32, V_CROSS_VEC3_F32, V_NORM_VEC3_F32,
    // Compare
    V_CMP_F32,
    // Memory
    V_LD_GLOBAL_F32, V_ST_GLOBAL_F32, V_LD_SHARED_F32, V_ST_SHARED_F32,
    // Control flow
    BRA, BRA_DIV, RET,
};

struct Instruction {
    Opcode  op;
    uint8_t dst;    // 목적지 레지스터 인덱스 (벡터/행렬은 첫 인덱스)
    uint8_t src0;   // 소스 레지스터 0
    uint8_t src1;   // 소스 레지스터 1
    float   imm;    // 즉시값 / 메모리 오프셋 / CmpOp (비트 재해석)
};
static_assert(sizeof(Instruction) == 8);

using Program = std::vector<Instruction>;
```

**enum 값은 직렬화되지 않는다.** `Program`은 메모리 상에만 존재하고 파일로
저장되지 않으므로, 카테고리 중간에 opcode를 추가해 뒤 값이 밀려도 무방하다.
`test/unit/test_isa.cpp`가 경계값을 검사하는 것은 계약이 아니라 **변경을 인지시키는
트립와이어**다.

---

## imm 필드 인코딩 규칙

`imm`은 4바이트 한 칸을 명령어별로 다르게 쓴다.

| 명령어 | 담는 것 | 방식 |
|--------|---------|------|
| `V_MOV_F32` | 실수 즉시값 | 그대로 |
| `V_LD_*` / `V_ST_*` | 메모리 오프셋 | 그대로 |
| `V_CMP_F32` | `CmpOp` (0~5) | **비트 재해석** |
| `BRA`, `BRA_DIV` | PC 상대 오프셋 (정수) | **값 변환** |

### CmpOp — 비트 재해석

```cpp
inline float encode_cmp_op(CmpOp op) {
    const uint32_t bits = static_cast<uint32_t>(op);
    float imm;
    std::memcpy(&imm, &bits, sizeof(float));
    return imm;
}

inline CmpOp decode_cmp_op(float imm) {
    uint32_t bits;
    std::memcpy(&bits, &imm, sizeof(uint32_t));
    return static_cast<CmpOp>(bits);
}
```

인코더와 디코더를 `gpurt/isa.hpp` 한 곳에 두어 Scheduler가 다른 규칙을 쓰는 일을
막는다. 어긋나면 커널이 `LT`를 `GE`로 실행하고, 증상이 렌더링 결과로만
나타나 원인 추적이 사실상 불가능하다.

### 분기 오프셋 — 값 변환 + 범위 검사

`float` 가수부가 24비트이므로 `|offset| <= 2^24`에서만 왕복이 정확하다.
`static_cast<float>(16777217)`은 경고 없이 `16777216.0f`가 된다. 잘못된 PC는
엉뚱한 명령어부터 실행을 시작시켜 디버깅이 가장 어려운 부류의 버그를 만들므로,
`make_bra` 계열에서 범위를 벗어나면 `std::runtime_error`를 던진다.

비트 재해석을 쓰면 int32 전 범위를 담을 수 있으나 값 변환을 택했다.
디버거에서 `inst.imm`이 `-7.0`으로 읽히는 편이 명령어 덤프를 눈으로 확인할 때
유리하고, 실 사용 범위(프로그램 길이 ≪ 2^24)에서 제약이 걸릴 일이 없다.

---

## 팩토리 함수 목록

함수 이름은 opcode 니모닉을 소문자로 그대로 옮긴다 (`V_ADD_VEC3_F32` →
`make_v_add_vec3_f32`). 길지만 opcode와 1:1이라 확장 시 기계적으로 늘릴 수 있다.

```cpp
// 스칼라
Instruction make_v_mul_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_add_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_sub_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_rcp_f32(uint8_t dst, uint8_t src0);
Instruction make_v_sqrt_f32(uint8_t dst, uint8_t src0);
Instruction make_v_fma_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_min_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_max_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_mov_f32(uint8_t dst, float imm);

// 벡터 (VEC3)
Instruction make_v_add_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_sub_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_scale_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1_scalar);
Instruction make_v_dot_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_cross_vec3_f32(uint8_t dst, uint8_t src0, uint8_t src1);
Instruction make_v_norm_vec3_f32(uint8_t dst, uint8_t src0);

// 비교
Instruction make_v_cmp_f32(uint8_t dst, uint8_t src0, uint8_t src1, CmpOp op);

// 메모리
Instruction make_v_ld_global_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_global_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);
Instruction make_v_ld_shared_f32(uint8_t dst, uint8_t addr_reg, float offset = 0.0f);
Instruction make_v_st_shared_f32(uint8_t addr_reg, uint8_t src, float offset = 0.0f);

// 제어 흐름
Instruction make_bra(int32_t offset);
Instruction make_bra_div(uint8_t cond_reg, int32_t offset);
Instruction make_ret();
```

---

## 확장 계획 (예약)

지금 구현하지 않되, 네이밍과 구조가 이를 수용할 수 있음을 확인해 둔다.

### 1. 행렬 연산 — Ray Tracing의 실제 요구

카메라/오브젝트 변환을 지원하려면 다음이 필요하다.
**언제 넣을지는 `09_graphics_pipeline.md` 에 계획으로 정리했다** — 6단계
(삼각형 1개가 화면에 보이는 것)를 끝낸 직후다.

```
V_MATVEC_MAT4_F32   reg[dst..+3]  = mat4(reg[src0..+15]) * vec4(reg[src1..+3])
V_MATMUL_MAT4_F32   reg[dst..+15] = mat4(src0) * mat4(src1)      (변환 합성)
V_TRANSPOSE_MAT3_F32
V_INVERSE_MAT3_F32  법선 변환용 inverse-transpose
```

**형상 토큰은 주된 피연산자(행렬)를 가리키고, 니모닉이 나머지를 구분한다.**
`MATVEC`/`MATMUL`처럼 연산 이름에 조합을 담으면 `MAT4_VEC4` 같은 이중 토큰이
필요 없다.

제약:

- **레지스터 압박** — mat4 하나가 16개를 먹는다. `regs`가 256개이고
  Möller–Trumbore가 이미 ~43개를 쓰므로, 변환 행렬 2~3개까지가 현실적이다.
- **정렬** — mat4는 4의 배수 인덱스에서 시작해야 한다 (`VEC4`와 동일 규칙).
- **`Instruction` 변경 불필요** — 행렬 연산은 즉시값을 쓰지 않으므로
  8바이트 구조체를 그대로 유지한다.

### 2. 벡터 단위 메모리 접근

```
V_LD_GLOBAL_VEC3_F32   reg[dst..+2] = global[reg[src0] + imm .. +2]
V_ST_GLOBAL_VEC4_F32
V_LD_GLOBAL_MAT4_F32
```

단순한 축약이 아니다. 실제 GPU에서 `ld.global.v4.f32`는 **한 번의 넓은 메모리
트랜잭션**으로 처리된다.

**`V_LD_GLOBAL_VEC3_F32` 은 구현했고, 그 예측이 맞았다.** 레이 트레이서의 정점
읽기가 스칼라 9회 → wide 3회가 되어 `Coalesced` 에서 -24.7%, 사이클 -42.9%.
`Flat` 은 한 lane-op 도 안 움직인다 — 그래서 이 opcode 가 `MemoryModel` 을
기다렸다. 값은 `instruction_cost` 가 300(세 float), 라인당 과금은 스칼라와 같은
100 이다. 둘을 갈라놓지 않으면 라인당 300 이 되어 이득이 그대로 상쇄된다.

`V_ST_GLOBAL_VEC4_F32` / `V_LD_GLOBAL_VEC4_F32` 은 아직 없다. 래스터 경로의
화면 정점이 x·y·z·1/w 넷이라 다음에 값을 하는 자리가 거기다.

### 3. FP64 / FP16

이름은 `V_ADD_F64`로 자리만 채우면 되지만, **실제 걸림돌은 데이터 레이아웃이다.**

- `float imm`이 4바이트라 f64 즉시값이 안 들어간다 → `V_MOV_F32` 2회로
  상·하위 32비트를 나눠 싣거나 상수 풀을 도입해야 한다.
- `Thread::regs`가 `std::array<float, 256>`이다 → f64 하나를 **연속 레지스터
  2개(짝수 정렬)** 에 나눠 담는다. NVIDIA가 f64를 처리하는 방식과 동일하며,
  `VEC3`의 연속 레지스터 관례를 그대로 확장한 것이다.

Ray Tracing 워크로드는 전 구간 FP32이므로 실제 추가 가능성은 낮다.

### 4. 스칼라 유닛 (`S_` 접두사)

AMD가 `s_`(scalar ALU)로 워프 내 모든 레인이 동일한 값을 다루는 연산을
분리하듯, 워프 균일(warp-uniform) 주소 계산 등을 `S_ADD_U32` 형태로 추가할 수
있다. 현재 시뮬레이터에는 연산 유닛이 하나뿐이라 `V_`만 존재한다 — `V_`
접두사가 의미를 갖는 것은 이 확장 여지 때문이다.

---

## Möller–Trumbore ISA 매핑 (검증)

완성형 ISA로 알고리즘 전체를 표현할 수 있음을 확인.

```
// 레지스터 할당
// r0..2  = ray origin O        r3..5  = ray direction D
// r6..8  = V0                  r9..11 = V1            r12..14 = V2
// r15..17 = E1                 r18..20 = E2
// r21..23 = h (cross)          r24    = a (dot)       r25 = f (rcp)
// r26..28 = s                  r29    = u
// r30..32 = q (cross)          r33    = v             r34 = t
// r35    = eps                 r36    = zero          r37 = one
// r38    = cond                r39    = cond_tmp
// r40..42 = result P (hit point)

V_MOV_F32        r35, 1e-6f          // eps
V_MOV_F32        r36, 0.0f           // zero
V_MOV_F32        r37, 1.0f           // one

V_SUB_VEC3_F32   r15, r9,  r6        // E1 = V1 - V0
V_SUB_VEC3_F32   r18, r12, r6        // E2 = V2 - V0

V_CROSS_VEC3_F32 r21, r3,  r18       // h = cross(D, E2)
V_DOT_VEC3_F32   r24, r15, r21       // a = dot(E1, h)

V_CMP_F32        r38, r24, r35, LT   // |a| < eps?
BRA_DIV          r38, miss_offset    // → miss (평행)

V_RCP_F32        r25, r24            // f = 1/a

V_SUB_VEC3_F32   r26, r0,  r6        // s = O - V0
V_DOT_VEC3_F32   r29, r26, r21       // u = dot(s, h)
V_MUL_F32        r29, r25, r29       // u *= f

V_CMP_F32        r38, r29, r36, LT   // u < 0?
V_CMP_F32        r39, r29, r37, GT   // u > 1?
V_ADD_F32        r38, r38, r39       // OR
BRA_DIV          r38, miss_offset

V_CROSS_VEC3_F32 r30, r26, r15       // q = cross(s, E1)
V_DOT_VEC3_F32   r33, r3,  r30       // v = dot(D, q)
V_MUL_F32        r33, r25, r33       // v *= f

V_ADD_F32        r39, r29, r33       // u + v
V_CMP_F32        r38, r33, r36, LT   // v < 0?
V_CMP_F32        r39, r39, r37, GT   // u+v > 1?
V_ADD_F32        r38, r38, r39
BRA_DIV          r38, miss_offset

V_DOT_VEC3_F32   r34, r18, r30       // t = dot(E2, q)
V_MUL_F32        r34, r25, r34       // t *= f

V_CMP_F32        r38, r34, r35, LT   // t < eps?
BRA_DIV          r38, miss_offset

// HIT: P = O + t*D
V_SCALE_VEC3_F32 r40, r3,  r34       // t*D
V_ADD_VEC3_F32   r40, r0,  r40       // P = O + t*D
// → V_ST_GLOBAL_F32 프레임버퍼에 색상 기록
BRA              ret_offset

// miss: 배경색 기록
// ...
RET
```

향후 변환 행렬이 도입되면 커널 앞단에 다음이 붙는다 (예약):

```
V_MATVEC_MAT4_F32 r0, r_view_inv, r_screen   // 스크린 → 월드 좌표 변환
```

---

## 구현 흐름

```
[1] gpurt/isa.hpp 작성
    └─ enum class CmpOp : uint32_t
    └─ enum class Opcode : uint8_t (23개, 네이밍 규칙 준수)
    └─ struct Instruction (+ static_assert sizeof == 8)
    └─ using Program = std::vector<Instruction>
    └─ 팩토리 함수 선언 전체 (23개)
    └─ encode_cmp_op / decode_cmp_op / decode_branch_offset
    └─ opcode_name() 선언

[2] gpurt/isa.cpp 작성
    └─ opcode_name(): switch-case → string_view (전체 opcode 커버)
       └─ default 절을 두지 않는다 — opcode 추가 시 -Wswitch가 누락을 잡는다
    └─ 팩토리 함수 구현
       └─ make_v_cmp_f32(): CmpOp → float 비트 인코딩
       └─ make_bra 계열: 오프셋 범위 검사 후 값 변환
       └─ 나머지: Instruction 값 초기화 후 반환

[3] test/unit/test_isa.cpp 작성 (TDD)
    └─ TEST(Isa, InstructionSize)          — sizeof(Instruction) == 8
    └─ TEST(Isa, OpcodeCount)              — Opcode::RET 값 == 22 (0-indexed)
    └─ TEST(Isa, OpcodeNamesFollowScheme)  — 네이밍 규칙 기계 검증
    └─ TEST(Isa, ShapeTokensAreFromTheAllowedSet)
    └─ TEST(Isa, MemoryOpcodesCarryAddressSpace)
    └─ TEST(Isa, OpcodeNameCoversAll)      — 모든 opcode에 name 있음
    └─ TEST(Isa, FactoryFieldsCorrect)     — make_v_mul_f32(1,2,3) 필드값
    └─ TEST(Isa, VcmpImmEncoding)          — CmpOp 인코딩/디코딩 roundtrip
    └─ TEST(Isa, VmovImm)                  — make_v_mov_f32(r,3.14f).imm
    └─ TEST(Isa, BranchOffsetEncoding)     — 경계값(±2^24) 왕복 + 초과 시 예외
    └─ TEST(Isa, ProgramIsVector)          — push_back, size() 확인
```

---

## 설계 결정 및 이유

**네이밍 — `V_<OP>[_<SHAPE>]_<TYPE>` 고정**
: 타입 접미사를 항상 마지막에 두어 `F64`/`F16` 추가 시 기존 이름을 고칠 필요가 없게 했다. 형상 토큰을 별도 자리로 분리한 것은 `MAT3`/`MAT4` 확장이 목표이기 때문이다 — `VADD3` 같은 형태였다면 행렬 버전을 표현할 자리가 없다.

**`V_` 접두사 — 레인별 실행 표식**
: AMD `v_`/`s_` 관례를 따른다. 레지스터 파일에 쓰는 명령어와 워프 상태만 바꾸는 제어 흐름을 이름만으로 구분할 수 있고, 이 구분이 4단계 Scheduler에서 "레인 루프 안/밖"이라는 구현 구조와 그대로 일치한다.

**`F32` (not `FP32`)**
: 명령어 니모닉의 사실상 표준. PTX `add.f32`, AMD `v_add_f32`, WGSL/SPIR-V/Rust `f32`. `FP32`는 성능 스펙 문서의 용어다.

**V_SUB_F32 / V_SUB_VEC3_F32 추가**
: Möller–Trumbore의 `E1 = V1 - V0` 등 벡터 뺄셈이 필수. `ADD + NEG` 조합으로 대체 가능하지만 명시적 SUB가 커널 코드 가독성을 높인다.

**V_CROSS_VEC3_F32 추가**
: 외적은 6 MUL + 3 SUB로 분해 가능하나, Ray-Triangle에서 2회 사용되므로 단일 명령어로 제공. 명령어 수 12 → 2 감소.

**V_RCP_F32 (역수) — VDIV 대신**
: GPU ISA는 일반적으로 나눗셈 대신 역수 명령어를 제공한다 (PTX `rcp.f32`). `f = 1/a` 후 MUL 조합이 DIV 단일 명령어보다 파이프라인 친화적.

**V_CMP_F32 단일 opcode + CmpOp 인코딩**
: 6개 비교 조건을 개별 opcode로 나누면 enum이 비대해진다. imm 필드를 비트 재해석으로 사용하면 opcode 1개로 모든 조건 표현 가능. 팩토리 함수가 인코딩을 캡슐화.

**V_FMA_F32 (in-place 누적)**
: `dst += src0 * src1` 형태. Ray tracing에서 누적 합산 패턴 (색상 블렌딩, AO 샘플 평균)에 사용. 독립적 3-피연산자 FMA는 `Instruction` 구조체 변경 없이 표현 불가능하여 in-place 선택.

**V_MIN_F32 / V_MAX_F32**
: BVH AABB 교차 테스트에서 `tmin = max(tmin, t0)`, `tmax = min(tmax, t1)` 패턴이 핵심. 나중에 BVH 확장 시 별도 ISA 수정 없이 바로 사용 가능.

**벡터/행렬 — 연속 레지스터 관례**
: 별도 벡터 레지스터 파일 없이 스칼라 레지스터 배열을 재사용. `dst`/`src`가 첫 번째 레지스터 인덱스. PTX `v4` 관례와 유사. 레지스터 인덱스 범위 초과 시 Scheduler에서 `std::runtime_error`.

**팩토리 이름이 니모닉과 1:1 (`make_v_add_vec3_f32`)**
: 길지만 opcode를 추가할 때 이름을 고민할 필요가 없다. 축약형(`make_vadd3`)은 형상·타입이 늘어나는 순간 규칙이 무너진다.

---

## 다음 단계

→ [02_memory_model.md](02_memory_model.md) — Host/Device 메모리 할당 및 전송

---

## 현재 코드·검증 대조

- 구현: [gpurt/isa.hpp](../../gpurt/isa.hpp), [gpurt/isa.cpp](../../gpurt/isa.cpp)
- 검증: [test/unit/test_isa.cpp](../../test/unit/test_isa.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./00_machine_spec.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./02_memory_model.md)
