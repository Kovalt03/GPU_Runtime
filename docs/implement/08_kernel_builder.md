# 8단계 — IRBuilder (ISA 위의 커널 작성 층)

> **상태: 구현 완료.** `gpurt/ir_builder.hpp` + `gpurt/ir_builder.cpp` +
> `test/unit/test_ir_builder.cpp` (테스트 20개). 계획과 달라진 부분은 아래
> "구현 결과" 절에 정리했다.

## 목표

커널을 **손으로 어셈블리 치듯 쓰는 상태**에서 벗어난다.

```cpp
// 지금
p.push_back(make_v_sub_vec3_f32(R_E1, R_V1, R_V0));
p.push_back(make_v_cross_vec3_f32(R_H, R_DIR, R_E2));
p.push_back(make_v_dot_vec3_f32(R_A, R_E1, R_H));
p.push_back(make_v_cmp_f32(R_COND, R_A, R_EPS, CmpOp::LT));
branch_to_miss(R_COND);

// 빌더
const auto e1 = k.sub(v1, v0);
const auto h  = k.cross(dir, e2);
const auto a  = k.dot(e1, h);
k.branch_to(miss, k.lt(a, eps));
```

---

## 왜 필요한가 — 이미 대가를 치렀다

6단계 커널 하나를 쓰면서 나온 것:

| 항목 | 수치 |
|------|------|
| 손으로 정한 레지스터 상수 | 24개 |
| `push_back` 호출 | 60회 |
| 분기 오프셋 수동 패치 | 3곳 |

그리고 실제로 낸 버그:

```
· 값을 레지스터 번호 자리에 넘김    3회   ← 컴파일 통과, 조용히 오동작
· 벡터 연산에 스칼라 명령어 사용     3곳   ← V_SUB_F32 vs V_SUB_VEC3_F32
· make_bra_div 인자 개수 착각       1회
· R_TMP 재사용 추적                매번 수동
```

**9단계는 커널을 3개 더 만든다** (vertex, raster, raytrace). 지금 구조로는
이 실수가 세 배가 된다. 그리고 **전부 타입으로 막을 수 있는 종류다.**

---

## 실제 GPU 스택에서의 위치

```
CUDA C / GLSL / HLSL      언어             ← 범위 밖
      ↓ 컴파일러 프론트엔드
PTX / SPIR-V              IR               ← 여기를 만든다
      ↓ 어셈블러
SASS                      기계어            ← 우리 Program (완성)
```

우리는 맨 아래만 갖고 있었다. 이 단계는 **IR 빌더**를 만든다 — LLVM 의
`IRBuilder` 와 같은 자리다.

> **언어 프론트엔드는 범위 밖.** 파서·타입 검사기·최적화 패스를 만드는 것은
> 다른 프로젝트다. 여기서 만드는 것은 C++ 안에 심는 임베디드 DSL 이다.

---

## 설계

### 형상 타입 — 오용을 컴파일 타임에 막는다

```cpp
struct Scalar { static constexpr uint32_t REGISTERS = 1;  static constexpr uint32_t ALIGNMENT = 1; };
struct Vec3   { static constexpr uint32_t REGISTERS = 3;  static constexpr uint32_t ALIGNMENT = 1; };
struct Vec4   { static constexpr uint32_t REGISTERS = 4;  static constexpr uint32_t ALIGNMENT = 4; };
struct Mat4   { static constexpr uint32_t REGISTERS = 16; static constexpr uint32_t ALIGNMENT = 4; };

template <typename Shape>
class Reg {                       // 레지스터 범위에 대한 타입 있는 손잡이
    uint8_t first_;
};
```

`Reg<Vec3>` 와 `Reg<Scalar>` 가 **다른 타입**이므로:

- `k.dot(a, b)` 가 `Reg<Vec3>` 두 개만 받는다 → 스칼라를 넣으면 컴파일 에러
- `k.sub(x, y)` 가 오버로드로 `V_SUB_F32` / `V_SUB_VEC3_F32` 를 알아서 고른다
- `float` 를 레지스터 자리에 넣을 수 없다 → **3회 낸 버그가 원천 차단**

`ALIGNMENT` 는 `V_MATVEC_MAT4_F32` 가 요구하는 4정렬을 할당기가 지키게 한다.

### 레지스터 할당 — bump

```cpp
template <typename Shape> Reg<Shape> alloc();
```

r0 부터 위로 올려가며 배정한다. 해제는 없다.

**충분한 이유**: 빌더는 커널을 만들 때 **한 번만** 실행된다. 런타임 루프는
방출된 코드의 역방향 분기이지 빌더의 루프가 아니므로, 반복 할당이 일어나지
않는다. Möller-Trumbore 가 46개를 쓰는데 253개가 있다.

**한계와 확장**: 진짜 컴파일러는 생존 구간 분석 + 그래프 색칠로 레지스터를
재사용하고, 모자라면 메모리로 밀어낸다(spill). 여기서는 하지 않는다. 다만
할당이 253개를 넘으면 **예외로 끊는다** — 조용히 r253(스레드 좌표)을 덮어쓰는
것이 최악이다.

### 상수 — 값이 들어오는 유일한 문

```cpp
Reg<Scalar> k.constant(1e-6f);
Reg<Vec3>   k.constant(Vec3{0.0f, 0.5f, -2.0f});
```

`V_MOV_F32` 를 방출하고 레지스터를 돌려준다. 호스트 값이 ISA 로 들어가는
통로가 여기 하나로 좁혀지므로, "값을 레지스터 번호 자리에 넣는" 실수가
문법적으로 불가능해진다.

### 스레드 좌표

```cpp
Reg<Scalar> k.thread_x();   // REG_GLOBAL_ID_X 에 묶인 핸들
Reg<Scalar> k.thread_y();
```

예약 레지스터라 할당기가 건드리지 않는다.

### 제어 흐름 — 두 가지를 다 제공한다

```cpp
// (1) 구조적 — 읽기 좋다
k.if_(cond, [&]{ ... });
k.if_else(cond, [&]{ ... }, [&]{ ... });

// (2) 레이블 — 조기 탈출에 맞는다
const auto miss = k.label();
k.branch_to(miss, cond);      // cond != 0 이면 점프
...
k.place(miss);
```

**둘 다 필요하다.** Möller-Trumbore 는 다섯 군데에서 같은 miss 경로로 빠지는데,
중첩 `if_` 로 쓰면 5중 중첩이 된다. 레이블이 그 형태에 맞다.

`if_` 는 한 명령어를 더 쓴다. `BRA_DIV` 는 "0이 아니면 점프"뿐이라, 본문을
건너뛰려면 조건을 뒤집어야 한다:

```
V_CMP_F32 not_cond, cond, zero, EQ
BRA_DIV   not_cond, +본문길이
본문
```

빌더가 이걸 숨긴다. 손으로 쓸 때 매번 헷갈리던 부분이다.

### 메모리

```cpp
Reg<Scalar> k.load(Reg<Scalar> addr, float offset = 0.0f);
void        k.store(Reg<Scalar> addr, Reg<Scalar> value, float offset = 0.0f);

Reg<Vec3>   k.load_vec3(Reg<Scalar> addr, float offset = 0.0f);   // 로드 3회
void        k.store_vec3(Reg<Scalar> addr, Reg<Vec3> value, float offset = 0.0f);
```

`load` 는 `(dst, addr)`, `store` 는 `(addr, value)` 로 인자 순서가 뒤집혀 있어
자주 틀리는데, 빌더에서는 반환값과 인자로 구분되므로 뒤집을 수가 없다.

### 완성

```cpp
Program k.build();     // 필요하면 마지막에 RET 을 붙인다
```

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gpurt/ir_builder.hpp` | `Reg<Shape>`, `IRBuilder` 선언 |
| `gpurt/ir_builder.cpp` | 할당기, 방출, 분기 패치 |
| `test/unit/test_ir_builder.cpp` | 방출 결과가 손으로 쓴 것과 같은지 |

---

## 검증 전략

**빌더의 출력을 직접 비교한다.**

```cpp
TEST(IRBuilder, DotEmitsTheSameProgramAsHandWriting)
{
    IRBuilder k;
    const auto a = k.alloc<Vec3>();
    const auto b = k.alloc<Vec3>();
    const auto d = k.dot(a, b);
    ...
    EXPECT_EQ(k.build(), expected);   // Instruction 비교
}
```

그리고 **6단계 커널을 빌더로 다시 쓴다.** 같은 그림이 나오면 빌더가 맞다는
가장 강한 증거이고, before/after 가 그대로 포트폴리오 자료가 된다.

---

## 구현 순서

```
[1] Reg<Shape> + 형상 타입 + bump 할당기
    └─ 테스트: 정렬, 고갈 시 예외, 예약 레지스터 회피

[2] 산술 방출 (스칼라/VEC3 오버로드)
    └─ 테스트: 방출된 Program 이 손으로 쓴 것과 동일

[3] constant / thread_x / thread_y

[4] 메모리 (load / store / load_vec3 / store_vec3)

[5] 제어 흐름 — label / branch_to / place
    └─ 테스트: 패치된 오프셋이 정확한가, 역방향 분기

[6] 구조적 제어 흐름 — if_ / if_else
    └─ 조건 반전이 필요하므로 [5] 다음

[7] 6단계 커널을 빌더로 이식
    └─ 검증: 같은 PPM 이 나오는가
```

각 단계마다 `ctest` 통과를 유지한다.

---

## 설계 결정 및 이유

**타입으로 형상을 구분하는 것**
: 이 대화에서 낸 버그 6건 중 6건이 타입으로 막힌다. 런타임 검사(`require_register_range`)는 이미 있지만 그것은 **실행할 때** 잡는다. 컴파일 타임에 막는 편이 낫다.

**bump 할당, 해제 없음**
: 빌더는 커널당 한 번 실행되고 런타임 루프는 방출된 코드에 있으므로 반복 할당이 없다. 생존 구간 분석은 이 프로젝트의 관심사가 아니다.

**고갈 시 clamp 가 아니라 예외**
: 조용히 r253 을 덮어쓰면 스레드 좌표가 깨져 원인 추적이 불가능해진다. `make_warp` 에서 lane_count 를 clamp 하지 않기로 한 것과 같은 판단.

**레이블과 구조적 제어 흐름을 모두 두는 것**
: Möller-Trumbore 는 다섯 군데에서 하나의 miss 경로로 빠진다. `if_` 만 있으면 5중 중첩이 된다. 반대로 단순한 조건부 블록에 레이블을 쓰면 장황하다.

**`make_v_*` 팩토리를 그대로 두는 것**
: 빌더가 그것을 호출하는 구조라 1단계 테스트가 하나도 깨지지 않는다. 저수준으로 내려갈 길도 남는다.

**언어 프론트엔드를 만들지 않는 것**
: 파서와 최적화 패스는 별개의 프로젝트다. 임베디드 DSL 로도 "ISA 위에 IR 층을 얹었다"는 요점은 충분히 보인다.

---

## 구현 결과 — 계획과 달라진 것

### [+] 명령어를 비교하는 것만으로는 부족했다

계획한 검증은 "방출된 `Program` 이 손으로 쓴 것과 같은가"였다. 그 테스트
16개는 **분기 오프셋의 원점이 틀려도 전부 통과한다.** 방출된 명령어를 자기
자신과 비교하는 것이라 스케줄러의 해석과 맞는지는 아무도 묻지 않는다.

그래서 실행 테스트를 하나 넣었다 (`GeneratedProgramRunsOnTheScheduler`) —
빌더로 짠 커널을 `myrt_launch` 로 실제로 돌려 역방향 루프, `if_`, `if_else`
결과를 확인한다. **이 테스트만이 오프셋 규약을 검증한다.**

규약: `gpurt/scheduler.cpp` 의 `branch_target` 이 `instr_pc + offset` 으로 풀고
`instr_pc` 는 분기 명령 자신의 주소다. 따라서

```
offset = 라벨 위치 − 분기 명령 위치
```

### [+] 나중에 추가한 원시 연산 3개

9단계 vertex 커널을 스케치하다 빠진 것이 드러났다. 셋 다 **VEC4 조립**이라는
한 가지 필요에서 나왔다.

| 추가 | 이유 |
|------|------|
| `void set(Reg<Scalar> dst, float)` | `V_MOV_F32` 를 **기존** 레지스터에 쓴다. `(x,y,z,1)` 의 w 를 넣을 방법이 달리 없었다 |
| `void load_into(Reg<Scalar> dst, ...)` | 위와 짝. VEC4 의 앞 3칸을 채운다 |
| `Reg<Vec3> Reg<Shape>::xyz()` | 원근 나눗셈이 `clip.xyz * (1/clip.w)` 라서 VEC3 뷰가 필요했다 |

헤더 주석에는 처음부터 "VEC4 의 앞 3개를 VEC3 로 본다"고 적혀 있었는데
`component()`(스칼라 뷰)만 만들어 두고 VEC3 뷰를 빠뜨렸다. **인터페이스를
먼저 설계해도 쓰는 코드를 써 보기 전에는 구멍이 보이지 않는다.**

`xyz()` 만 헤더에 인라인으로 둔다 — `.cpp` 에 두고 명시적 인스턴스화를 하면
`Reg<Scalar>` 에 대고 부르는 실수가 **링크 에러**가 된다. 진단 품질이 존재
이유인 클래스에서 그건 앞뒤가 안 맞는다.

### [−] 제로 레지스터를 캐싱하지 않는다

`copy()` 와 `if_` 가 0 상수를 필요로 한다. 하나 만들어 재사용하는 것이
자연스러워 보이지만 **틀린다.**

빌더의 `constant()` 는 `V_MOV_F32` 를 **지금 위치에** 방출한다. 캐시된 0 이
처음 만들어진 곳이 `if_` 본문 안이면, 그 분기를 건너뛴 실행 경로에서는 이후
모든 사용처가 초기화되지 않은 레지스터를 읽는다.

**bump 할당과 "값이 위치를 갖는다"는 성질이 겹치는 지점이다.** 진짜 컴파일러가
상수 풀을 진입 블록으로 끌어올리는(hoisting) 이유이기도 하다. 여기서는 매번
새로 방출한다 — 명령어 하나 값이 싸다.

### [ ] 아직 안 한 것: 6단계 커널 이식

계획의 `[7] 6단계 커널을 빌더로 이식` 은 하지 않았다. 실행 테스트가 생겨
"빌더가 도는 코드를 만든다"는 것은 확인됐으므로 급하지 않지만, **같은 PPM 이
나오는가**가 여전히 가장 강한 증거이고 before/after 가 포트폴리오 자료가
된다는 판단은 유효하다. 9단계 커널 3개를 다 쓴 뒤에 하는 편이 낫다 — 그때는
빌더가 실전에서 검증된 뒤라 이식이 순수한 리팩터링이 된다.

---

## 다음 단계

→ [09_graphics_pipeline.md](09_graphics_pipeline.md) — 이 빌더로 vertex /
raster / raytrace 커널을 작성한다.

---

## 현재 코드·검증 대조

- 구현: [gpurt/ir_builder.hpp](../../gpurt/ir_builder.hpp), [gpurt/ir_builder.cpp](../../gpurt/ir_builder.cpp)
- 검증: [test/unit/test_ir_builder.cpp](../../test/unit/test_ir_builder.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./07_divergence_benchmark.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./09_graphics_pipeline.md)
