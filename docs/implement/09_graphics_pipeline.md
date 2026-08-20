# 9단계 — 렌더링 파이프라인 (래스터라이저 + 레이 트레이서)

## 목표

같은 장면을 **두 가지 방식으로** 렌더링한다.

```
draw_mesh_raster(rt, mesh, camera, fb)   삼각형을 화면으로 옮겨 커버 판정
draw_mesh_ray(rt, mesh, camera, light, fb)   픽셀에서 광선을 쏘아 교차 판정
```

두 결과가 **같은 그림이어야 한다**는 것이 서로의 검증이 되고, `divergence_rate`
가 어떻게 달라지는지가 이 프로젝트에서 가장 좋은 비교 자료가 된다.

> IRBuilder를 사용해 커널을 조립한다. 손으로 `make_v_*`를 나열했을 때 생기던
> 레지스터와 분기 배치의 실수를 줄이기 위해서다.

---

## 두 방식의 차이

```
래스터화                              레이 트레이싱
삼각형 → MVP → 화면 좌표              픽셀 → 광선 → 월드 좌표
픽셀이 "이 삼각형 안에 있나"           광선이 "이 삼각형과 만나나"
2D edge function                      Möller-Trumbore
변환 필요 (MVP)                       변환 불필요 (월드 좌표 그대로)
그림자·반사 불가                      그림자·반사 자연스러움
```

**우리 구조에서는 둘이 만나는 지점이 있다.** 픽셀당 스레드로 짜면 양쪽 모두
"나를 덮는 삼각형 중 가장 가까운 것"을 찾는 같은 문제가 되고, 원자적 연산도
깊이 버퍼도 필요 없다.

---

## 공통 기반

### `V_MATVEC_MAT4_F32`

```
reg[dst..+3] = mat4(reg[src0..+15]) × vec4(reg[src1..+3])

row-major:  reg[src0 + row*4 + col]
정렬:       dst / src0 / src1 모두 4의 배수 인덱스
검사:       require_register_range(dst,4) (src0,16) (src1,4)
            require_register_alignment(...,4) × 3
겹침:       모든 출력이 모든 입력을 읽으므로 임시 배열 필수
비용:       16 (곱 16 + 합 12)
```

**정렬을 ALU에서도 강제하는 이유**: `DOC/01`이 `VEC4` 이상에 4정렬을 예약한
근거는 "벡터 로드를 한 트랜잭션으로 모델링"이었다. 레지스터 연산 자체에는
이득이 없지만, 나중에 `V_LD_GLOBAL_MAT4_F32` 가 정렬된 주소를 요구할 때
레지스터 쪽 규칙이 다르면 관례가 둘로 갈린다. 하나로 둔다.

enum 위치는 벡터 ALU 그룹 끝(인덱스 15). 뒤 값이 밀리지만 enum은 직렬화되지
않으므로 무해하고, `test/unit/test_isa.cpp` 의 경계 검사가 그 사실을 알려주는 것이 의도다.

### [A2] 호스트 수학 — `Float3` / `Float4x4`

`gpurt/math3d.hpp`. GPU 에서 도는 코드가 아니라, **런치 전에 호스트가
계산해서 커널에 상수로 넘기는 값**이다. 실제 엔진이 uniform 을 올리는 자리와
같다.

#### 이름을 `Vec3` 가 아니라 `Float3` 로 한 이유

`gpurt/ir_builder.hpp` 가 `Vec3` / `Mat4` 를 이미 **형상 태그**로 쓰고 있다.

```cpp
struct Vec3 { static constexpr uint32_t REGISTERS = 3; ... };   // ir_builder.hpp
Reg<Vec3> e1 = k.sub(v1, v0);                                   // 레지스터 3칸
```

커널 빌더는 두 헤더를 다 include 하므로 충돌은 피할 수 없다. 둘 중 하나가
양보해야 하는데, **ISA 쪽 이름은 ISA 쪽에 두는 게 맞다.** 호스트 절반은
HLSL 의 `float3` / `float4x4` 를 빌려온다.

```
Float3      호스트가 소유하는 값
Reg<Vec3>   장치의 레지스터 3칸 (호스트 메모리를 볼 수 없다)
```

이름이 다르니 **모든 사용처에서 어느 주소 공간의 것인지가 드러난다.**
`hostMem` 과 `deviceMem` 을 섞지 않는 규칙을 타입 이름에서도 지키는 셈이다.
`ThreadBlock` 을 `Block` 과 충돌하지 않게 지은 것과 같은 판단.

#### 행 우선(row-major)

```
m[row * 4 + col]
```

`V_MATVEC_MAT4_F32` 가 레지스터 16개를 읽는 순서와 **같다.** 행렬이 장치로
갈 때 전치가 끼어들지 않으므로, 거꾸로 뒤집힐 관례가 하나 줄어든다.
`test/unit/test_math3d.cpp` 의 `StorageIsRowMajor` 가 이 성질을 결과가 아니라 직접
검사한다 — 전치된 행렬은 "그럴듯하게 틀린" 그림을 만든다.

#### 검증 기준

`look_at` 과 `perspective` 는 부호 하나만 틀려도 그림이 그럴듯하게 나온다.
그래서 테스트를 **기하학적 사실**로 쓴다:

| 테스트 | 성질 |
|--------|------|
| `CrossIsRightHanded` | `x × y = +z`. 뒤집히면 화면이 좌우 반전된다 |
| `LookAtPutsTheTargetDownNegativeZ` | 카메라가 보는 점이 뷰 공간 `-z` 에 온다 |
| `PerspectivePutsDistanceIntoW` | `-5` 에 있는 점의 `w` 가 `5`. w 나눗셈 = 거리 나눗셈 |
| `PerspectiveMapsTheFrustumEdgeToOne` | fov 90°/정사각에서 `(d, 0, -d)` 가 `ndc.x = 1` |
| `PerspectiveMapsNearAndFarToTheClipCube` | near → `-1`, far → `+1` |

`normalize` 는 길이 0 에서 **예외를 던진다.** NaN 을 돌려주면 이후 모든 곱셈을
타고 넘어가 "카메라가 아무것도 안 비추는 빈 화면"으로 나타나는데, 그건 실패로
보이지 않는다.

### [B] 삼각형 버퍼 + 루프

두 방식 모두 삼각형 N개를 순회한다. **여기가 진짜 관문이고, 새 opcode는
필요 없다.**

```
device 메모리 배치
  [ tri0.v0(3) tri0.v1(3) tri0.v2(3) | tri1... ]   삼각형당 36바이트

커널
  r_i = 0                              루프 카운터
  loop:
    addr = tri_base + r_i * 36
    V_LD_GLOBAL_F32 × 9                정점 3개
    ... 판정 ...
    r_i += 1
    V_CMP_F32 r_cond, r_i, r_count, LT
    BRA_DIV r_cond, -(루프 길이)       역방향 분기
```

루프 자체는 **모든 레인이 같은 횟수를 돌므로 divergence를 만들지 않는다.**
갈라짐은 판정 결과에서만 나온다.

### 여기서 `V_LD_GLOBAL_VEC3_F32` 를 추가한다

`DOC/01` 이 예약하고 `DOC/04` 가 측정하겠다고 적어둔 항목이다. 삼각형 루프가
정확히 그 자리다.

```
삼각형 하나 = 정점 3개 × 3성분 = 스칼라 로드 9회   비용 9 × 100 = 900
VEC3 로드면                            3회        비용 3 × 120 ≈ 360
```

실제 GPU 에서 `ld.global.v4.f32` 는 **한 번의 넓은 트랜잭션**으로 처리된다.
삼각형 N개 루프는 로드가 지배적이라, 같은 그림을 두 버전으로 렌더링해
throughput 을 비교하면 coalescing 효과가 그대로 드러난다.

**7단계가 divergence 비용을 보였다면 이쪽은 메모리 접근 패턴 비용을 보인다.**
실제 GPU 최적화의 양대 축이고, 하나만 있으면 반쪽이다.

### 정정 — 지금 비용 모델로는 이 측정이 안 된다

`[5c]` 를 하면서 확인했다. `step_warp` 가 비용을 이렇게 센다:

```cpp
stats_.weighted_lane_ops += lanes * instruction_cost(op);
```

`V_LD_GLOBAL_F32` 는 **레인당** 100이다. 따라서:

```
32 레인이 같은 주소를 읽음      →  32 × 100 = 3,200
32 레인이 흩어진 주소를 읽음    →  32 × 100 = 3,200   ← 똑같다
```

실제 하드웨어는 전자가 1 트랜잭션, 후자가 32 트랜잭션인데 **우리 모델은 둘을
구분하지 못한다.** L1/L2 캐시도 없다.

즉 `V_LD_GLOBAL_VEC3_F32` 를 넣어도 "스칼라 3회와 결과가 같다"까지만 확인되고,
**coalescing 이득은 여전히 측정할 수 없다.** opcode 가 부족한 게 아니라 비용
모델이 접근 패턴을 안 본다.

재려면 워프의 32개 주소가 몇 개의 128바이트 라인에 걸치는지 세야 한다 —
`step_warp` 에서 20줄쯤이고, 그게 선행 조건이다. 그때까지 이 항목은 보류.

**이 보류는 [11_hardware_gaps.md](11_hardware_gaps.md) 의 `[d]` 로 옮겨
적었다.** 거기서 다른 구멍들(Input Assembler, latency hiding, early-Z)과 함께
우선순위를 매겼다 — 이건 opcode 하나가 아니라 실행 모델의 문제고, 손대면
`RESULTS.md` 의 모든 수치가 다시 나온다.

한편 `[5c]` 의 공유 메모리 staging 이 같은 낭비의 다른 절반은 이미 잡았다 —
같은 값을 32 레인이 각자 전역에서 읽던 것을 한 번 읽어 재사용한다.

### [C] 최근접 추적 — 깊이 버퍼 불필요

픽셀당 스레드라 각 스레드가 자기 픽셀을 독점한다.

```
r_best_t = +무한대
r_best_color = 배경
loop:
   더 가까우면 r_best_t / r_best_color 갱신
끝나면 한 번 기록
```

메모리 z버퍼도 원자적 연산도 필요 없다. 삼각형당 스레드였다면 둘 다
필요했을 것이다.

---

## 경로 1 — 래스터라이저

### 2패스 구조

MVP는 **정점당**, 커버 판정은 **픽셀당** 연산이다. 한 커널에 넣으면 스레드를
어느 쪽에 맞출지 답이 없다. 실제 GPU가 이를 파이프라인 분리로 풀고, 우리
런타임은 이미 여러 번 런치할 수 있으므로 추가 기능 없이 흉내낼 수 있다.

```
패스 1 — vertex_kernel        스레드 1개 = 정점 1개
  myrt_launch(vertex_kernel, grid={ceil(3N/32)}, block={32}, args)

  world 정점 로드
  clip = MVP × vec4(world, 1)          ← V_MATVEC_MAT4_F32
  ndc  = clip.xyz * (1/clip.w)         ← V_RCP_F32 + V_MUL_F32
  screen.x = (ndc.x * 0.5 + 0.5) * width
  screen.y = (0.5 - ndc.y * 0.5) * height    ← y 뒤집기
  screen.z = ndc.z                      ← 깊이
  저장
        │
        ▼  device 메모리에 화면 좌표 삼각형이 남는다
        │
패스 2 — raster_kernel        스레드 1개 = 픽셀 1개
  삼각형 순회 → edge function 커버 판정 → 최근접 갱신 → 기록
```

**vertex shader → rasterizer → fragment shader** 그 자체다. 레지스터 압박도
이 분리가 풀어준다 — mat4가 16개를 먹지만 정점 패스만 행렬을 쓰고, 래스터
패스는 화면 좌표만 다룬다.

### 패스 1 상세 — `gpurt/pipeline/vertex.hpp`

#### 버퍼 배치

```
world  버퍼   [ x y z | x y z | ... ]   정점당 12바이트
screen 버퍼   [ x y d | x y d | ... ]   정점당 12바이트
                    └ NDC 깊이
```

`screen` 이 4개가 아니라 3개인 이유: **원근 나눗셈이 패스 1에서 끝나므로**
패스 2에 도달하는 시점에 `w` 는 할 일이 없다.

#### 인자

```cpp
struct VertexStageArgs {
    Float4x4 view_projection;
    size_t   world_offset;    // myrt_device_offset() 으로 얻는다
    size_t   screen_offset;
    uint32_t vertex_count, width, height;
};
```

**주소는 포인터가 아니라 바이트 오프셋이다.** ISA 에 호스트 주소를 역참조할
방법이 없다. 6단계는 버퍼가 하나뿐이라 오프셋 0 을 하드코딩했지만, 패스 1은
버퍼가 둘이라 그럴 수 없다 — 그래서 `MemoryManager::device_offset()` /
`MyGPURuntime::myrt_device_offset()` 을 추가했다. **없는 상태로 짐작하면 두
번째 버퍼가 첫 번째를 덮는다.**

#### 커널이 하는 일

```
[a] mat4 상수 16개    k.set(mvp.component(i), vp.at(row, col))   ← V_MOV_F32 × 16
[b] 범위 검사         live = k.lt(thread_x(), vertex_count)
                      이하 전부 k.if_(live, ...) 안
[c] VEC4 조립         load_into ×3 + set(w, 1.0)
[d] 변환              clip = k.transform(mvp, position)
[e] 원근 나눗셈       ndc = k.scale(clip.xyz(), k.rcp(clip.component(3)))
[f] 뷰포트            x = (ndc.x+1)*(w/2),  y = (1-ndc.y)*(h/2),  z = ndc.z
[g] 저장              screen_offset + id * 12
```

`[b]` 가 없으면 안 된다. 런치는 워프 단위로 올림되므로 `vertex_count` 가 32의
배수가 아니면 남는 레인이 **버퍼 밖을 읽고 밖에 쓴다.**

`[a]` 는 uniform 을 프로그램에 굽는 것이다. `KernelFunc` 이 런치당 한 번
호출되므로 사실상 프레임마다 셰이더를 다시 컴파일하는 셈인데, 시뮬레이터라
비용이 없고 **uniform 업로드 경로를 만들지 않고도 카메라가 움직인다.**

#### 검증 — 호스트 기준선

`project_vertex()` 가 같은 계산을 호스트에서 한다. 커널 결과를 이것과 비교하는
`VertexStageMatchesTheHostProjection` 이 패스 1의 핵심 테스트다.

**호스트 기준선을 먼저 통과시키고 커널을 쓴다.** 커널만 있으면 틀렸을 때
행렬이 문제인지 나눗셈이 문제인지 저장 주소가 문제인지 알 방법이 없다.

### 커버 판정 (2D edge function)

```
e(p, a, b) = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)

세 변의 부호가 모두 같으면 안쪽.
무게중심 좌표는 세 edge 값의 비율로 바로 나온다.
```

3D 외적의 z 성분이므로 **스칼라 연산만으로 된다** — 있는 명령어로 충분하다.

### 분기 vs 프리디케이션 — 7단계와 직결

```
(A) 분기        V_CMP_F32 → BRA_DIV → 안쪽만 갱신
                삼각형 경계에서 워프가 갈라진다

(B) 프리디케이션 커버 여부를 0/1 로 만들어 V_MIN/V_MAX 로 선택
                분기가 없으므로 divergence 0, 대신 항상 양쪽 비용을 낸다
```

7단계가 밝힌 것 — **레인 1개가 갈라져도 16개와 비용이 같다(1.80x)** — 이
선택이 왜 중요한지 설명해준다. 삼각형이 작아 경계 픽셀 비율이 높을수록
(B)가 유리해진다. README 에 약속한 "마스킹 전략 변경 전후 비교"가 이것이다.

---

## 경로 2 — 레이 트레이서

### 변환이 없다

**월드 좌표로 들어온 메쉬는 변환할 필요가 없다.** 6단계 커널이 하던 그대로에
삼각형만 여러 개가 된다.

```
raytrace_kernel               스레드 1개 = 픽셀 1개
  카메라에서 이 픽셀로 가는 광선 생성
  for tri in 0..N-1:
      Möller-Trumbore          ← 6단계에서 완성됨
      t 가 더 작으면 갱신 (최근접)
  조명 계산
  기록
```

### 자유 카메라 — 두 선택지

광선 방향을 카메라 기준에서 월드로 옮겨야 한다.

```
(A) 기저 벡터   호스트에서 right / up / forward 계산
                dir = right*x + up*y + forward
                V_SCALE_VEC3_F32 ×3 + V_ADD_VEC3_F32 ×2 = 5 명령어
                새 opcode 불필요

(B) 행렬        dir = inverse_view × vec4(x, y, -1, 0)
                V_MATVEC_MAT4_F32 1 명령어, 비용 16
```

기능적으로 (A)로 충분하다. (B)는 래스터라이저에서 이미 만든 opcode를
재사용하는 것이라 일관성 면에서 낫다. **래스터라이저를 먼저 만드는 이유
중 하나.**

### 조명 — 새 opcode 불필요

```
normal  = normalize(cross(E1, E2))       V_CROSS_VEC3_F32 + V_NORM_VEC3_F32
to_light = normalize(light_pos - hit)    V_SUB_VEC3_F32 + V_NORM_VEC3_F32
diffuse  = max(0, dot(normal, to_light)) V_DOT_VEC3_F32 + V_MAX_F32
color    = base * diffuse                V_SCALE_VEC3_F32
```

**전부 있는 명령어다.** hit 지점은 `O + t*D` 로 `V_SCALE_VEC3_F32` +
`V_ADD_VEC3_F32`.

그림자를 넣으려면 hit 지점에서 광원으로 광선을 한 번 더 쏘면 된다 — 삼각형
루프를 한 번 더 도는 것이라 구조 변경이 없다. 래스터라이저로는 불가능한
부분이고, 두 방식을 모두 만드는 이유이기도 하다.

---

## 메쉬 입력

```cpp
struct Mesh {
    std::vector<Vec3> vertices;   // 삼각형당 3개
    uint32_t triangle_count() const { return vertices.size() / 3; }
};

Mesh load_obj(const std::string& path);   // 호스트 코드. ISA 무관
```

OBJ 의 `v` / `f` 행만 읽으면 충분하다. 텍스처 좌표와 법선은 나중 문제.
검증용으로 정육면체(삼각형 12개)를 코드로 만드는 것부터 시작한다.

---

## 목표 API

```cpp
MyGPURuntime rt;
const Mesh mesh = load_obj("assets/cube.obj");

const Camera cam{ .eye = {0, 0, 3}, .target = {0, 0, 0}, .up = {0, 1, 0}, .fov = 60.0f };
const Vec3 light{2, 3, 1};

draw_mesh_raster(rt, fb, mesh, cam, width, height);
rt.myrt_sync();     // [STATS] divergence: ...

draw_mesh_ray(rt, fb, mesh, cam, light, width, height);
rt.myrt_sync();     // [STATS] divergence: ...   ← 비교
```

두 래퍼 모두 `myrt_launch` 를 감싸는 얇은 층이다. 런타임은 CUDA 처럼 범용으로
남고, 그래픽스 지식은 이 층에만 둔다.

---

## 설계 결정 및 이유

**래스터 경로에서 행렬 opcode를 사용하는 것**
: MVP 변환은 `V_MATVEC_MAT4_F32`를 실제 렌더링 작업에 연결한다. 레이 경로는
월드 좌표에서 교차하므로 같은 요구가 없으며, 두 경로가 서로 다른 비용을 보이는
이유도 여기서 시작한다.

**둘 다 만드는 것**
: 같은 장면이 같은 그림으로 나와야 하므로 서로가 서로의 검증이 된다. 그리고 divergence 특성이 달라 비교 자료가 된다 — 래스터화는 경계에서, 레이 트레이싱은 그림자 광선에서 갈라진다.

**2패스 (래스터라이저)**
: MVP는 정점당, 커버 판정은 픽셀당이다. 나누면 각 패스가 자기 차원에 맞는 런치를 쓰고 레지스터 압박도 갈라진다. 무엇보다 실제 GPU 구조와 같다.

**MVP를 호스트에서 미리 곱하는 것**
: 행렬 곱은 런치당 한 번이면 되는데 GPU에서 하면 정점마다 반복된다. 실제 엔진도 uniform 으로 올린다. `V_MATMUL_MAT4_F32` 를 뒤로 미룰 수 있는 근거.

**픽셀당 스레드 (양쪽 모두)**
: 삼각형당 스레드가 일반적이지만 원자적 연산이 필요하다. 우리 ISA에 없고, 넣으면 divergence 측정이라는 본래 목적에서 멀어진다. 픽셀당 스레드는 경합이 원천적으로 없고 6단계 구조를 그대로 잇는다.

**깊이 버퍼를 메모리에 두지 않는 것**
: 위와 같은 이유. 스레드가 픽셀을 독점하므로 레지스터 하나면 된다.

---

## 범위 밖 (지금은 안 한다)

- 클리핑 (near plane 뒤 삼각형) — 화면 밖 정점은 커버 판정에서 걸러진다
- 텍스처 샘플링 — 별도 주소 공간과 필터링 필요
- 반사·굴절 — 레이 트레이서에 재귀가 필요한데 ISA에 호출 스택이 없다
- 백페이스 컬링 — edge function 부호로 쉽게 되지만 우선순위 낮음
- `V_MATMUL_MAT4_F32` — MVP 를 호스트에서 미리 곱하므로 불필요
- `V_TRANSPOSE_MAT3_F32` — 법선 변환용. 조명을 넣을 때 재검토
- VEC4 산술 (`V_ADD_VEC4_F32` 등) — vec4 는 MATVEC 입출력으로만 잠깐 존재하고,
  VEC3 명령어가 VEC4 의 앞 3개에 그대로 동작한다 (원근 나눗셈이 그 예)
- `V_ABS_F32` — 양면 삼각형에 필요하지만 `a*a < eps*eps` 로 우회 가능

---

## 8단계에서 넘어오며 추가한 것

9단계를 스케치하다 아래 층에 구멍이 드러났다. 전부 **패스 1 하나를 쓰려다**
나온 것이라, 커널을 한 줄도 쓰기 전에 인터페이스를 완성했다고 믿으면 안 된다는
증거이기도 하다.

| 추가 | 어디 | 없으면 |
|------|------|--------|
| `IRBuilder::set` | 기존 레지스터에 `V_MOV_F32` | VEC4 의 `w = 1` 을 넣을 수 없다 |
| `IRBuilder::load_into` | 기존 레지스터로 로드 | VEC4 의 앞 3칸을 채울 수 없다 |
| `Reg<Shape>::xyz()` | VEC3 뷰 | 원근 나눗셈 `clip.xyz * (1/clip.w)` 을 못 쓴다 |
| `MemoryManager::device_offset` | 포인터 → 바이트 오프셋 | 두 번째 버퍼 주소를 짐작해야 한다 |
| `MyGPURuntime::myrt_device_offset` | 위의 런타임 노출 | 위와 동일 |

---

## 현재 코드·검증 대조

- 구현: [gpurt/pipeline/vertex.cpp](../../gpurt/pipeline/vertex.cpp), [gpurt/pipeline/raster.cpp](../../gpurt/pipeline/raster.cpp), [gpurt/pipeline/raytrace.cpp](../../gpurt/pipeline/raytrace.cpp)
- 검증: [test/unit/test_pipeline.cpp](../../test/unit/test_pipeline.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./08_kernel_builder.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./10_barrier_and_shared_memory.md)
