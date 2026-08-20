# 24 — `[shader]` 사용자 프래그먼트 셰이더 + varying 구현 보고서
> **2단계(사용자 표면)가 닫혔다.** 그리고 계획에 있던 `[api]` 를 안 만들고 이걸 만들었다.
>
> 원래 5 번 항목은 `GraphicsContext gfx; gfx.clear(); gfx.draw(); gfx.present();`
> 형태의 명령 층이었다. 바꾼 이유는 하나다 — **그 층은 GL 모양인데, 이 프로젝트가
> 증명하려는 것은 CUDA 모양이다.** 사용자가 정말로 못 하던 것은 `draw` 를 부르는
> 것이 아니라 **자기 계산을 장치에 올리는 것**이었다.

## 무엇이 없었나

시작 지점에서 셰이딩은 `ShadingMode` 열거형 둘이었다 — `Barycentric` 과 `Diffuse`.
둘 다 `gpurt/pipeline/raster_emit.hpp` 안에 명령으로 박혀 있었다. 사용자가 세 번째
효과를 원하면 **라이브러리를 고쳐야** 했고, 그건 쓰는 일이 아니라 만드는 일이다.

그리고 정점이 픽셀에 **아무것도 못 보냈다.** 화면 정점이 4 float 고정(`x, y, depth,
1/w`)이라, 색·법선·UV 가 지나갈 자리가 없었다. `Diffuse` 가 점광을 계산할 수 있었던
것은 위치를 무게중심으로 되짚었기 때문이고, 그건 위치에만 통하는 요령이다.

**둘은 같은 항목이다.** 셰이더만 열면 보간할 것이 없고, varying 만 열면 보간한 것을
쓸 데가 없다. 그래서 하나로 했다.

## 셰이더가 텍스트가 아니라 빌더다

```cpp
using ShadeFn = std::function<void(IRBuilder&, const Fragment&)>;
```

**커널이 조립될 때 한 번 불린다.** 픽셀마다가 아니다 — 이게 이 프로젝트에서 커널이
무엇인지와 정확히 같은 구조다(`KernelFunc` 도 런치마다 한 번). 사용자 함수는 값을
계산하는 것이 아니라 **명령을 낸다.**

`Fragment` 가 GLSL 프래그먼트 스테이지의 입력을 그대로 준다:

| 필드 | 무엇 |
|---|---|
| `varyings[i]`, `varying_count` | 보간된 사용자 값 (최대 `MAX_VARYINGS = 8`) |
| `x`, `y`, `depth` | 픽셀 좌표와 깊이 |
| `w0`, `w1`, `w2` | **원근 보정된** 무게중심 좌표 |
| `out` | 색이 들어갈 `Reg<Vec3>` |

**무게중심을 보정해서 준 것이 판단이다.** 애트리뷰트를 보간하려면 반드시 보정된
무게중심이어야 하는데, `1/w` 를 곱하고 합의 역수로 정규화하는 것은 호출자가 다시
알아낼 일이 아니다. `emit_shade` 가 이미 색에 대해 하고 있던 것을 넘겨주는 쪽으로
바꿨다.

### 사 오는 것 셋

1. **명령어 전체가 열린다.** 분기 포함. 셰이딩 언어가 없으니 문법이 막을 것이 없다.
   대가는 오류가 C++ 오류라는 것 — 셰이딩 문법에 대한 진단이 아니다.
2. **varying 루프가 커널이 생기기 전에 펼쳐진다.** 호스트 `for` 가 장치의 직선
   코드가 된다. 셰이딩 언어라면 컴파일러가 해야 하는 일을 임베디드 DSL 이 공짜로
   가진다.
3. **낸 만큼 값을 매긴다.** divergence·throughput 수치에 셰이더가 들어간다.
   셰이더 안의 분기가 래스터라이저의 분기와 같은 표에 나타난다.

### 셰이더가 `Shading` 안에 산다

처음에는 `RasterStageArgs::shade` 로 놨다. **틀렸다.** `ShadingMode` 는 `Shading`
안에 있고 `Shading` 은 네 경로가 전부 받는데, 함수만 딴 데 있으면 **`Custom` 이 그
함수를 안 보는 경로에 도달한다.** 레이 트레이서가 정확히 그랬다:

```cpp
// gpurt/pipeline/raytrace.cpp — 옮기기 전
if (a.shading.mode == ShadingMode::Diffuse) { ... }
else { /* 무게중심 */ }        // ← Custom 이 조용히 여기로 떨어졌다
```

**조용히 틀린 프레임**이다. 이 저장소가 다른 자리에서 전부 거부로 막아 둔 종류의
버그이고, `Shading` 안으로 옮기니 사라졌다 — 모드와 함수가 같은 값으로 다니면
"모드는 왔는데 함수는 안 온" 상태가 표현 불가능하다.

## 레이 트레이서도 셰이딩한다

옮기고 나니 트레이서에 `Custom` 팔을 다는 것이 자연스러웠고, **경로별로 무엇을 의미할
수 있나**가 그대로 드러났다.

| | walk | 레이 트레이서 | tiled / shared |
|---|---|---|---|
| 사용자 셰이더 | ○ | ○ | 거부 |
| varying | ○ | 나를 정점 스테이지가 없다 | 거부 |
| `f.depth` | NDC, 깊이 버퍼와 같은 것 | 레이 파라미터 `t` | — |

**무게중심의 이름이 주장이다.** walk 은 투영된 삼각형 위에서 보간하고 원근 보정을
하는데, 트레이서는 Möller-Trumbore 의 `u`, `v` 를 월드 공간 해에서 바로 얻는다
(보정할 투영이 없으므로 정확하다). `w1 = u`, `w2 = v`, `w0 = 1-u-v` 로 놓으면
**셰이더 하나가 두 경로에서 같은 그림**을 낸다 — 그리고 그것이 테스트다
(`OneShaderDrawsTheSamePictureDownTheWalkAndTheTracer`). 무게중심 셋을 서로 다른
계수로 쓰게 짜서, 순서가 돌아갔으면 못 살아남게 했다.

**`depth` 는 정직하게 다른 값이다.** 둘 다 "얼마나 멀리" 이고 둘 다 히트를 같은
순서로 놓지만, 서로의 숫자를 줄 방법이 없다. 깊이를 **비교만** 하는 셰이더는 이식
가능하고, `[0,1]` 을 기대하는 셰이더는 아니다. 헤더에 그렇게 적었다.

**지는 프래그먼트에서도 돈다.** `predicated` 는 아무 레인도 마스킹하지 않고 색을
계산한 뒤 블렌드로 버리고, 트레이서는 삼각형 후보마다 셰이딩한다. `out` 을 쓰는 것은
안전하고 **그 외의 효과는 아니다** — 셰이더 안의 store 나 atomic 은 프레임에 안
나오는 후보에 대해서도 발사된다.

## varying 이 지나가는 길

```
정점 버퍼 ─┐
애트리뷰트 ─┴→ 패스 1 → 화면 정점 [x, y, depth, 1/w | v0 v1 … vN-1]
                                    └ 4 float 고정 ┘└ varying_count ┘
                                            ↓
                    패스 2: 세 정점의 슬롯 i 를 보정 무게중심으로 섞는다
```

`screen_vertex_bytes(varyings)` 가 스트라이드를 쥔다. **패스 2 가 스트라이드를 인자로
받게 바꾼 것이 이 항목의 실제 diff 중 가장 큰 부분이다** — 그전에는 `SCREEN_VERTEX_BYTES`
상수가 여섯 자리에 박혀 있었다.

**장치가 애트리뷰트를 통과시키는 것도 명령이다.** 패스 1 이 varying 마다 로드 하나와
저장 하나를 낸다. 호스트에서 미리 섞어 올리는 쪽이 명령은 아끼지만, 그러면 정점
스테이지가 프로그램 가능한 스테이지라는 것이 거짓이 된다.

## 검증이 공짜로 붙었다

`Barycentric` 은 이미 호스트 레퍼런스로 픽셀 단위 검증이 되어 있다. 그래서:

> 정점 애트리뷰트를 `{0,0,1, 1,0,0, 0,1,0}` 으로 두고, varying 을 그대로 색으로 쓰는
> 사용자 셰이더는 내장 `Barycentric` 과 **픽셀 단위로 같은 프레임**을 낸다.

varying 경로가 기존 검증을 **물려받는다** — 새 오라클을 안 지어도 된다. 애트리뷰트가
단위행렬이 아니라 회전된 이유는 `emit_shade` 가 채널을 돌리기 때문이다
(`dst[0]=w1, dst[1]=w2, dst[2]=w0`). 그걸 모르고 단위행렬로 시작해서 테스트가 한 번
실패했고, **실패가 알려준 것이 내장 경로의 관례**였다.

테스트 셋:

| 테스트 | 무엇을 잡나 |
|---|---|
| `AVaryingInterpolatesTheWayTheBuiltInShadingDoes` | 보간이 내장과 같다 (픽셀 단위) |
| `AShaderMaySpendTheWholeInstructionSet` | 셰이더 안의 분기가 돈다 |
| `CustomShadingWithNothingToEmitIsRefused` | walk: `Custom` + 빈 `shade` 는 거부 |
| `OneShaderDrawsTheSamePictureDownTheWalkAndTheTracer` | 두 경로의 무게중심 이름이 같다 |
| `TheTracerRefusesCustomShadingWithNothingToEmit` | 트레이서: 같은 거부 |

## 거부하는 것

**타일드/공유 경로는 varying 을 못 나른다.** 타일 목록이 정점당 4 float 로 **고정**
이고(`3 * SCREEN_VERTEX_BYTES` 가 커널 안에 있다), 늘리면 타일 하나에 들어가는 삼각형
수가 varying 수에 따라 변한다. **조용히 버리는 대신 거부한다** — `DOC/15` 이 `Diffuse`
에 대해 이미 같은 자리에 서 있었고, 메시지에 두 가지를 함께 적었다.

`varying_count > MAX_VARYINGS` 도 거부한다. 8 은 레지스터 파일에서 온 숫자다.

## 현재 코드·검증 대조

- 구현: [gpurt/pipeline/types.hpp](../../gpurt/pipeline/types.hpp), [gpurt/pipeline/raster_emit.hpp](../../gpurt/pipeline/raster_emit.hpp)
- 검증: [test/unit/test_pipeline.cpp](../../test/unit/test_pipeline.cpp), [apps/hello_shader.cpp](../../apps/hello_shader.cpp)
- 측정이 있는 단계는 해당 벤치마크의 소스와 `test/benchmark/RESULTS.md`를 함께
  기준으로 삼는다. 이 문서의 수치는 실행 당시의 결과이며, 현재 재현값은 실행 환경에
  따라 달라질 수 있다.

---

[← 이전 문서](./23_bandwidth.md) · [구현 노트 목차](./README.md) · [다음 문서 →](./25_bvh.md)
