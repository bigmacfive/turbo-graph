# turbo-graph 문서

<p align="left">
  <a href="../README.ko.md">프로젝트 README</a> ·
  <a href="../README.md">Project README (EN)</a> ·
  <a href="README.md">Docs index (English)</a>
</p>

코드에서 바로 들어왔다면 여기서 시작하세요. 목적에 맞는 문서만 빠르게 고르면 됩니다.

---

## 어디서부터 볼까

| 목적 | 문서 |
|---|---|
| [turbovec](https://github.com/RyanCodrai/turbovec)와 다른 점 | [비교 가이드](benchmark_turbo_graph_vs_turbo_vec.md) |
| 타입·메서드 레퍼런스 | [API 레퍼런스](api.md) |
| Rust/Python 그래프 메모리 검색 | [Graph memory layer](graph_memory_layer.md) |
| LangChain 등 프레임워크 연동 | 아래 [통합](#통합) |

---

## 핵심 개념

**한 crate, 두 레이어입니다.**

1. **벡터 코어**: turbovec와 공유합니다. `TurboQuantIndex`, `IdMapIndex`, `.tv` / `.tvim`, 커널 `allowlist` / `mask`가 여기에 속합니다.
2. **그래프 레이어**: 이 포크가 더한 부분입니다. `GraphMemoryIndex`, `SlotMask` 캐시, 메타데이터 뷰, rerank, explain 리포트를 제공합니다.

**그래프 레이어가 빛나는 경우**

벡터 코어만으로도 SIMD 커널 안에서 필터링됩니다. turbo-graph는 `graph ∩ tags ∩ source ∩ time ∩ 외부 candidates`를 **매번 다시 조립**하는 비용이 클 때 빛납니다. 쿼리마다 고정 id list 하나만 넘기면 turbovec 코어만으로 충분할 수 있습니다.

---

## 가이드

### [API 레퍼런스](api.md)

`TurboQuantIndex`, `IdMapIndex`, 필터링, 파일 포맷, `GraphMemoryIndex` 운영 API를 한곳에 모았습니다.

### [Graph memory layer](graph_memory_layer.md)

`GraphMemoryIndex` / `SlotMask` 설계, 뷰 캐시, preset, 배치 경로, 디버그 export, 벤치 harness를 설명합니다.

### [turbo-graph vs turbovec](benchmark_turbo_graph_vs_turbo_vec.md)

기능 표, 벤치 수치, 마이그레이션 체크리스트, PR 템플릿을 담았습니다. 수치는 `benchmarks/results/*.json`과 연결됩니다.

### [Python graph-memory RAG 예제](../turbo-graph-python/examples/graph_memory_rag.py)

candidates, batch search, explain telemetry, cache control, persistence가 들어간 실행 가능한 graph-memory workflow입니다.

---

## 통합

Python 래퍼는 `IdMapIndex` 위에 올라갑니다. turbovec 통합 모듈과 역할은 같고 패키지명은 `turbo_graph`입니다. RAG 서비스에서 쓰는 add/link/search/explain/cache/persist 중심의 `GraphMemoryIndex`도 Python에서 바로 쓸 수 있습니다.

| 프레임워크 | 모듈 | 설치 |
|---|---|---|
| [LangChain](integrations/langchain.md) | `turbo_graph.langchain` | `pip install turbo-graph[langchain]` |
| [LlamaIndex](integrations/llama_index.md) | `turbo_graph.llama_index` | `pip install turbo-graph[llama-index]` |
| [Haystack](integrations/haystack.md) | `turbo_graph.haystack` | `pip install turbo-graph[haystack]` |
| [Agno](integrations/agno.md) | `turbo_graph.agno` | `pip install turbo-graph[agno]` |

세밀한 그래프 튜닝 API는 Rust가 우선입니다. Python은 운영에서 자주 쓰는 표면을 먼저 제공합니다.

---

## 벤치마크·차트

| 자료 | 경로 |
|---|---|
| Recall / Speed / Compression JSON | [`benchmarks/results/`](../benchmarks/results/) |
| SVG 차트 재생성 | `python3 benchmarks/create_diagrams.py` |
| Graph view CSV 벤치 | `cargo run -p turbo-graph --release --example graph_view_bench` |

차트·다이어그램:

| 용도 | 파일 |
|---|---|
| turbovec vs turbo-graph 구조 | `stack.svg` |
| 쿼리 경로 비교 | `query_paths.svg` |
| R@1 차이 막대 | `recall_delta.svg` |
| Speed 승/패 그리드 | `speed_grid.svg` |
| 선택도 (graph bench) | `selectivity.svg` |
| 마이그레이션 흐름 | `migration.svg` |
| Recall 곡선 | `recall_d1536.svg`, `recall_d3072.svg`, `recall_glove.svg` |
| Speed 상세 | `arm_speed_*.svg`, `x86_speed_*.svg` |
| 압축 | `compression.svg` |

---

## 외부 참고

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [turbovec upstream](https://github.com/RyanCodrai/turbovec)
- [RaBitQ](https://arxiv.org/abs/2405.12497)

---

## 프로젝트 운영

- [기여 가이드](../CONTRIBUTING.md)
- [변경 기록](../CHANGELOG.md)
- [보안 정책](../SECURITY.md)
