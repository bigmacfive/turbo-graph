# turbo-graph documentation

<p align="left">
  <a href="../README.md">Project README (EN)</a> ·
  <a href="../README.ko.md">프로젝트 README (KO)</a> ·
  <a href="README.ko.md">문서 목록 (한국어)</a>
</p>

Welcome to the turbo-graph docs. Start here if you landed from code search rather than the root README.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand why this fork exists vs [turbovec](https://github.com/RyanCodrai/turbovec) | [Comparison guide](benchmark_turbo_graph_vs_turbo_vec.md) |
| Look up types and methods | [API reference](api.md) |
| Build graph-scoped memory search in Rust or Python | [Graph memory layer](graph_memory_layer.md) |
| Drop into LangChain / LlamaIndex / Haystack / Agno | [Integrations](#integrations) below |

---

## Core concepts

**Two layers, one crate**

1. **Vector core** (shared with turbovec) — `TurboQuantIndex`, `IdMapIndex`, `.tv` / `.tvim`, kernel-level `allowlist` / `mask`.
2. **Graph layer** (this fork) — `GraphMemoryIndex`, `SlotMask` caches, metadata views, rerank, explain reports.

**When the graph layer helps**

The vector core already filters inside the SIMD kernel. turbo-graph matters when you repeatedly compile `graph ∩ tags ∩ source ∩ time ∩ external candidates` — not when you only need a static id list once per query.

---

## Guides

### [API reference](api.md)

Python-first reference for `TurboQuantIndex`, `IdMapIndex`, filtering, file formats, and the core `GraphMemoryIndex` operating surface.

### [Graph memory layer](graph_memory_layer.md)

Architecture of `GraphMemoryIndex`, `SlotMask`, view caching, presets, batch paths, debug export, and benchmark harnesses.

### [turbo-graph vs turbovec](benchmark_turbo_graph_vs_turbo_vec.md)

Feature matrix, benchmark tables, migration checklist, and PR/release template. Numbers are tied to `benchmarks/results/*.json`.

### [Python graph-memory RAG example](../turbo-graph-python/examples/graph_memory_rag.py)

Runnable graph-memory workflow with candidates, batch search, explain telemetry, cache controls, and persistence.

---

## Integrations

Python wrappers around `IdMapIndex` — same role as turbovec's framework modules, different package name. Python also exposes the core `GraphMemoryIndex` API for add/link/search/explain/cache/persist workflows.

| Framework | Module | Install extra |
|---|---|---|
| [LangChain](integrations/langchain.md) | `turbo_graph.langchain` | `pip install turbo-graph[langchain]` |
| [LlamaIndex](integrations/llama_index.md) | `turbo_graph.llama_index` | `pip install turbo-graph[llama-index]` |
| [Haystack](integrations/haystack.md) | `turbo_graph.haystack` | `pip install turbo-graph[haystack]` |
| [Agno](integrations/agno.md) | `turbo_graph.agno` | `pip install turbo-graph[agno]` |

The deepest graph tuning APIs are Rust-first; Python exposes the operating surface used by RAG services.

---

## Benchmarks & charts

| Artifact | Location |
|---|---|
| Raw JSON (recall, speed, compression) | [`benchmarks/results/`](../benchmarks/results/) |
| Regenerate SVG charts | `python3 benchmarks/create_diagrams.py` |
| Graph view CSV bench | `cargo run -p turbo-graph --release --example graph_view_bench` |

Published charts live in this folder:

| Visual | File |
|---|---|
| Stack (turbovec vs turbo-graph) | `stack.svg` |
| Query path comparison | `query_paths.svg` |
| R@1 delta bars | `recall_delta.svg` |
| Speed win/loss grid | `speed_grid.svg` |
| Selectivity (graph bench) | `selectivity.svg` |
| Migration flow | `migration.svg` |
| Recall curves | `recall_d1536.svg`, `recall_d3072.svg`, `recall_glove.svg` |
| Speed detail | `arm_speed_*.svg`, `x86_speed_*.svg` |
| Compression | `compression.svg` |

---

## External references

- [TurboQuant paper (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [turbovec upstream](https://github.com/RyanCodrai/turbovec)
- [RaBitQ length-renormalization](https://arxiv.org/abs/2405.12497)

---

## Project operations

- [Contributing](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)
- [Security policy](../SECURITY.md)
