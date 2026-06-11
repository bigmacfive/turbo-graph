<p align="center">
  <img src="docs/header.png" alt="turbo-graph: vector + graph metadata retrieval" width="100%">
</p>

<p align="center">
  <a href="https://github.com/bigmacfive/turbo-graph/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bigmacfive/turbo-graph" alt="License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/English-README-0A0A0A?logo=readthedocs&logoColor=white" alt="English README"></a>
  <a href="README.ko.md"><img src="https://img.shields.io/badge/Korean-README-blue" alt="Korean README"></a>
  <a href="https://github.com/RyanCodrai/turbovec"><img src="https://img.shields.io/badge/upstream-turbovec-555" alt="Upstream turbovec"></a>
  <a href="docs/README.md"><img src="https://img.shields.io/badge/docs-index-0969DA" alt="Docs"></a>
</p>

# turbo-graph

**A turbovec-compatible TurboQuant core, plus production graph memory for constrained RAG.**

[turbovec](https://github.com/RyanCodrai/turbovec) compresses embeddings to
2-4 bits per dimension, ingests vectors without a separate training step, and
searches them with SIMD kernels. **turbo-graph keeps that lineage and moves
the hard RAG orchestration into the index**: weighted graph edges,
tag/source/time views, cached `SlotMask` compilation, graph rerank, Python
graph memory, and query telemetry.

> **Operational rule:** choose **turbo-graph** when constraints are the product:
> semantic similarity plus tenant, tag, source, time, graph neighbors, explain
> reports, and cache reuse. Choose **turbovec alone** when you mostly run flat
> global top-k or a cheap id allowlist.

**Contents:** [How this relates to turbovec](#how-this-relates-to-turbovec) ·
[Comparison](#turbovec-vs-turbo-graph) · [Benchmarks](#benchmarks) ·
[Install](#install) · [Quick start](#quick-start) · [Documentation](#documentation)

---

## How this relates to turbovec

This repository is a **fork** of the turbovec codebase. TurboQuant
encoding/search, `.tv` / `.tvim`, and the core Python index APIs are the same
lineage. The new public surface is the graph-memory layer around that core.

<p align="center">
  <img src="docs/stack.svg" alt="turbovec vs turbo-graph stack" width="92%">
</p>

Purple block = **graph layer** in this fork. Shared violet block =
[turbovec](https://github.com/RyanCodrai/turbovec) **TurboQuant core**.

<details>
<summary>Full capability matrix</summary>

| Capability | turbovec | turbo-graph |
|---|---|---|
| TurboQuant encode / search | Yes | Yes, same core |
| `TurboQuantIndex` / `IdMapIndex` | Yes | Yes, compatible API |
| Kernel `allowlist` / `mask` | Yes, since v0.3 | Yes, plus reusable `SlotMask` |
| Graph neighborhood expansion | No | Yes |
| Tag / source / time views | Bring your own SQL | Indexed + cached |
| Graph rerank + BM25 hybrid blend | No | Yes |
| Explain / cache telemetry | Partial | First-class reports |
| Python `GraphMemoryIndex` | No | Core operating API |
| Framework integrations | Yes | Yes |

</details>

---

## turbovec vs turbo-graph

### What turbovec already solves

Upstream turbovec is **not** a naive "vector search, then filter in Python"
design:

- `IdMapIndex.search(..., allowlist=ids)` applies restrictions **inside the
  SIMD kernel**, skipping empty 32-vector blocks before LUT work
  ([#30](https://github.com/RyanCodrai/turbovec/issues/30)).
- `TurboQuantIndex.search(..., mask=...)` does the same for slot masks.
- Results come back as `(nq, min(k, n_allowed))`, so tight filters do not need
  padding or global over-fetch just to recover recall.
- Train-free ingest, TQ+ calibration, RaBitQ scoring correction, and strong
  ARM performance vs FAISS FastScan are inherited here.

turbo-graph **does not replace** kernel filtering. It adds the part that
turbovec leaves in application code: graph expansion, metadata indexes,
candidate-list intersection, reusable view caches, rerank, and explainability.

<p align="center">
  <img src="docs/query_paths.svg" alt="Query path comparison" width="96%">
</p>

Orange boxes = assembly work you still do in app code with turbovec. Indigo
box = what turbo-graph compiles once and reuses.

**Rule of thumb:** turbovec is enough when filters are light. **turbo-graph
wins when constraints are the product** and
`graph ∩ tag ∩ source ∩ time ∩ candidates` is rebuilt across hot queries.

The Python bindings release the GIL around long Rust add/search/prepare/write
paths, so threaded Python services can overlap independent vector and
graph-memory requests instead of serializing on the interpreter lock.

### Should you migrate?

<p align="center">
  <img src="docs/migration.svg" alt="Migration decision flow" width="72%">
</p>

Answer **yes** to three or more:

1. Most queries carry tenant, source, tag, or time constraints.
2. You expand graph neighborhoods before vector search.
3. The same filter predicates repeat in bursts.
4. You manually merge BM25/SQL scores with vector and graph scores.
5. You need production explainability: trace, cache hit, selectivity.
6. `allowlist=` is fine, but **constructing** the allowlist is the bottleneck.

Otherwise stay on turbovec for the flat core and use turbo-graph only for hot
filtered routes.

```python
from turbovec import IdMapIndex      # upstream
from turbo_graph import IdMapIndex   # this repo, compatible core API
```

Full matrix and PR checklist:
[`docs/benchmark_turbo_graph_vs_turbo_vec.md`](docs/benchmark_turbo_graph_vs_turbo_vec.md).

---

## Benchmarks

Numbers below come from `benchmarks/results/*.json`. Regenerate charts with
`python3 benchmarks/create_diagrams.py`.

**Setup (shared core):** 100K database vectors, 1K queries, `k=64`, seed 42,
unit-normalized embeddings.

### Recall vs FAISS IndexPQ

Baseline: FAISS `IndexPQ` with LUT256 and training. **Different** from the
speed baseline below.

<p align="center">
  <img src="docs/recall_delta.svg" alt="R@1 delta summary" width="88%">
</p>

<p align="center">
  <img src="docs/recall_d1536.svg" alt="Recall curves d=1536" width="48%">
  <img src="docs/recall_d3072.svg" alt="Recall curves d=3072" width="48%">
</p>

GloVe 2-bit is the one cell where FAISS edges ahead (-0.06pp). Both converge
by k around 16. Raw data: [`benchmarks/results/`](benchmarks/results/).

### Speed vs FAISS IndexPQFastScan

Median of 5 runs. Green = TurboQuant faster.

<p align="center">
  <img src="docs/speed_grid.svg" alt="Speed win loss grid" width="92%">
</p>

<p align="center">
  <img src="docs/arm_speed_st.svg" alt="ARM ST" width="24%">
  <img src="docs/arm_speed_mt.svg" alt="ARM MT" width="24%">
  <img src="docs/x86_speed_st.svg" alt="x86 ST" width="24%">
  <img src="docs/x86_speed_mt.svg" alt="x86 MT" width="24%">
</p>

ARM wins all 8 configs. x86 2-bit MT is the known gap vs FAISS AVX-512 VBMI.

<details>
<summary>All 16 speed numbers (ms/query)</summary>

| Dim | Bit | Arch | Thr | TQ | FAISS | Gain |
|---:|---:|---|---|---:|---:|---:|
| 1536 | 2 | ARM | ST | 1.083 | 1.235 | +12.3% |
| 1536 | 2 | ARM | MT | 0.103 | 0.115 | +10.4% |
| 1536 | 2 | x86 | ST | 1.271 | 1.172 | -8.4% |
| 1536 | 2 | x86 | MT | 0.304 | 0.295 | -3.1% |
| 1536 | 4 | ARM | ST | 1.992 | 2.450 | +18.7% |
| 1536 | 4 | ARM | MT | 0.185 | 0.220 | +15.9% |
| 1536 | 4 | x86 | ST | 2.439 | 2.560 | +4.7% |
| 1536 | 4 | x86 | MT | 0.576 | 0.590 | +2.4% |
| 3072 | 2 | ARM | ST | 2.124 | 2.439 | +12.9% |
| 3072 | 2 | ARM | MT | 0.201 | 0.224 | +10.3% |
| 3072 | 2 | x86 | ST | 2.657 | 2.582 | -2.9% |
| 3072 | 2 | x86 | MT | 0.626 | 0.590 | -6.1% |
| 3072 | 4 | ARM | ST | 3.968 | 4.925 | +19.4% |
| 3072 | 4 | ARM | MT | 0.375 | 0.448 | +16.3% |
| 3072 | 4 | x86 | ST | 5.342 | 5.474 | +2.4% |
| 3072 | 4 | x86 | MT | 1.177 | 1.177 | 0.0% |

</details>

### Compression (100K vectors)

<p align="center">
  <img src="docs/compression.svg" alt="Compression vs FP32" width="92%">
</p>

10M x 1536d at 2-bit is about **4 GB** of index RAM, vs about 31 GB for
float32 vectors.

### Graph layer

<p align="center">
  <img src="docs/selectivity.svg" alt="Selectivity latency" width="80%">
</p>

Low selectivity is already fast with kernel `SlotMask`. turbo-graph's target
win is repeated compilation and reuse of **`graph ∩ metadata ∩ candidates`**
views.

**Shared limits:** brute-force O(n) scan, not HNSW/IVF; 2-4 bit approximation;
TQ+ needs at least 1000 vectors on the first `add`; pin versions for production
services.

---

## Install

```bash
pip install turbo-graph
cargo add turbo-graph
```

For local development:

```bash
cd turbo-graph-python
python3 -m maturin develop --release
```

Requirements: Rust 1.70+, `dim % 8 == 0`, `bit_width` in `{2, 3, 4}`. x86_64
targets AVX2 (`x86-64-v3`).

---

## Quick start

### Python - turbovec-compatible core

```python
import numpy as np
from turbo_graph import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors.astype(np.float32), ids.astype(np.uint64))

allowed = np.array([1003, 1010, 1042], dtype=np.uint64)
scores, hit_ids = idx.search(query.astype(np.float32), k=10, allowlist=allowed)
```

### Python - graph memory for constrained RAG

```python
import numpy as np
from turbo_graph import GraphMemoryIndex

memory = GraphMemoryIndex(dim=1536, bit_width=4)
memory.add_records(
    embeddings.astype(np.float32),
    [
        {
            "id": 1001,
            "title": "Architecture note",
            "tags": ["architecture"],
            "source": "docs",
            "timestamp_ms": 1_700_000_000_000,
        },
        {
            "id": 1002,
            "title": "Retrieval cache note",
            "tags": ["architecture", "cache"],
            "source": "docs",
            "timestamp_ms": 1_700_000_010_000,
        },
    ],
)
memory.link_bidirectional(1001, 1002, 0.8)
memory.prepare()

hits = memory.search(
    query.astype(np.float32),
    k=10,
    seeds=[1001],
    required_tags=["architecture"],
    allowed_sources=["docs"],
    candidate_ids=[1001, 1002],  # optional BM25/SQL/ACL candidates
)

batch_hits = memory.search_batch(
    queries.astype(np.float32),
    k=10,
    seeds=[1001],
    required_tags=["architecture"],
    allowed_sources=["docs"],
    candidate_ids=[1001, 1002],
)

report = memory.explain(
    query.astype(np.float32),
    k=10,
    seeds=[1001],
    candidate_ids=[1001, 1002, 999],
)
print(report["plan"], report["telemetry"])
```

Runnable version:
[`turbo-graph-python/examples/graph_memory_rag.py`](turbo-graph-python/examples/graph_memory_rag.py).

### Rust - graph layer on the shared core

```rust
use turbo_graph::{GraphMemoryIndex, GraphSearchPreset, MemoryRecord, TurboQuantIndex};

let mut index = TurboQuantIndex::new(1536, 4)?;
index.add(&vectors);
index.prepare();

let mut memory = GraphMemoryIndex::new(1536, 4)?;
memory.add_records(
    &flat_vectors,
    vec![MemoryRecord::new(1001, "Architecture note", ["architecture"])
        .with_source("docs.example")
        .with_timestamp_ms(1_700_000_000_000)],
)?;

let report = memory.explain_graph_search_with_preset(
    &query,
    10,
    &[1001],
    GraphSearchPreset::balanced(),
    &["architecture"],
    &["docs.example"],
    Some(1_700_000_000_000),
    None,
);
println!(
    "hits={} cache_hit={}",
    report.hits.len(),
    report.plan.combined_cache_hit
);
```

---

## Run benchmarks

```bash
# Shared turbovec-style ANN (needs ~/data/py-turboquant/)
python3 benchmarks/download_data.py all
python3 benchmarks/suite/recall_d1536_2bit.py
python3 benchmarks/suite/speed_d1536_2bit_arm_mt.py

# turbo-graph graph/view layer
cargo run -p turbo-graph --release --example graph_view_bench -- --iters 3 --csv /tmp/graph-view-bench.csv
cargo run -p turbo-graph --release --example graph_view_bench_summary -- /tmp/graph-view-bench.csv
```

For release checks:

```bash
scripts/release_check.sh --quick
scripts/release_check.sh --full
```

The release script builds a fresh Python wheel, installs it into a temporary
venv, runs Rust/Python gates, and does not publish, tag, or mutate git history.

---

## Documentation

```
docs/
├── README.md ............... index
├── api.md .................. TurboQuantIndex · IdMapIndex · GraphMemoryIndex
├── graph_memory_layer.md ... views · presets · caches
├── benchmark_turbo_graph_vs_turbo_vec.md
└── integrations/ ........... LangChain · LlamaIndex · Haystack · Agno
```

→ [**Docs index**](docs/README.md) · [API](docs/api.md) ·
[Graph layer](docs/graph_memory_layer.md) ·
[vs turbovec](docs/benchmark_turbo_graph_vs_turbo_vec.md)

---

## Open Source

- [Contributing guide](CONTRIBUTING.md) — issue/PR workflow, test gates, and
  benchmark expectations.
- [Changelog](CHANGELOG.md) — public `0.1.0` release notes plus pre-0.1
  development history.
- [Security policy](SECURITY.md) — supported versions and vulnerability
  reporting.

---

## References

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [turbovec upstream](https://github.com/RyanCodrai/turbovec)
- [RaBitQ length-renormalization](https://arxiv.org/abs/2405.12497)

## Security

See [SECURITY.md](SECURITY.md) for supported versions and vulnerability
reporting.

## License

MIT - see [LICENSE](LICENSE). Core algorithms follow the turbovec lineage; the
graph layer is additional work in this fork.
