# turbo-graph vs turbovec

<p align="left">
  <a href="README.md">← Docs index</a> ·
  <a href="../README.md">Project README</a>
</p>

Release and PR artifact: reproducible numbers behind the README comparison.
For narrative and quick start, read the [project README](../README.md) first.

## 1) One-line verdict

| Choose **turbovec** | Choose **turbo-graph** |
|---|---|
| Mostly unfiltered global top-k | Most queries include metadata + graph constraints |
| You already build `allowlist=` cheaply before every search | Building `graph ∩ tag ∩ source ∩ time ∩ candidates` is repeated work |
| Minimal deps, smallest API | Need view cache, rerank, explain telemetry |
| Flat Python RAG with framework integrations only | Graph-scoped memory in Rust or Python |

**Important correction:** turbovec is **not** “ANN then post-filter only”. Since v0.3 it applies `allowlist` / `mask` **inside the SIMD kernel**. turbo-graph adds **view compilation, graph expansion, metadata indexes, rerank, and batch planning** — not a replacement for kernel filtering.

Upstream: [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec)

---

## 2) Feature matrix

| Capability | turbovec | turbo-graph |
|---|---|---|
| TurboQuant 2–4 bit encode | ✅ | ✅ shared |
| Train-free online `add` | ✅ | ✅ |
| TQ+ calibration | ✅ | ✅ |
| `.tv` / `.tvim` v3 | ✅ | ✅ |
| `IdMapIndex` + `allowlist` in kernel | ✅ | ✅ |
| `TurboQuantIndex` + `mask` in kernel | ✅ | ✅ |
| Block skip on selective mask | ✅ (`BLOCKS_SKIPPED_BY_MASK`) | ✅ + plan stats |
| Cached `SlotMask` reuse | manual | ✅ `prepare_graph_view*` |
| Weighted graph edges | ❌ | ✅ |
| Tag / source / time indexed filters | ❌ | ✅ |
| `GraphViewPolicy` (budgeted expansion) | ❌ | ✅ |
| Graph + vector rerank | ❌ | ✅ |
| Hybrid candidate-score blend | ❌ | ✅ |
| Batch search shared view | loop | ✅ `*_batch` APIs |
| Explain / debug JSON (`serde`) | ❌ | ✅ optional |
| LangChain / LlamaIndex / Haystack / Agno | ✅ | ✅ on core index |
| `GraphMemoryIndex` in Python | ❌ | ✅ core operating API |

---

## 3) Quantitative benchmarks

### 3.1 Shared core — Recall vs FAISS IndexPQ

100K DB, 1K queries, `k=64`, seed 42. Scripts: `benchmarks/suite/recall_*.py`.

| Dataset | Bit | R@1 TQ / FAISS | Δ |
|---|---:|---|---:|
| OpenAI-1536 | 2 | 0.891 / 0.872 | +1.90pp |
| OpenAI-1536 | 4 | 0.974 / 0.966 | +0.80pp |
| OpenAI-3072 | 2 | 0.929 / 0.912 | +1.70pp |
| OpenAI-3072 | 4 | 0.974 / 0.972 | +0.20pp |
| GloVe-200 | 2 | 0.5637 / 0.5643 | −0.06pp |
| GloVe-200 | 4 | 0.8498 / 0.8410 | +0.88pp |

### 3.2 Shared core — Speed vs FAISS IndexPQFastScan

Median of 5 runs. Scripts: `benchmarks/suite/speed_*`.

| Dim | Bit | Arch | Thr | TQ | FAISS | Gain |
|---:|---:|---|---|---:|---:|---:|
| 1536 | 2 | arm | st | 1.083 | 1.235 | +12.3% |
| 1536 | 2 | arm | mt | 0.103 | 0.115 | +10.4% |
| 1536 | 2 | x86 | st | 1.271 | 1.172 | −8.4% |
| 1536 | 2 | x86 | mt | 0.304 | 0.295 | −3.1% |
| 1536 | 4 | arm | st | 1.992 | 2.450 | +18.7% |
| 1536 | 4 | arm | mt | 0.185 | 0.220 | +15.9% |
| 1536 | 4 | x86 | st | 2.439 | 2.560 | +4.7% |
| 1536 | 4 | x86 | mt | 0.576 | 0.590 | +2.4% |
| 3072 | 2 | arm | st | 2.124 | 2.439 | +12.9% |
| 3072 | 2 | arm | mt | 0.201 | 0.224 | +10.3% |
| 3072 | 2 | x86 | st | 2.657 | 2.582 | −2.9% |
| 3072 | 2 | x86 | mt | 0.626 | 0.590 | −6.1% |
| 3072 | 4 | arm | st | 3.968 | 4.925 | +19.4% |
| 3072 | 4 | arm | mt | 0.375 | 0.448 | +16.3% |
| 3072 | 4 | x86 | st | 5.342 | 5.474 | +2.4% |
| 3072 | 4 | x86 | mt | 1.177 | 1.177 | 0.0% |

**Note:** Recall baseline (IndexPQ + train) ≠ Speed baseline (FastScan). Do not merge into one “FAISS wins/loses” claim.

### 3.3 Compression (100K)

| Key | Ratio |
|---|---:|
| openai_d1536_2bit | 15.8× |
| openai_d1536_4bit | 8.0× |
| openai_d3072_2bit | 15.9× |
| glove_d200_2bit | 14.8× |

Source: `benchmarks/results/compression.json`

### 3.4 turbo-graph layer — `graph_view_bench`

Synthetic 16,384 x 64, k=10, release, 3 measured iters after 3 warmup iters
for steady-state loops. Cold compile is a one-shot measurement.

```bash
cargo run -p turbo-graph --release --example graph_view_bench -- --iters 3 --csv /tmp/graph-view-bench.csv
cargo run -p turbo-graph --release --example graph_view_bench_summary -- /tmp/graph-view-bench.csv
```

#### Selectivity

| Sel. | mask build | cold compile | warm compile | slot_mask | graph_view |
|---:|---:|---:|---:|---:|---:|
| 0.10% | 0.000 | 0.007 | 0.000 | 0.008 | 0.012 |
| 1.00% | 0.001 | 0.022 | 0.000 | 0.009 | 0.012 |
| 100% | 0.034 | 1.392 | 0.000 | 0.054 | 0.057 |

#### Constrained retrieval

Balanced preset view: 24 selected slots, 8 active SIMD blocks.

| Case | ms/query | vs cached |
|---|---:|---:|
| cached mask search | 0.020 | 1.0x |
| rebuild graph+metadata view | 0.048 | 2.4x |
| candidate intersection on cached base view | 0.008 | 0.4x |

#### Post-filter curve (composition penalty)

| fetch_k | recall@10 | post_filter ms |
|---:|---:|---:|
| 10 | 0.00 | 0.062 |
| 640 | 0.20 | 1.684 |
| 8192 | 1.00 | 21.314 |

This models **wrong pipeline order** (global ANN then filter). turbovec avoids it when `allowlist` exists pre-search; turbo-graph targets expensive allowlist **construction**.

---

## 4) Migration gate (release checklist)

1. Document which routes are filter-heavy vs global.
2. Keep `benchmarks/results/*.json` in sync with upstream turbovec when merging core.
3. Attach `graph_view_bench` CSV for graph-layer changes.
4. State explicitly: core parity with turbovec for ANN; delta is graph/metadata path.
5. Pin versions in production — the public 0.1.x line is still Alpha.

---

## 5) PR template

- **Problem:** …
- **vs turbovec:** what stays identical (TurboQuant core) vs what is new (GraphMemoryIndex / caches).
- **Evidence:** §3.1–3.4 tables.
- **When not to merge to product:** filter-light paths — recommend turbovec core only.

## Related files

- `turbo-graph/examples/graph_view_bench.rs`
- `benchmarks/suite/recall_d1536_2bit.py` → `benchmarks/results/recall_d1536_2bit.json`
- `benchmarks/suite/speed_d1536_2bit_arm_mt.py` → `benchmarks/results/speed_d1536_2bit_arm_mt.json`
- `docs/graph_memory_layer.md`
