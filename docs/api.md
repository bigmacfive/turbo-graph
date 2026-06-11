# API Reference

<p align="left">
  <a href="README.md">← Docs index</a> ·
  <a href="../README.md">Project README</a>
</p>

`turbo_graph` exposes three index surfaces and three on-disk formats. The first two match [turbovec](https://github.com/RyanCodrai/turbovec); the third is unique to this fork.

| Type | Role |
|---|---|
| [`TurboQuantIndex`](#turboquantindex) | Positional slots, O(1) `swap_remove` |
| [`IdMapIndex`](#idmapindex) | Stable `u64` ids + kernel `allowlist` |
| [`GraphMemoryIndex`](#graphmemoryindex-python) | Graph + metadata views, rerank, explain, cache stats |
| [File formats](#file-formats) | `.tv`, `.tvim`, `.tvgm` |

Examples below use **Python** unless noted. Rust signatures live in rustdoc on each type.

---

## `TurboQuantIndex`

Positional index. Each vector is identified by its insertion slot (`0..n`). Fast and small, but external references to slots are invalidated by `swap_remove`. If you need stable ids, use [`IdMapIndex`](#idmapindex).

```python
from turbo_graph import TurboQuantIndex

idx = TurboQuantIndex(dim=1536, bit_width=4)
idx.add(vectors)                        # np.ndarray of shape (n, dim), float32
scores, indices = idx.search(queries, k=10)

idx.swap_remove(5)                      # O(1); the previously-last vector moves into slot 5

idx.write("index.tv")                   # .tv format
loaded = TurboQuantIndex.load("index.tv")
```

`dim` is optional. Omit it to let the index pick up the dimensionality from the first batch of vectors:

```python
idx = TurboQuantIndex(bit_width=4)      # dim inferred on first add
idx.add(vectors)                         # locks dim to vectors.shape[1]
```

Before the first add, `idx.dim` is `None`, `len(idx)` is `0`, and `search()` returns empty results.

### Methods

| Method | Notes |
|---|---|
| `TurboQuantIndex(dim=None, bit_width=4)` | `bit_width ∈ {2, 3, 4}`. `dim` is optional; when omitted it is inferred from the first `add` call. |
| `add(vectors)` | `vectors` is a contiguous float32 array of shape `(n, dim)`. On a lazy index the first call locks `dim`; subsequent calls must match. Raises `ValueError` on dim mismatch. |
| `search(queries, k, *, mask=None)` | Returns `(scores, indices)`, both shape `(nq, effective_k)`. Indices are `int64` slot positions. `mask` is an optional `bool` array of length `len(idx)`; when given, only slots with `mask[i] == True` contribute. `effective_k = min(k, mask.sum())`. |
| `swap_remove(idx)` | O(1). Moves the last vector into `idx`; returns the previous position of that moved vector (so external refs can be updated if needed). |
| `prepare()` | Optional. Eagerly builds the rotation matrix, Lloyd-Max centroids and SIMD-blocked layout so the first `search` call doesn't pay the one-time cost. No-op on a lazy index that hasn't seen its first add. |
| `write(path)` / `load(path)` | `.tv` format. |
| `len(idx)` / `idx.dim` / `idx.bit_width` | Introspection. `idx.dim` returns `int` once committed, or `None` on a lazy index that hasn't seen its first add. |

### `swap_remove` semantics

`swap_remove(i)` is named to match Rust's [`Vec::swap_remove`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove): the last element moves into slot `i`, and the vector is truncated by one. It is **not** a shift (FAISS's `IndexPQ::remove_ids` behaviour). Order is not preserved; slot indices of vectors you didn't delete may now point at different vectors than before.

Use [`IdMapIndex`](#idmapindex) if external references have to stay stable across deletes.

---

## `IdMapIndex`

Stable-id wrapper around `TurboQuantIndex`. Roughly equivalent to FAISS's `IndexIDMap2` — hash-table backed, O(1) `remove(id)`.

```python
import numpy as np
from turbo_graph import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors, np.array([1001, 1002, 1003], dtype=np.uint64))

scores, ids = idx.search(queries, k=10)   # ids are uint64 external ids

idx.remove(1002)                           # O(1) by id
assert 1003 in idx                         # __contains__ sugar

idx.write("index.tvim")                    # .tvim format
loaded = IdMapIndex.load("index.tvim")
```

As with [`TurboQuantIndex`](#turboquantindex), `dim` is optional and gets inferred from the first `add_with_ids` call:

```python
idx = IdMapIndex(bit_width=4)            # dim inferred on first add
idx.add_with_ids(vectors, ids)           # locks dim to vectors.shape[1]
```

### Methods

| Method | Notes |
|---|---|
| `IdMapIndex(dim=None, bit_width=4)` | `dim` is optional; when omitted it is inferred from the first `add_with_ids` call. |
| `add_with_ids(vectors, ids)` | `ids` is a `uint64` array with length `vectors.shape[0]`. On a lazy index the first call locks `dim`. Raises `ValueError` on dim mismatch, duplicate ids, or `len(ids) != vectors.shape[0]`. |
| `remove(id) -> bool` | `True` if the id was present and removed, `False` otherwise. O(1). |
| `search(queries, k, *, allowlist=None)` | Returns `(scores, ids)` — `ids` are `uint64` external ids. `allowlist` is an optional `uint64` array of ids; when given, results are restricted to those ids and `effective_k = min(k, len(allowlist))`. Raises `ValueError` on empty allowlist and `KeyError` on unknown ids. |
| `contains(id)` / `id in idx` | Membership. |
| `write(path)` / `load(path)` | `.tvim` format. |
| `len(idx)` / `idx.dim` / `idx.bit_width` / `prepare()` | Same as `TurboQuantIndex`. |

### When to use which

- `TurboQuantIndex` — you never delete, or you're fine with positional ids.
- `IdMapIndex` — you need stable external ids (e.g. string-id → vector mapping maintained by the caller).

All the framework integrations (LangChain, LlamaIndex, Haystack) use `IdMapIndex` internally for exactly this reason.

---

## Filtering

Both index types support restricting the returned top-`k` to a caller-supplied subset of vectors. Unlike post-filtering (search then drop), the kernel never inserts disallowed vectors into the per-query heap, so you always get up to `k` results from the allowed set rather than fewer.

```python
# IdMapIndex — allowlist of external ids (typical use)
allowed = np.array([1003, 1010, 1042], dtype=np.uint64)
scores, ids = idx.search(queries, k=10, allowlist=allowed)
# scores.shape == (nq, min(k, len(allowed))) == (nq, 3)

# TurboQuantIndex — bool mask over slots
mask = np.ones(len(idx), dtype=bool)
mask[disabled_slots] = False
scores, slots = idx.search(queries, k=10, mask=mask)
```

The output shape is `(nq, min(k, n_allowed))` — same shrinking behaviour you already see when `k > len(idx)`. No `-1` / `NaN` padding; pad on the caller side if you need a fixed-width batch.

Common use cases:

- Hybrid retrieval where a SQL/BM25 stage produces a candidate id set.
- Access control or multi-tenant queries (only return ids the caller can see).
- Time-windowed search (e.g. only documents from the last 7 days).

### Rust: cached packed masks

Rust callers that reuse the same candidate set can build a `SlotMask` once and
pass it directly to `TurboQuantIndex::search_with_slot_mask`. This avoids
rebuilding and packing a full boolean mask on every query. `SlotMask` also
caches the allowed count and the non-empty 32-slot SIMD blocks, so selective
views can iterate only candidate blocks instead of probing every block in the
index.

```rust
use turbo_graph::{SlotMask, TurboQuantIndex};

let mut mask = SlotMask::new(index.len());
for slot in graph_view_slots {
    mask.allow(slot);
}

let results = index.search_with_slot_mask(&query, 10, &mask);
```

`SlotMask` also supports `union_with` and `intersect_with`, which are useful for
composing graph, tag, time-window, and ACL views before search. Use
`allowed_slots()` when debug/export/planning code needs the selected slot ids
without scanning `0..len`.

---

## `GraphMemoryIndex` (Python)

Python exposes the core operating surface for graph-scoped RAG services:
add/link/search/explain/cache/persist. Rust keeps the deepest tuning surface.
For a runnable Python workflow that combines graph, metadata, upstream
candidate ids, batch search, explain telemetry, cache trimming, and persistence,
see `turbo-graph-python/examples/graph_memory_rag.py`.

The Python extension releases the GIL around long Rust add/search/prepare/write
paths for `TurboQuantIndex`, `IdMapIndex`, and `GraphMemoryIndex`, so threaded
services can overlap independent requests.

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
            "tags": ["architecture", "memory"],
            "source": "docs.example",
            "timestamp_ms": 1_700_000_000_000,
        },
        {
            "id": 1002,
            "title": "TurboQuant note",
            "tags": ["architecture", "vector"],
            "source": "docs.example",
            "timestamp_ms": 1_700_100_000_000,
        },
    ],
)
memory.link_bidirectional(1001, 1002, 0.8)

hits = memory.search(
    query.astype(np.float32),
    k=10,
    seeds=[1001],
    max_hops=2,
    required_tags=["architecture"],
    allowed_sources=["docs.example"],
    start_ms=1_700_000_000_000,
    candidate_ids=[1001, 1002],  # Optional upstream BM25/SQL/ACL candidates.
)
batch_hits = memory.search_batch(
    batch_queries.astype(np.float32),
    k=10,
    seeds=[1001],
    required_tags=["architecture"],
    candidate_ids=[1001, 1002],
)

report = memory.explain(
    query.astype(np.float32),
    k=10,
    seeds=[1001],
    candidate_ids=[1001, 1002, 999],
)
stats = memory.cache_stats()
budget = memory.trim_caches_for_preset("balanced")

memory.write("memory.tv", "memory.tvgm")
loaded = GraphMemoryIndex.load("memory.tv", "memory.tvgm")
```

### Python methods

| Method | Notes |
|---|---|
| `GraphMemoryIndex(dim, bit_width=4)` | Eager graph-memory index. `dim` must be a positive multiple of 8. |
| `add_records(vectors, records)` | `vectors` is contiguous float32 `(n, dim)`. `records` is a sequence of dicts with `id`, `title`, `tags`, optional `source`, optional `timestamp_ms`. |
| `add_node(id, title, vector, tags, source=None, timestamp_ms=None)` | Single-record convenience wrapper. |
| `link_directed(from_id, to_id, weight)` / `link_bidirectional(a, b, weight)` | Weighted graph edges. Raises `ValueError` for missing ids or invalid weights. |
| `search(query, k, seeds, *, max_hops=2, required_tags=None, allowed_sources=None, start_ms=None, end_ms=None, candidate_ids=None)` | Returns `list[dict]` hits with id, score, title, tags, source, and timestamp. `candidate_ids` intersects an upstream retriever/ACL list with graph and metadata before vector search. |
| `search_batch(queries, k, seeds, *, max_hops=2, required_tags=None, allowed_sources=None, start_ms=None, end_ms=None, candidate_ids=None)` | Returns `list[list[dict]]` and compiles the shared graph/metadata/candidate view once for all query rows. |
| `explain(query, k, seeds, *, preset="balanced", required_tags=None, allowed_sources=None, start_ms=None, end_ms=None, candidate_ids=None)` | Returns a dict with `hits`, `plan`, `prefetch_k`, `telemetry`, and `trace`. Presets: `low_latency`, `balanced`, `broad`. With `candidate_ids`, `plan` also reports candidate input/live/missing/duplicate counts and ratios. |
| `cache_stats()` | Returns cache sizes, access counts, and hit/miss ratios. |
| `cache_budget_for_preset(preset)` / `trim_caches_for_preset(preset)` | Inspect or apply bounded cache budgets derived from `low_latency`, `balanced`, or `broad`. |
| `clear_query_caches()` / `clear_metadata_caches()` / `clear_all_caches()` | Drop cached graph/search or metadata masks without changing stored memories. |
| `trim_query_caches(n)` / `trim_metadata_caches(n)` / `trim_all_caches(n)` | Bound internal cache maps for long-running local memory services. |
| `record(id)` / `neighbors(id)` | Lightweight inspection helpers for API/debug surfaces. `record` returns `None` for unknown ids; `neighbors` raises `KeyError`. |
| `remove_node(id)` / `contains(id)` / `id in memory` / `slot_of(id)` | Stable-id graph memory maintenance. |
| `prepare()` | Warms the underlying TurboQuant search caches. |
| `write(index_path, graph_path)` / `load(index_path, graph_path)` | Writes `.tv` plus `.tvgm` graph sidecar. |

## `GraphMemoryIndex` (Rust)

`GraphMemoryIndex` is a local context-memory layer for graph-scoped retrieval.
It stores vectors in a raw `TurboQuantIndex`, keeps stable `id ↔ slot` side
tables, stores node metadata and weighted graph edges, and compiles graph views
into cached `SlotMask` values before search.

```rust
use turbo_graph::{GraphMemoryIndex, GraphRerankConfig, GraphSearchPreset, GraphViewPolicy, MemoryRecord};

let mut memory = GraphMemoryIndex::new(1536, 4)?;
memory.add_records(
    &embeddings,
    vec![
        MemoryRecord::new(1001, "Architecture note", ["architecture", "memory"])
            .with_source("kuku.mom")
            .with_timestamp_ms(1_700_000_000_000),
        MemoryRecord::new(1002, "TurboQuant note", ["architecture", "vector"])
            .with_source("kuku.mom")
            .with_timestamp_ms(1_700_100_000_000),
    ],
)?;
memory.link_bidirectional(1001, 1002, 0.8)?;

let hits = memory.search_graph_view(&query, 10, &[1001], 2, &["architecture"]);

let preset = GraphSearchPreset::balanced()
    .with_target_active_blocks(16)
    .with_nodes_per_active_block(24);
let explained_from_preset = memory.explain_graph_search_with_preset(
    &query,
    10,
    &[1001],
    preset,
    &["architecture"],
    &["kuku.mom"],
    Some(1_700_000_000_000),
    None,
);

let policy = GraphViewPolicy::new(2)
    .with_max_nodes(128)
    .with_max_active_blocks(16)
    .with_min_path_weight(0.25);
let report = memory.search_graph_view_with_policy_metadata_plan(
    &query,
    10,
    &[1001],
    policy,
    &["architecture"],
    &["kuku.mom"],
    Some(1_700_000_000_000),
    None,
);

let reranked = memory.search_graph_view_with_policy_metadata_rerank(
    &query,
    10,
    &[1001],
    policy,
    &["architecture"],
    &["kuku.mom"],
    Some(1_700_000_000_000),
    None,
    GraphRerankConfig::new(1.0, 0.2),
);

let timed = memory.search_graph_view_with_policy_metadata_rerank_timed(
    &query,
    10,
    &[1001],
    policy,
    &["architecture"],
    &["kuku.mom"],
    Some(1_700_000_000_000),
    None,
    GraphRerankConfig::new(1.0, 0.2),
);

let explained = memory.explain_graph_search_with_policy_metadata_rerank_timed(
    &query,
    10,
    &[1001],
    policy,
    &["architecture"],
    &["kuku.mom"],
    Some(1_700_000_000_000),
    None,
    GraphRerankConfig::new(1.0, 0.2),
);
let snapshot = explained.debug_snapshot();
```

For batch ingest, use `add_records(vectors, records)` to encode many vectors in
one TurboQuant pass instead of calling `add_node` repeatedly.
Use `replace_embedding(id, embedding)` when only the vector changes, and
`replace_record_metadata(record)` when title, tags, source, or timestamp change
without changing the vector or graph edges. Metadata-only replacement refreshes
tag/source/time and combined graph+metadata caches while preserving raw graph
view caches.

For long-running local memory processes, use `cache_stats()` to inspect
`GraphMemoryCacheStats`: graph views, policy visits, policy views, combined
graph+metadata views, tag masks, source masks, time masks, and total entries.
`GraphSearchPreset::cache_budget(total_slots)` derives a
`GraphMemoryCacheBudget` from the preset's active-block and hop targets;
`GraphMemoryIndex::cache_budget_for_preset(preset)` applies that to the current
index size. `trim_caches_to_budget(budget)` and `trim_caches_for_preset(preset)`
bound each internal cache independently for long-running services with many ad
hoc graph/search filters. `clear_query_caches()`, `clear_metadata_caches()`, and
`clear_all_caches()` drop cached masks without changing stored memories or
edges.

`search_graph_view_with_stats` returns the same hits plus `GraphViewStats`
(`total_slots`, `selected_slots`, `cache_hit`, and `selectivity()`), which is
useful for query planning and benchmarking graph-view pruning. For filtered
searches, `cache_hit` is true when either the graph stage or the combined
graph+metadata view was reused.

Tag filters use an in-memory tag-to-id index and cached tag masks rather than
scanning every record for each query. Source filters use the same indexed-mask
pattern, and timestamp windows use a sorted `(timestamp_ms, id)` index. Use
`tag_view_mask(tag)`, `source_view_mask(source)`, and
`time_range_view_mask(start_ms, end_ms)` when a caller wants to compose its own
cached metadata view.

`search_graph_view_with_metadata` combines graph seeds, required tags, allowed
sources, and a half-open timestamp window `[start_ms, end_ms)` before vector
search. This is the direct path for search-engine views such as
`graph neighborhood ∩ site/domain ∩ recency`.
Use `search_graph_view_with_metadata_batch_plan` when several query embeddings
share the same graph+metadata context; it compiles the view once, runs one
TurboQuant batch search through that cached `SlotMask`, and returns one hit list
per query row plus the shared plan telemetry. Use
`search_graph_view_with_policy_metadata_rerank_batch` when those rows should
also share one policy expansion and graph-aware rerank configuration.
Use `search_graph_view_with_policy_metadata_candidate_scores_hybrid_batch` when
the same external candidate-score list should also be blended into every query
row.

When another planner already has candidate memory ids, use
`candidate_id_mask(ids)` or
`search_graph_view_with_metadata_candidates_plan`. This intersects
`graph neighborhood ∩ metadata filters ∩ candidate ids` before TurboQuant
scoring, which is the right handoff for BM25, keyword/entity retrieval, ACL, or
dedup stages. Candidate ids are not added to the combined-view cache key because
they are usually query-specific; the cached graph+metadata base view is reused,
then intersected with the candidate mask. Unknown/stale candidate ids are
ignored. Candidate plans also report raw candidate input count, unique live
candidate slots, duplicate ids, and missing ids so upstream retrievers can be
tuned without parsing logs.
Use `search_graph_view_with_policy_metadata_candidates_rerank` or its `_timed`
variant when that candidate set should still be reranked by graph path strength
after TurboQuant candidate retrieval. Use the `_batch` variant when the same
candidate-id list should constrain several query embeddings. For UI/debug export, use
`explain_graph_search_with_policy_metadata_candidates_rerank_timed` and call
`debug_snapshot()` on its report; the candidate snapshot includes graph slots,
metadata slots, candidate input quality, selected slots, timing, hits, trace
nodes, and trace edges.
If the external planner has scores, pass `(id, score)` pairs to
`search_graph_view_with_policy_metadata_candidate_scores_hybrid` or its `_timed`
variant. `GraphHybridRerankConfig` blends TurboQuant vector score, graph path
score, and candidate score, and `GraphHybridHit` exposes all three components
for ranking inspection. `GraphCandidateScoreNormalization::{None, MinMax,
MaxAbs}` can scale external scores before blending; keep `None` for already
calibrated scores, use `MinMax` for raw BM25/keyword scores, and use `MaxAbs`
when signed candidate priors should keep their sign. Normalization is computed
over the candidates that survive the final graph+metadata+candidate view, so a
high-scoring stale or out-of-view candidate cannot compress in-view scores. The
hybrid path builds candidate masks, score maps, duplicate counts, and stale-id
counts in one pass over the external candidate-score list.
The `_batch` variant reuses that same candidate-score map and final candidate
mask across all query rows before sorting/truncating hybrid hits per row.
`explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed` adds
the graph trace, and its `debug_snapshot()` preserves `candidate_score` on ranked
hits and trace nodes.

For repeated local-context searches, use
`search_graph_view_with_metadata_plan`. It returns `GraphPlannedSearchReport`
with `GraphSearchPlan` telemetry: graph slots before metadata pruning, final
selected slots, active SIMD blocks, graph-cache hit, combined graph+metadata
cache hit, and selectivity helpers. The combined cache key normalizes seed ids,
required tags, allowed sources, hop count, and timestamp range, so equivalent
filter orders reuse the same compiled `SlotMask`.

For larger memory graphs, `GraphViewPolicy` adds weighted, budgeted graph
expansion. `graph_view_mask_with_policy_stats` and
`search_graph_view_with_policy_metadata_plan` visit stronger edge paths first,
cap the graph candidate set with `max_nodes`, cap SIMD block spread with
`max_active_blocks`, and can drop weak paths below `min_path_weight` before
metadata filtering and vector scoring. `max_active_blocks` is the most direct
budget for TurboQuant work because masked search only visits active 32-slot
blocks.

When a caller wants a higher-level starting point, use `GraphSearchPreset`.
`low_latency`, `balanced`, and `broad` derive a `GraphSearchTuning` from the
index size and requested `k`, choosing `GraphViewPolicy` and
`GraphRerankConfig` together. `explain_graph_search_with_preset` runs the full
explainable search path from that preset.

`search_graph_view_with_policy_metadata_rerank` adds a second-stage ranker over
prefetched TurboQuant candidates. `GraphRerankConfig` blends the compressed
vector score with the graph path score and returns `GraphRerankedHit` values
containing `score`, `vector_score`, `graph_score`, depth, parent, and metadata.
Use this when a search product should prefer memories that are both semantically
close and close in the local context graph. The `_batch` variant uses the same
compiled view and path-score map for every query row, then sorts/truncates
reranked hits independently per row.

Use `search_graph_view_with_policy_metadata_rerank_timed` when tuning query
latency. It returns `GraphTimedRerankedSearchReport` with `GraphSearchTelemetry`
for view-build time, TurboQuant vector-search time, rerank time, trace-build
time, total time, and the masked-kernel block-skip delta observed during the
request.

Use `explain_graph_search_with_policy_metadata_rerank_timed` for product/debug
surfaces. It returns `GraphExplainedSearchReport`, combining reranked hits,
planning telemetry, per-query timing, and the `GraphViewTrace` that shows the
nodes and edges considered by the local context view.

Call `debug_snapshot()` on an explained report to get a UI-friendly
`GraphSearchDebugSnapshot`. The snapshot keeps primitive fields only, includes a
summary, telemetry, ranked hits, trace edges, and trace nodes annotated with
`hit_rank`, final score, vector score, and graph score when a node was returned
as a hit. This is intended as the stable handoff for JSON serialization or graph
debug panels without adding a serialization dependency to the crate.
For a runnable JSON export example, use:

```bash
cargo run -p turbo-graph --features serde --example graph_memory_debug_export
cargo run -p turbo-graph --features serde --example graph_memory_debug_export -- \
  --output /tmp/graph-memory-snapshot.json
```

To inspect that payload visually, open
`turbo-graph/examples/graph_memory_panel.html` in a browser and load or paste the
snapshot JSON.

Enable the optional `serde` feature when the caller wants direct JSON export:

```toml
turbo-graph = { version = "0.1", features = ["serde"] }
```

With that feature enabled, reports, hits, traces, telemetry, cache stats, cache
budgets, and debug snapshots derive `Serialize` and `Deserialize`, so an
application can call `serde_json::to_string(&snapshot)` in its own HTTP/logging
layer.

`search_graph_view_with_trace` returns hits plus a `GraphViewTrace` containing
view nodes, visible edges, depths, parent links, edge weights, path weights,
source, and timestamp metadata. Use this when a search UI needs to draw or
explain the local context view. `explain_graph_view_with_policy` exports the
same trace shape for a weighted, budgeted view.

For persistence, call `memory.write("memory.tv", "memory.tvgm")`; load with
`GraphMemoryIndex::load("memory.tv", "memory.tvgm")`.

---

## File formats

### `.tv` — `TurboQuantIndex`

```
┌──────────────────────────────────────┐
│ 9-byte header                         │
│   bit_width  (u8)                     │
│   dim        (u32 LE)                 │
│   n_vectors  (u32 LE)                 │
├──────────────────────────────────────┤
│ packed codes                          │
│   (dim / 8) * bit_width * n_vectors   │
├──────────────────────────────────────┤
│ norms  (n_vectors × f32 LE)           │
└──────────────────────────────────────┘
```

### `.tvim` — `IdMapIndex`

```
┌──────────────────────────────────────┐
│ magic   "TVIM"  (4 bytes)             │
│ version  u8   = 1                     │
├──────────────────────────────────────┤
│ core payload (same as .tv)            │
├──────────────────────────────────────┤
│ slot_to_id  (n_vectors × u64 LE)      │
└──────────────────────────────────────┘
```

On load, the reverse `id → slot` map is rebuilt in memory. Duplicate ids in the `slot_to_id` table are rejected as corrupt.

`dim = 0` in the header signals a lazy uncommitted index (the constructor asserts `dim ≥ 8` so this value is unambiguous). `dim = 0` is only valid alongside `n_vectors = 0`; on load it produces an index whose `dim` is `None` until the first `add` / `add_with_ids` call.

Both formats are stable across minor versions. Breaking changes bump the file-format version byte (`.tvim`) or the header length (`.tv`).
