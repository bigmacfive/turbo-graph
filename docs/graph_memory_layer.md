# Graph-View Local Memory Layer

<p align="left">
  <a href="README.md">← Docs index</a> ·
  <a href="benchmark_turbo_graph_vs_turbo_vec.md">vs turbovec</a> ·
  <a href="api.md">API reference</a>
</p>

A local context-memory layer for workloads where retrieval is scoped by graph
neighborhood, metadata, and external candidate stages — not just global
semantic top-k.

## Core Baseline

The shared TurboQuant core (from [turbovec](https://github.com/RyanCodrai/turbovec)) already handles the expensive part well:

- `TurboQuantIndex` stores compressed positional vectors and searches with
  architecture-specific SIMD kernels.
- `IdMapIndex` adds stable external ids over positional slots.
- Filtered search is pushed into the kernel, so disallowed slots never enter
  the heap and empty 32-vector blocks can be skipped.

The avoidable cost for graph/memory workloads was the filter representation.
The public Rust path accepted a full `&[bool]` mask, and `IdMapIndex` built that
mask from ids before packing it into `u64` words for the kernel. A graph view
will reuse the same candidate set across many queries, so rebuilding that
full-length mask each time is unnecessary work.

## Reusable Slot Masks

`SlotMask` is a packed reusable slot allowlist:

- one bit per positional vector slot
- cached allowed-count for `effective_k`
- cached active 32-slot SIMD block list for sparse graph views
- union/intersection operations for graph + tag + time + ACL composition
- direct iteration over selected slots for debug/export/planning paths
- direct handoff to `TurboQuantIndex::search_with_slot_mask`

This lets a higher layer compile a graph view once and run many vector queries
against the same view without allocating or scanning a full boolean mask each
time. When the view is sparse, the search loop visits only the active 32-slot
blocks instead of iterating over every block in the index and checking whether
the block is empty.

## Graph Memory Layer

The production implementation lives in `turbo-graph/src/graph_memory.rs` as
`GraphMemoryIndex`. See `turbo-graph/examples/graph_memory_layer.rs` for a small
end-to-end Rust usage example and
`turbo-graph-python/examples/graph_memory_rag.py` for the Python operating API.

The layer keeps four sidecars:

- `id_to_slot`: stable memory id to TurboQuant slot
- `slot_to_id`: TurboQuant slot back to stable memory id
- `nodes`: titles, tags, source, and timestamp metadata
- `edges`: graph links between memory nodes

Query flow:

1. Start from seed memory ids such as the active project, page, user, or topic.
2. Walk the graph up to `max_hops`.
3. Compile the visited slots into a `SlotMask`.
4. Optionally intersect with tag, time, ACL, or source masks.
5. Run `TurboQuantIndex::search_with_slot_mask`.
6. Translate result slots back to stable memory ids.

That turns graph context into an early candidate-pruning step rather than a
post-filter. For selective views, the cached block list skips whole 32-vector
blocks before SIMD scoring begins, and lane-level mask checks still prevent
disallowed slots inside a non-empty block from entering top-k.

## Why this beats a bare allowlist path

Global vector search with a one-shot `allowlist=` is already fast. A search engine or
local memory layer usually adds more context:

- active workspace or site
- user/session access rules
- graph neighborhood around the current document
- time windows
- tags, collections, entities, and source types
- query planner stages from BM25 or keyword/entity retrieval

If those constraints cut the candidate set before scoring, TurboQuant does less
SIMD work while preserving top-k correctness within the intended view. The
advantage is largest when views are selective and reused.

## Benchmark Gates

To keep proving the efficiency claim, measure:

- `IdMapIndex::search_with_allowlist` versus cached `SlotMask` search for
  repeated graph-view queries.
- End-to-end latency at selectivity levels: 0.1%, 1%, 5%, 20%, global.
- Preset-driven graph+metadata+rerank latency for `low_latency`, `balanced`,
  and `broad` local-context settings.
- Mask build time, allocation count, blocks skipped, and search time.
- Recall within graph view versus global search + post-filter.
- Delete/update cost once graph memory supports compaction and slot moves.

A first quick harness is available at `turbo-graph/examples/graph_view_bench.rs`:

```bash
cargo run -p turbo-graph --release --example graph_view_bench
cargo run -p turbo-graph --release --example graph_view_bench -- \
  --vectors-f32 embeddings.f32 --dim 768 --query-row 0 --iters 50 \
  --csv graph-view-bench.csv
cargo run -p turbo-graph --example graph_view_bench_summary -- graph-view-bench.csv
```

It compares global search, boolean-mask filtered search, cached `SlotMask`
filtered search, `GraphMemoryIndex` graph-view search, and preset-driven
graph+metadata+rerank search on synthetic indexes. The benchmark runs a
single-query search-engine style workload across 0.1%, 1%, 5%, 20%, and 100%
graph-view selectivity, then reports `low_latency`, `balanced`, and `broad`
preset behavior over a fanout graph with tag/source/time metadata filters.
It also compares direct graph-view masked search against global overfetch plus
post-filtering, using the direct masked result as the view-local top-k
reference.
The optional corpus mode reads raw little-endian row-major `f32` embeddings from
`--vectors-f32`; `--queries-f32` can provide a separate query matrix, otherwise
`--query-row` selects one vector row as the benchmark query. This keeps the
harness dependency-free while allowing preset, cache-budget, and post-filter
recall checks against real or exported embedding corpora instead of only toy
vectors.
If `--queries-f32` is omitted, a single query is still expanded to
`BATCH_QUERY_ROWS` internally so benchmark CSV always includes a batch rerank
timing row for prepared-view workloads.
Pass `--csv path.csv` to save a long-form result file with global, selectivity,
preset, and post-filter rows. That makes corpus runs comparable across preset
tuning changes instead of relying only on terminal tables.
Use `graph_view_bench_summary` to read one or more CSV files back into a compact
tuning summary: selectivity ratios, preset graph overhead, and the first
global-overfetch size that recovers full view-local recall.

## Production Usage Patterns

- When seed/policy/metadata filters are stable and only the query embedding
  changes, compile the view once with `prepare_graph_view` or
  `prepare_graph_view_with_policy_metadata`, then run repeated queries through
  `search`, `search_batch`, or `search_rerank_batch`.
- For batch queries, prefer `search_with_slot_mask_batch` or
  `GraphPreparedPolicyView::search_rerank_batch` so mask construction and graph
  path-score calculation are reused.
- When exposing cache health, report `query_cache_hit_ratio`,
  `metadata_cache_hit_ratio`, and the total `cache_hit_ratio` from
  `cache_stats()`.

On this checkout after active-block caching, combined graph+metadata view
caching, and batched mask composition, a release run over 16,384 synthetic 64-d
vectors with a single query showed cached `SlotMask` search at about 13% of
global search time for 0.1-1% selectivity, about 15% at 5%, and about 27% at
20%. The full `GraphMemoryIndex` cached-view path landed at about 17-21% for
0.1-1%, about 19% at 5%, about 33% at 20%, and about parity for a 100% view. A full-view
`SlotMask` falls back to the unmasked search path, so the 100% case avoids
masked-kernel overhead.

The preset workload in the same run selected 11 slots for `low_latency`, 24 for
`balanced`, and 56 for `broad` after graph and metadata filtering. End-to-end
cached graph+metadata+rerank latency landed at about 18%, 28%, and 42% of
global TurboQuant search respectively, while the exact same compiled masks
searched directly through `TurboQuantIndex::search_with_slot_mask` stayed around
13% of global. That gap is the current planning/rerank overhead budget to keep
shrinking.

The post-filter section used the `balanced` view as the reference. Direct
masked search returned the view-local top 10 in about 0.007 ms. Global search
followed by filtering returned zero of those 10 with fetch sizes 10, 40, and
160; fetch 640 recovered only 20% recall while already taking about 108x the
direct masked latency. Larger overfetch eventually recovers the view-local
answers, but it does so by spending far more work outside the graph view. This
is the correctness reason to pre-filter inside the TurboQuant kernel instead of
searching globally and dropping disallowed results afterwards.

## Implemented Capabilities

- `GraphMemoryIndex` wraps `TurboQuantIndex` directly instead of `IdMapIndex`,
  so it owns stable-id side tables and can repair slots after `swap_remove`.
- Graph views are cached as `SlotMask` values and can be intersected with tag
  masks before search.
- `SlotMask` caches active 32-slot SIMD blocks, allowing sparse graph/tag views
  to visit only candidate blocks while preserving lane-level allowlist checks.
- `SlotMask::allowed_slots()` walks packed bits directly when callers need the
  selected slot ids again, avoiding a full-domain scan for sparse views.
- `SlotMask::union_with_many` and `SlotMask::intersect_with_many` compose
  repeated tag/source/time predicates in packed words and rebuild counts once,
  reducing mask-planning overhead for realistic multi-predicate RAG filters.
- `SlotMask::all` and all-slot views use the unmasked TurboQuant search path,
  avoiding overhead when no graph pruning is possible.
- `add_records` performs batch ingest so large local memory indexes avoid
  repeated one-vector TurboQuant encode calls.
- Tags are indexed into a tag-to-id map and repeated tag filters are cached as
  `SlotMask` values, so metadata filters avoid scanning all records per query.
- Sources are indexed into source-to-id maps and timestamp windows use a sorted
  `(timestamp_ms, id)` index, so site/domain/collection and recency filters can
  be composed with graph views without scanning all records.
- `search_graph_view_with_metadata` combines graph seeds, required tags,
  allowed sources, and half-open timestamp windows before vector scoring.
- `search_graph_view_with_metadata_plan` reuses a combined graph+metadata
  `SlotMask` cache for repeated context windows and reports graph slots,
  selected slots, active SIMD blocks, graph-cache hits, combined-cache hits, and
  selectivity.
- `search_graph_view_with_metadata_batch_plan` compiles the same graph+metadata
  view once and sends multiple query embeddings through one TurboQuant batch
  search, returning row-wise hits with shared plan telemetry.
- `candidate_id_mask` and
  `search_graph_view_with_metadata_candidates_plan` intersect graph+metadata
  views with query-specific candidate ids from BM25, keyword/entity retrieval,
  ACL, or dedup stages. Candidate ids reuse the cached graph+metadata base view
  and are intentionally not part of the persistent combined-view cache key.
  Candidate plans report raw input ids, unique live candidate slots, duplicate
  ids, and missing/stale ids for upstream retriever tuning.
- `search_graph_view_with_policy_metadata_candidates_rerank` and its timed
  variant run graph-aware reranking over that query-specific candidate
  intersection, so hybrid search can blend vector score with local graph-path
  strength after BM25/keyword/ACL pruning.
- `search_graph_view_with_policy_metadata_candidates_rerank_batch` reuses that
  candidate intersection and graph path-score map across multiple query
  embeddings, then reranks each row independently.
- `explain_graph_search_with_policy_metadata_candidates_rerank_timed` and
  `GraphCandidateSearchDebugSnapshot` package hybrid candidate search results,
  candidate/metadata/graph selectivity, telemetry, hits, and trace nodes/edges
  for graph panels and query-debug logs.
- `search_graph_view_with_policy_metadata_candidate_scores_hybrid` accepts
  `(memory_id, score)` pairs from external planners. `GraphHybridRerankConfig`
  blends vector, graph, and candidate scores, while `GraphHybridHit` exposes all
  components for ranking audits. `GraphCandidateScoreNormalization` can keep
  raw calibrated scores, min/max-scale BM25-style scores, or max-abs-scale
  signed priors before the hybrid blend. Score normalization is scoped to the
  final graph+metadata+candidate view, so stale candidates outside the local
  context cannot distort the scale used for returned hits. The hybrid path
  builds the candidate mask, score map, duplicate count, and stale-id count in
  one pass over the external candidate-score list.
- `search_graph_view_with_policy_metadata_candidate_scores_hybrid_batch` reuses
  that candidate-score mask/map plus one TurboQuant batch search across multiple
  query embeddings, then applies hybrid rerank independently per row.
- `explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed` and
  `GraphCandidateHybridSearchDebugSnapshot` preserve those three scoring
  components on hits and trace nodes for hybrid ranking diagnostics.
- `GraphViewPolicy` adds weighted, budgeted expansion for large memory graphs:
  visit stronger path weights first, cap candidates with `max_nodes`, and drop
  paths below `min_path_weight` before metadata filters and vector scoring.
  `max_active_blocks` can also cap spread across TurboQuant's 32-slot SIMD
  blocks, which is closer to actual masked-search cost than raw node count.
- `GraphSearchPreset` derives a policy and rerank config from index size and
  requested `k`, giving callers `low_latency`, `balanced`, and `broad` starting
  points before hand-tuning the lower-level knobs.
- `search_graph_view_with_policy_metadata_plan` gives the same combined-cache
  and planning telemetry for budgeted views.
- `search_graph_view_with_policy_metadata_rerank` blends TurboQuant vector
  scores with graph path scores over a bounded prefetch set, returning both
  component scores for ranking/debug UI.
- `search_graph_view_with_policy_metadata_rerank_batch` reuses one policy
  expansion, one compiled graph+metadata mask, and one path-score map across
  multiple query embeddings, then reranks each row independently.
- `search_graph_view_with_policy_metadata_rerank_timed` adds per-query
  telemetry for view-build time, vector-search time, rerank time, trace-build
  time, total time, and skipped masked-kernel blocks.
- `explain_graph_search_with_policy_metadata_rerank_timed` packages reranked
  hits, plan telemetry, query timing, and a `GraphViewTrace` into one report for
  graph-view debugging and UI export.
- `GraphExplainedSearchReport::debug_snapshot()` merges ranked hits into trace
  nodes and produces a primitive-field `GraphSearchDebugSnapshot` for JSON
  serialization, graph panels, and query-debug logs without adding a dependency.
- An optional `serde` feature derives `Serialize`/`Deserialize` for reports,
  hits, traces, telemetry, cache stats, and debug snapshots, keeping default
  builds lean while letting server/UI crates export snapshots as JSON.
- `GraphMemoryIndex::cache_stats()` exposes query-cache and metadata-cache
  entry counts. `clear_*_caches` and `trim_*_caches` give long-running local
  memory processes a simple way to bound cache growth under diverse graph,
  source, tag, and time-window filters.
- `GraphMemoryCacheBudget` and `GraphSearchPreset::cache_budget` derive
  preset-sized cache caps from hop and active-block targets. `trim_caches_to_budget`
  and `trim_caches_for_preset` let a service periodically enforce those caps
  without touching stored vectors, graph edges, or metadata.
- `graph_view_bench` now reports both raw selectivity sweeps and preset-driven
  graph+metadata+rerank workloads, including selected slots, active blocks,
  prefetch size, block skips, and cache entries.
- The same CSV includes `mask_build`, `view_compile`, and `constrained` phases so
  release artifacts can separate one-shot graph+metadata compilation from warm
  cached search and candidate intersection.
- `graph_view_bench` also reports global overfetch plus post-filter recall
  against direct graph-view masked search, showing when post-filtering misses
  the view-local top-k.
- `graph_view_bench --vectors-f32 ... --dim ...` adds a dependency-free corpus
  mode for real/exported embeddings, with optional query files and iteration
  counts for longer local workload traces.
- `graph_view_bench --csv ...` writes long-form benchmark rows for global,
  selectivity, preset, and post-filter phases so cache-budget and preset tuning
  can be compared across corpus runs.
- `graph_view_bench_summary` reads one or more benchmark CSV files and prints
  the key tuning indicators: sparse-mask speedups, graph planning overhead,
  preset ratios, and full-recall overfetch cost.
- `graph_memory_debug_export` emits a serde-backed JSON debug snapshot for a
  hybrid graph+metadata+candidate-score search, giving UI and search-flow
  logging code a concrete payload shape to consume, including candidate input,
  duplicate, and stale-id counts. It can write directly to a file with
  `--output`, which makes repeatable graph-panel fixtures easy to produce from
  local traces.
- `graph_memory_panel.html` is a dependency-free browser panel for loading that
  JSON snapshot, inspecting summary/telemetry/hit score components, and seeing
  the graph trace with hit ranks merged into nodes.
- `search_graph_view_with_stats` exposes selected slot count, total slot count,
  selectivity, and graph-view cache hit/miss.
- `search_graph_view_with_trace` returns a materialized graph-view trace with
  nodes, visible edges, hop depth, parent id, edge weight, path weight, source,
  and timestamp metadata so product UIs can show why a memory was inside the
  retrieval view.
- `replace_embedding` appends the new vector first, then uses `swap_remove` to
  move it into the old slot, preserving graph edges and stable ids.
- `replace_record_metadata` updates titles, tags, source, and timestamps
  without touching vectors or graph edges. It preserves cached raw graph views
  and invalidates only metadata masks plus combined graph+metadata views.
- `write(index.tv, graph.tvgm)` persists the compressed TurboQuant index and a
  graph-memory sidecar; `load(index.tv, graph.tvgm)` validates slot count,
  duplicate ids, records, edges, dim, and bit width.

## Verification Checklist

For routine release readiness, run:

```bash
scripts/release_check.sh --quick
```

For integration or packaging-heavy changes, run:

```bash
scripts/release_check.sh --full
```

For graph-memory planning, candidate handling, cache policy, or explain
telemetry changes, also run:

```bash
cargo run -p turbo-graph --release --example graph_view_bench -- --iters 3 --csv /tmp/graph-view-bench.csv
cargo run -p turbo-graph --release --example graph_view_bench_summary -- /tmp/graph-view-bench.csv
python turbo-graph-python/examples/graph_memory_rag.py
```

If you have a real embedding corpus, run corpus mode:

```bash
cargo run -p turbo-graph --release --example graph_view_bench -- \
  --vectors-f32 embeddings.f32 --dim 768 --query-row 0 --iters 50 \
  --csv /tmp/graph-view-bench.csv
```

Cache-budget changes should include a pre/post check of
`query_cache_hit_ratio`, `metadata_cache_hit_ratio`, `cache_hit_ratio`, and
`cache_miss_ratio` under comparable traffic.
