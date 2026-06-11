# turbo-graph

Python bindings for `turbo-graph`: a turbovec-compatible TurboQuant vector
core plus graph memory for filter-heavy RAG.

Use `turbo-graph` when retrieval constraints are part of the product:
semantic similarity plus graph neighborhoods, tags, sources, time windows,
upstream candidate ids, cache reuse, and explain telemetry. If you only need
flat global top-k or a light id allowlist, the shared TurboQuant core remains
compatible with the turbovec lineage.

Long-running Rust operations release the Python GIL, including add, search,
batch search, prepare, and write paths. That lets threaded Python services
overlap independent vector and graph-memory requests.

## Install

```bash
pip install turbo-graph
```

Optional framework integrations:

```bash
pip install "turbo-graph[langchain]"
pip install "turbo-graph[llama-index]"
pip install "turbo-graph[haystack]"
pip install "turbo-graph[agno]"
```

## Vector index

```python
import numpy as np
from turbo_graph import TurboQuantIndex

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors.astype(np.float32))
scores, slots = index.search(query.astype(np.float32)[None, :], k=10)
```

For stable external ids, use `IdMapIndex`.

## Graph memory

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

hits = memory.search(
    query.astype(np.float32),
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

`GraphMemoryIndex` also provides:

- `search_batch(...)` to reuse one graph/metadata/candidate view across many
  query embeddings.
- `cache_stats()`, `cache_budget_for_preset(...)`,
  `trim_caches_for_preset(...)`, and `clear_all_caches()` for long-running
  local memory services.
- `record(id)`, `neighbors(id)`, `write(index_path, graph_path)`, and
  `GraphMemoryIndex.load(...)` for inspection and persistence.

## Runnable example

From the repository root:

```bash
python turbo-graph-python/examples/graph_memory_rag.py
```

The example covers records, graph links, tags, source/time filters, upstream
candidate ids, batch search, explain telemetry, cache trimming, and
persistence.

## Project

- Repository: https://github.com/bigmacfive/turbo-graph
- API docs: https://github.com/bigmacfive/turbo-graph/blob/main/docs/api.md
- Benchmark notes: https://github.com/bigmacfive/turbo-graph/blob/main/docs/benchmark_turbo_graph_vs_turbo_vec.md
- Contributing: https://github.com/bigmacfive/turbo-graph/blob/main/CONTRIBUTING.md
- Changelog: https://github.com/bigmacfive/turbo-graph/blob/main/CHANGELOG.md
- Security policy: https://github.com/bigmacfive/turbo-graph/blob/main/SECURITY.md
