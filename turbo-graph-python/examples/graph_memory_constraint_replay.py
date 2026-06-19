"""Replay the RAG constraint failure that turbo-graph is built to avoid.

Run after installing the wheel:

    python turbo-graph-python/examples/graph_memory_constraint_replay.py

The same query is evaluated three ways:

1. global vector top-k, then application post-filtering
2. upstream BM25/entity candidates only
3. turbo-graph graph + metadata + candidate constrained search
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable

import numpy as np

from turbo_graph import GraphMemoryIndex


DIM = 8
SEED_ID = 1_000
START_MS = 1_710_000_000_000
END_MS = 1_710_050_000_000
REQUIRED_TAGS = ["tenant:acme", "acl:workspace-alpha"]
ALLOWED_SOURCES = ["docs", "tickets"]
BM25_CANDIDATES = [1_050, 1_060, 424_242, 1_020, 1_020, 1_040, 1_030, 1_070]


@dataclass(frozen=True)
class Row:
    id: int
    title: str
    tags: tuple[str, ...]
    source: str
    timestamp_ms: int
    axes: tuple[tuple[int, float], ...]


ROWS = [
    Row(
        1_000,
        "Acme workspace root: active launch plan",
        ("tenant:acme", "acl:workspace-alpha", "project:launch"),
        "docs",
        1_710_000_000_000,
        ((2, 1.0), (3, 0.3)),
    ),
    Row(
        1_010,
        "Acme launch architecture decision record",
        ("tenant:acme", "acl:workspace-alpha", "architecture"),
        "docs",
        1_710_010_000_000,
        ((2, 0.8), (4, 0.5)),
    ),
    Row(
        1_020,
        "Acme retrieval cache rollout checklist",
        ("tenant:acme", "acl:workspace-alpha", "cache"),
        "docs",
        1_710_020_000_000,
        ((0, 0.82), (1, 0.25)),
    ),
    Row(
        1_030,
        "Acme support ticket: stale cache after deploy",
        ("tenant:acme", "acl:workspace-alpha", "incident"),
        "tickets",
        1_710_030_000_000,
        ((0, 0.78), (1, 0.18)),
    ),
    Row(
        1_040,
        "Acme archived launch notes from last quarter",
        ("tenant:acme", "acl:workspace-alpha", "archive"),
        "archive",
        1_690_000_000_000,
        ((0, 0.97), (5, 0.08)),
    ),
    Row(
        1_050,
        "Beta customer cache design with similar wording",
        ("tenant:beta", "acl:workspace-beta", "cache"),
        "docs",
        1_710_025_000_000,
        ((0, 1.00), (1, 0.03)),
    ),
    Row(
        1_060,
        "Acme private finance memo outside source ACL",
        ("tenant:acme", "acl:finance", "cache"),
        "finance",
        1_710_024_000_000,
        ((0, 0.99), (6, 0.04)),
    ),
    Row(
        1_070,
        "Acme release FAQ for customer-facing rollout",
        ("tenant:acme", "acl:workspace-alpha", "release"),
        "docs",
        1_710_040_000_000,
        ((0, 0.70), (7, 0.28)),
    ),
]


def embedding(axes: Iterable[tuple[int, float]]) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for axis, weight in axes:
        vector[axis] += weight
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-6)


def build_memory() -> tuple[GraphMemoryIndex, np.ndarray]:
    vectors = np.ascontiguousarray(np.vstack([embedding(row.axes) for row in ROWS]))
    records = [
        {
            "id": row.id,
            "title": row.title,
            "tags": list(row.tags),
            "source": row.source,
            "timestamp_ms": row.timestamp_ms,
        }
        for row in ROWS
    ]
    memory = GraphMemoryIndex(dim=DIM, bit_width=4)
    memory.add_records(vectors, records)
    memory.link_bidirectional(1_000, 1_010, 0.95)
    memory.link_bidirectional(1_010, 1_020, 0.90)
    memory.link_bidirectional(1_020, 1_030, 0.72)
    memory.link_bidirectional(1_000, 1_040, 0.35)
    memory.link_directed(1_020, 1_070, 0.60)
    memory.link_bidirectional(1_050, 1_060, 0.80)
    return memory, vectors


def row_by_id(id: int) -> Row | None:
    return next((row for row in ROWS if row.id == id), None)


def passes_rag_policy(id: int) -> bool:
    row = row_by_id(id)
    if row is None:
        return False
    return (
        all(tag in row.tags for tag in REQUIRED_TAGS)
        and row.source in ALLOWED_SOURCES
        and START_MS <= row.timestamp_ms < END_MS
    )


def global_overfetch(query: np.ndarray, vectors: np.ndarray, fetch_k: int) -> list[int]:
    scores = vectors @ query
    order = np.argsort(-scores)[:fetch_k]
    return [ROWS[int(i)].id for i in order]


def candidate_only(query: np.ndarray, vectors: np.ndarray, k: int) -> list[int]:
    live_unique = []
    seen = set()
    for id in BM25_CANDIDATES:
        if id in seen or row_by_id(id) is None:
            continue
        seen.add(id)
        live_unique.append(id)
    id_to_pos = {row.id: i for i, row in enumerate(ROWS)}
    ranked = sorted(
        live_unique,
        key=lambda id: float(vectors[id_to_pos[id]] @ query),
        reverse=True,
    )
    return ranked[:k]


def fmt_ids(ids: list[int]) -> str:
    return "[" + ", ".join(str(id) for id in ids) + "]"


def main() -> None:
    memory, vectors = build_memory()
    query = embedding(((0, 1.0), (1, 0.08)))

    started = perf_counter_ns()
    global_top3 = global_overfetch(query, vectors, fetch_k=3)
    global_filtered = [id for id in global_top3 if passes_rag_policy(id)]
    global_ns = perf_counter_ns() - started

    started = perf_counter_ns()
    candidates_top3 = candidate_only(query, vectors, k=3)
    candidates_filtered = [id for id in candidates_top3 if passes_rag_policy(id)]
    candidates_ns = perf_counter_ns() - started

    started = perf_counter_ns()
    constrained_hits = memory.search(
        query,
        3,
        [SEED_ID],
        max_hops=3,
        required_tags=REQUIRED_TAGS,
        allowed_sources=ALLOWED_SOURCES,
        start_ms=START_MS,
        end_ms=END_MS,
        candidate_ids=BM25_CANDIDATES,
    )
    constrained_ns = perf_counter_ns() - started

    report = memory.explain(
        query,
        3,
        [SEED_ID],
        preset="broad",
        required_tags=REQUIRED_TAGS,
        allowed_sources=ALLOWED_SOURCES,
        start_ms=START_MS,
        end_ms=END_MS,
        candidate_ids=BM25_CANDIDATES,
    )
    plan = report["plan"]

    print("RAG constraint replay: tenant ∩ graph ∩ source ∩ time ∩ BM25 candidates")
    print("")
    print("path                         fetched        after_policy    ns")
    print(f"global top-3 then filter     {fmt_ids(global_top3):<14} {fmt_ids(global_filtered):<14} {global_ns}")
    print(
        f"candidate ids only           {fmt_ids(candidates_top3):<14} "
        f"{fmt_ids(candidates_filtered):<14} {candidates_ns}"
    )
    print(
        f"turbo-graph constrained      {fmt_ids([hit['id'] for hit in constrained_hits]):<14} "
        f"{fmt_ids([hit['id'] for hit in constrained_hits]):<14} {constrained_ns}"
    )
    print("")
    print(
        "planner: "
        f"selected_slots={plan['selected_slots']} "
        f"active_blocks={plan['active_blocks']} "
        f"candidate_missing_ids={plan['candidate_missing_ids']} "
        f"candidate_duplicate_ids={plan['candidate_duplicate_ids']}"
    )
    print("takeaway: global top-k found similar but unauthorized/stale chunks; the constrained path still returns the in-view answer.")


if __name__ == "__main__":
    main()
