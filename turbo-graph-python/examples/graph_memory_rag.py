"""Minimal graph-memory RAG workflow.

Run after installing the wheel:

    python turbo-graph-python/examples/graph_memory_rag.py

The example models the part turbovec leaves to application code: combine a
local graph neighborhood with tags, source ACLs, time windows, and upstream
candidate ids before vector search.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from turbo_graph import GraphMemoryIndex


DIM = 8


def unit(axis: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def build_memory() -> tuple[GraphMemoryIndex, np.ndarray]:
    vectors = np.ascontiguousarray(
        np.vstack([unit(0), unit(1), unit(2), unit(3), unit(4)])
    )
    records = [
        {
            "id": 100,
            "title": "Architecture root",
            "tags": ["architecture", "root"],
            "source": "docs",
            "timestamp_ms": 1_700_000_000_000,
        },
        {
            "id": 110,
            "title": "Retrieval cache design",
            "tags": ["architecture", "cache"],
            "source": "docs",
            "timestamp_ms": 1_700_000_100_000,
        },
        {
            "id": 120,
            "title": "Ops incident",
            "tags": ["ops"],
            "source": "tickets",
            "timestamp_ms": 1_700_000_200_000,
        },
        {
            "id": 130,
            "title": "Release checklist",
            "tags": ["release"],
            "source": "docs",
            "timestamp_ms": 1_700_000_300_000,
        },
        {
            "id": 140,
            "title": "Archived idea",
            "tags": ["archive"],
            "source": "archive",
            "timestamp_ms": 1_700_000_400_000,
        },
    ]

    memory = GraphMemoryIndex(dim=DIM, bit_width=4)
    memory.add_records(vectors, records)
    memory.link_bidirectional(100, 110, 1.0)
    memory.link_directed(110, 120, 0.7)
    memory.link_directed(100, 130, 0.4)
    return memory, vectors


def main() -> None:
    memory, vectors = build_memory()

    upstream_candidates = [110, 120, 999, 110]
    hits = memory.search(
        vectors[1],
        k=5,
        seeds=[100],
        max_hops=2,
        required_tags=["architecture"],
        allowed_sources=["docs"],
        start_ms=1_700_000_050_000,
        end_ms=1_700_000_200_000,
        candidate_ids=upstream_candidates,
    )

    report = memory.explain(
        vectors[1],
        k=5,
        seeds=[100],
        preset="balanced",
        required_tags=["architecture"],
        allowed_sources=["docs"],
        candidate_ids=upstream_candidates,
    )

    batch_hits = memory.search_batch(
        np.ascontiguousarray(np.vstack([vectors[1], vectors[2]])),
        k=5,
        seeds=[100],
        max_hops=2,
        required_tags=["architecture"],
        candidate_ids=upstream_candidates,
    )

    before_trim = memory.cache_stats()
    budget = memory.trim_caches_for_preset("low_latency")
    after_trim = memory.cache_stats()

    with TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "memory.tv"
        graph_path = Path(tmp) / "memory.tvgm"
        memory.write(str(index_path), str(graph_path))
        loaded = GraphMemoryIndex.load(str(index_path), str(graph_path))
        assert loaded.record(110)["title"] == "Retrieval cache design"

    assert [hit["id"] for hit in hits] == [110]
    assert [len(row) for row in batch_hits] == [1, 1]
    assert report["plan"]["candidate_input_ids"] == 4
    assert report["plan"]["candidate_missing_ids"] == 1
    assert report["plan"]["candidate_duplicate_ids"] == 1
    assert before_trim["total_entries"] >= after_trim["total_entries"]

    print(
        json.dumps(
            {
                "hits": hits,
                "batch_hit_ids": [[hit["id"] for hit in row] for row in batch_hits],
                "candidate_plan": {
                    key: report["plan"][key]
                    for key in (
                        "selected_slots",
                        "candidate_input_ids",
                        "candidate_slots",
                        "candidate_missing_ids",
                        "candidate_duplicate_ids",
                    )
                },
                "cache_budget_total_entries": budget["total_entries"],
                "cache_entries_after_trim": after_trim["total_entries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
