import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

import turbo_graph
from turbo_graph import GraphMemoryIndex


DIM = 32


def _unit(row: int) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    vec[row % DIM] = 1.0
    return vec


def _fixture() -> tuple[GraphMemoryIndex, np.ndarray]:
    vectors = np.vstack([_unit(i) for i in range(5)]).astype(np.float32)
    records = [
        {
            "id": 10,
            "title": "architecture root",
            "tags": ["architecture", "root"],
            "source": "docs",
            "timestamp_ms": 1000,
        },
        {
            "id": 20,
            "title": "retrieval cache",
            "tags": ["architecture", "cache"],
            "source": "docs",
            "timestamp_ms": 2000,
        },
        {
            "id": 30,
            "title": "ops metric",
            "tags": ["ops"],
            "source": "tickets",
            "timestamp_ms": 3000,
        },
        {
            "id": 40,
            "title": "release note",
            "tags": ["release"],
            "source": "docs",
            "timestamp_ms": 4000,
        },
        {
            "id": 50,
            "title": "distant note",
            "tags": ["archive"],
            "source": "archive",
            "timestamp_ms": 5000,
        },
    ]
    memory = GraphMemoryIndex(DIM, 4)
    memory.add_records(vectors, records)
    memory.link_bidirectional(10, 20, 1.0)
    memory.link_directed(20, 30, 0.8)
    memory.link_directed(10, 40, 0.5)
    return memory, vectors


def test_graph_memory_search_filters_by_graph_tag_source_and_time():
    memory, vectors = _fixture()

    hits = memory.search(
        vectors[1],
        5,
        [10],
        max_hops=2,
        required_tags=["architecture"],
        allowed_sources=["docs"],
        start_ms=1500,
        end_ms=2500,
    )

    assert [hit["id"] for hit in hits] == [20]
    assert hits[0]["title"] == "retrieval cache"
    assert hits[0]["tags"] == ["architecture", "cache"]
    assert hits[0]["source"] == "docs"
    assert hits[0]["timestamp_ms"] == 2000


def test_graph_memory_search_accepts_candidate_ids_from_upstream_retrievers():
    memory, vectors = _fixture()

    hits = memory.search(
        vectors[2],
        5,
        [10],
        max_hops=2,
        required_tags=["architecture"],
        candidate_ids=[20, 30, 999, 20],
    )

    assert [hit["id"] for hit in hits] == [20]
    assert memory.search(vectors[1], 5, [10], candidate_ids=[]) == []


def test_graph_memory_search_batch_reuses_graph_metadata_and_candidate_view():
    memory, vectors = _fixture()
    queries = np.ascontiguousarray(np.vstack([vectors[1], vectors[2]]))

    batch = memory.search_batch(
        queries,
        5,
        [10],
        max_hops=2,
        required_tags=["architecture"],
        candidate_ids=[20, 30, 999, 20],
    )

    single = [
        memory.search(
            query,
            5,
            [10],
            max_hops=2,
            required_tags=["architecture"],
            candidate_ids=[20, 30, 999, 20],
        )
        for query in queries
    ]
    assert [[hit["id"] for hit in row] for row in batch] == [
        [hit["id"] for hit in row] for row in single
    ]
    assert [len(row) for row in memory.search_batch(queries, 5, [10], candidate_ids=[])] == [0, 0]


def test_graph_memory_explain_returns_plan_telemetry_and_trace():
    memory, vectors = _fixture()

    report = memory.explain(
        vectors[1],
        3,
        [10],
        preset="balanced",
        required_tags=["architecture"],
        allowed_sources=["docs"],
    )

    assert set(report) == {"hits", "plan", "prefetch_k", "telemetry", "trace"}
    assert report["hits"]
    assert {"total_slots", "selected_slots", "combined_cache_hit"} <= set(report["plan"])
    assert {"total_ns", "blocks_skipped_by_mask"} <= set(report["telemetry"])
    assert {"nodes", "edges", "stats", "seeds", "max_hops"} <= set(report["trace"])
    assert any(node["id"] == 20 for node in report["trace"]["nodes"])


def test_graph_memory_explain_reports_candidate_plan_quality():
    memory, vectors = _fixture()

    report = memory.explain(
        vectors[1],
        5,
        [10],
        preset="balanced",
        required_tags=["architecture"],
        candidate_ids=[20, 30, 999, 20],
    )

    assert report["hits"]
    assert report["plan"]["candidate_input_ids"] == 4
    assert report["plan"]["candidate_slots"] == 2
    assert report["plan"]["candidate_missing_ids"] == 1
    assert report["plan"]["candidate_duplicate_ids"] == 1
    assert report["plan"]["selected_slots"] == 1
    assert report["plan"]["metadata_slots"] == 2
    assert report["plan"]["candidate_live_ratio"] == pytest.approx(0.5)

    empty = memory.explain(vectors[1], 5, [10], candidate_ids=[])
    assert empty["hits"] == []
    assert empty["prefetch_k"] == 0
    assert empty["plan"]["candidate_input_ids"] == 0
    assert empty["plan"]["selected_slots"] == 0


def test_graph_memory_cache_stats_change_after_repeated_query():
    memory, vectors = _fixture()
    before = memory.cache_stats()

    for _ in range(2):
        memory.search(
            vectors[1],
            3,
            [10],
            required_tags=["architecture"],
            allowed_sources=["docs"],
        )

    after = memory.cache_stats()
    assert after["total_entries"] >= before["total_entries"]
    assert after["cache_accesses"] > before["cache_accesses"]
    assert after["query_cache_hits"] >= before["query_cache_hits"]


def test_graph_memory_cache_controls_for_long_running_services():
    memory, vectors = _fixture()

    for seed in ([10], [20], [30]):
        memory.search(
            vectors[1],
            3,
            seed,
            required_tags=["architecture"],
            allowed_sources=["docs"],
        )
    assert memory.cache_stats()["total_entries"] > 0

    budget = memory.cache_budget_for_preset("low_latency")
    assert budget["total_entries"] == budget["query_entries"] + budget["metadata_entries"]
    assert budget["graph_views"] > 0

    memory.trim_all_caches(1)
    trimmed = memory.cache_stats()
    assert trimmed["graph_views"] <= 1
    assert trimmed["combined_views"] <= 1
    assert trimmed["tag_masks"] <= 1
    assert trimmed["source_masks"] <= 1

    applied = memory.trim_caches_for_preset("balanced")
    assert applied["total_entries"] >= applied["query_entries"]

    memory.clear_query_caches()
    query_cleared = memory.cache_stats()
    assert query_cleared["graph_views"] == 0
    assert query_cleared["combined_views"] == 0
    assert query_cleared["tag_masks"] >= 0

    memory.clear_metadata_caches()
    assert memory.cache_stats()["metadata_entries"] == 0

    memory.search(vectors[1], 3, [10], required_tags=["architecture"])
    assert memory.cache_stats()["total_entries"] > 0
    memory.clear_all_caches()
    assert memory.cache_stats()["total_entries"] == 0

    with pytest.raises(ValueError, match="unknown graph search preset"):
        memory.cache_budget_for_preset("fast")


def test_graph_memory_write_load_roundtrip(tmp_path):
    memory, vectors = _fixture()
    index_path = tmp_path / "memory.tv"
    graph_path = tmp_path / "memory.tvgm"

    memory.write(str(index_path), str(graph_path))
    loaded = GraphMemoryIndex.load(str(index_path), str(graph_path))

    assert len(loaded) == len(memory)
    assert loaded.contains(20)
    assert loaded.slot_of(20) is not None
    hits = loaded.search(vectors[1], 2, [10], required_tags=["architecture"])
    assert any(hit["id"] == 20 for hit in hits)


def test_graph_memory_record_and_neighbors_helpers():
    memory, _vectors = _fixture()

    record = memory.record(20)
    assert record == {
        "id": 20,
        "title": "retrieval cache",
        "tags": ["architecture", "cache"],
        "source": "docs",
        "timestamp_ms": 2000,
    }
    assert memory.record(999) is None

    neighbors = memory.neighbors(10)
    assert {"to": 20, "weight": pytest.approx(1.0)} in neighbors
    assert {"to": 40, "weight": pytest.approx(0.5)} in neighbors
    with pytest.raises(KeyError, match="999"):
        memory.neighbors(999)


def test_package_exposes_version():
    assert turbo_graph.__version__ == "0.1.0"
    assert "GraphMemoryIndex" in turbo_graph.__all__


def test_graph_memory_rag_example_runs():
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "graph_memory_rag.py"
    )
    completed = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"candidate_missing_ids": 1' in completed.stdout
    assert '"batch_hit_ids"' in completed.stdout


def test_graph_memory_constraint_replay_example_runs():
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "graph_memory_constraint_replay.py"
    )
    completed = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "global top-3 then filter" in completed.stdout
    assert "turbo-graph constrained" in completed.stdout
    assert "candidate_missing_ids=1" in completed.stdout
    assert "candidate_duplicate_ids=1" in completed.stdout


def test_graph_memory_record_validation_and_duplicate_ids():
    memory = GraphMemoryIndex(DIM, 4)
    vectors = np.vstack([_unit(0), _unit(1)]).astype(np.float32)

    with pytest.raises(ValueError, match="missing required 'title'"):
        memory.add_records(vectors[:1], [{"id": 1, "tags": ["x"]}])

    memory.add_records(
        vectors[:1],
        [{"id": 1, "title": "one", "tags": ["x"]}],
    )
    with pytest.raises(ValueError, match="already present"):
        memory.add_records(
            vectors[1:],
            [{"id": 1, "title": "duplicate", "tags": ["x"]}],
        )


def test_graph_memory_missing_link_endpoint_and_remove():
    memory, _vectors = _fixture()

    with pytest.raises(ValueError, match="not present"):
        memory.link_directed(10, 999, 1.0)

    assert memory.remove_node(30) is True
    assert memory.remove_node(30) is False
    assert not memory.contains(30)


def test_graph_memory_rejects_non_contiguous_and_wrong_dim_inputs():
    memory = GraphMemoryIndex(DIM, 4)
    non_contiguous = np.zeros((DIM, 2), dtype=np.float32).T
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="vectors must be C-contiguous"):
        memory.add_records(
            non_contiguous,
            [
                {"id": 1, "title": "one", "tags": []},
                {"id": 2, "title": "two", "tags": []},
            ],
        )

    with pytest.raises(ValueError, match="vector dim"):
        memory.add_node(1, "bad", np.zeros(DIM + 1, dtype=np.float32), [])

    with pytest.raises(ValueError, match="query dim"):
        memory.search(np.zeros(DIM + 1, dtype=np.float32), 1, [1])

    with pytest.raises(ValueError, match="query dim"):
        memory.search_batch(np.zeros((1, DIM + 1), dtype=np.float32), 1, [1])
