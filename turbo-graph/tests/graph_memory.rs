//! Tests for the graph-view local memory layer.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use turbo_graph::{
    GraphCandidateScoreNormalization, GraphHybridRerankConfig, GraphMemoryError, GraphMemoryIndex,
    GraphRerankConfig, GraphRerankedHit, GraphSearchPreset, GraphViewPolicy, MemoryHit,
    MemoryRecord,
};

fn normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut data = vec![0.0f32; n * dim];
    for x in &mut data {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        *x = unit * 2.0 - 1.0;
    }
    for row in data.chunks_mut(dim) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row {
            *x /= norm.max(1e-6);
        }
    }
    data
}

fn vec_at(data: &[f32], dim: usize, row: usize) -> &[f32] {
    &data[row * dim..(row + 1) * dim]
}

fn assert_memory_hits_approx_eq(lhs: &[MemoryHit], rhs: &[MemoryHit], rel_eps: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert_eq!(left.id, right.id);
        assert!(
            (left.score - right.score).abs()
                <= rel_eps * left.score.abs().max(right.score.abs()).max(1.0),
            "memory hit score mismatch: {} != {}",
            left.score,
            right.score
        );
    }
}

fn assert_graph_rerank_hits_approx_eq(
    lhs: &[GraphRerankedHit],
    rhs: &[GraphRerankedHit],
    rel_eps: f32,
) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert_eq!(left.id, right.id);
        assert_eq!(left.depth, right.depth);
        assert_eq!(left.parent, right.parent);
        assert_eq!(left.title, right.title);
        assert_eq!(left.tags, right.tags);
        assert_eq!(left.source, right.source);
        assert_eq!(left.timestamp_ms, right.timestamp_ms);
        assert!(
            (left.vector_score - right.vector_score).abs()
                <= rel_eps
                    * left
                        .vector_score
                        .abs()
                        .max(right.vector_score.abs())
                        .max(1.0),
            "vector score mismatch for id {}: {} != {}",
            left.id,
            left.vector_score,
            right.vector_score
        );
        assert!(
            (left.graph_score - right.graph_score).abs()
                <= rel_eps * left.graph_score.abs().max(right.graph_score.abs()).max(1.0),
            "graph score mismatch for id {}: {} != {}",
            left.id,
            left.graph_score,
            right.graph_score
        );
        assert!(
            (left.score - right.score).abs()
                <= rel_eps * left.score.abs().max(right.score.abs()).max(1.0),
            "rerank score mismatch for id {}: {} != {}",
            left.id,
            left.score,
            right.score
        );
    }
}

fn build_memory() -> GraphMemoryIndex {
    let dim = 64;
    let data = normalized_vectors(6, dim, 0x6A9A_0001);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory
        .add_node(
            10,
            "root product brief",
            vec_at(&data, dim, 0),
            ["product", "search"],
        )
        .unwrap();
    memory
        .add_node(
            20,
            "architecture note",
            vec_at(&data, dim, 1),
            ["architecture", "memory"],
        )
        .unwrap();
    memory
        .add_node(
            30,
            "turboquant note",
            vec_at(&data, dim, 2),
            ["architecture", "vector"],
        )
        .unwrap();
    memory
        .add_node(
            40,
            "graph pruning note",
            vec_at(&data, dim, 3),
            ["graph", "memory"],
        )
        .unwrap();
    memory
        .add_node(
            50,
            "web ranking note",
            vec_at(&data, dim, 4),
            ["ranking", "web"],
        )
        .unwrap();
    memory
        .add_node(
            60,
            "outside island",
            vec_at(&data, dim, 5),
            ["architecture", "outside"],
        )
        .unwrap();

    memory.link_bidirectional(10, 20, 1.0).unwrap();
    memory.link_bidirectional(20, 30, 0.9).unwrap();
    memory.link_bidirectional(20, 40, 0.8).unwrap();
    memory.link_bidirectional(10, 50, 0.4).unwrap();
    memory
}

fn temp_paths(name: &str) -> (PathBuf, PathBuf) {
    let mut base = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    base.push(format!(
        "turbo-graph_{name}_{}_{}",
        std::process::id(),
        nanos
    ));
    let mut index = base.clone();
    index.set_extension("tv");
    let mut graph = base;
    graph.set_extension("tvgm");
    (index, graph)
}

#[test]
fn graph_view_search_respects_hops_and_tags() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1001);
    let report = memory.search_graph_view_with_stats(&query, 10, &[10], 2, &["architecture"]);
    let hits = report.hits;
    let ids: Vec<u64> = hits.iter().map(|hit| hit.id).collect();

    assert!(!report.view.cache_hit);
    assert_eq!(report.view.total_slots, 6);
    assert_eq!(report.view.selected_slots, 2);
    assert!((report.view.selectivity() - (2.0 / 6.0)).abs() < 1e-6);
    assert!(ids.contains(&20));
    assert!(ids.contains(&30));
    assert!(!ids.contains(&10), "root has no architecture tag");
    assert!(
        !ids.contains(&40),
        "graph/memory tag should be filtered out"
    );
    assert!(!ids.contains(&50), "ranking/web tag should be filtered out");
    assert!(!ids.contains(&60), "outside node is not in the graph view");
}

#[test]
fn graph_view_stats_report_cache_hits() {
    let mut memory = build_memory();
    let (_first_mask, first) = memory.graph_view_mask_with_stats(&[10], 2);
    let (_second_mask, second) = memory.graph_view_mask_with_stats(&[10], 2);

    assert!(!first.cache_hit);
    assert!(second.cache_hit);
    assert_eq!(first.selected_slots, second.selected_slots);
    assert_eq!(first.total_slots, second.total_slots);
}

#[test]
fn explain_graph_view_exports_depth_parent_and_edges() {
    let mut memory = build_memory();
    let trace = memory.explain_graph_view(&[10], 2);

    assert!(!trace.stats.cache_hit);
    assert_eq!(trace.stats.total_slots, 6);
    assert_eq!(trace.stats.selected_slots, 5);
    assert_eq!(trace.nodes.len(), 5);
    assert_eq!(trace.edges.len(), 8);

    let root = trace.nodes.iter().find(|node| node.id == 10).unwrap();
    assert_eq!(root.depth, 0);
    assert_eq!(root.parent, None);
    assert_eq!(root.path_weight, 1.0);

    let turboquant = trace.nodes.iter().find(|node| node.id == 30).unwrap();
    assert_eq!(turboquant.depth, 2);
    assert_eq!(turboquant.parent, Some(20));
    assert!((turboquant.via_weight - 0.9).abs() < 1e-6);
    assert!((turboquant.path_weight - 0.9).abs() < 1e-6);
    assert!(turboquant.tags.contains(&"architecture".to_string()));

    assert!(trace
        .edges
        .iter()
        .any(|edge| edge.from == 20 && edge.to == 30 && (edge.weight - 0.9).abs() < 1e-6));
}

#[test]
fn policy_graph_view_prefers_strong_paths_and_reuses_cache() {
    let mut memory = build_memory();
    let policy = GraphViewPolicy::new(2).with_max_nodes(3);
    let (mask, first) = memory.graph_view_mask_with_policy_stats(&[10], policy);

    assert!(!first.cache_hit);
    assert_eq!(first.total_slots, 6);
    assert_eq!(first.selected_slots, 3);
    assert!(mask.contains(memory.slot_of(10).unwrap()));
    assert!(mask.contains(memory.slot_of(20).unwrap()));
    assert!(mask.contains(memory.slot_of(30).unwrap()));
    assert!(!mask.contains(memory.slot_of(40).unwrap()));
    assert!(!mask.contains(memory.slot_of(50).unwrap()));

    let (_cached_mask, second) = memory.graph_view_mask_with_policy_stats(&[10], policy);
    assert!(second.cache_hit);
    assert_eq!(second.selected_slots, 3);

    let trace = memory.explain_graph_view_with_policy(&[10], policy);
    let ids: Vec<u64> = trace.nodes.iter().map(|node| node.id).collect();
    assert_eq!(ids, vec![10, 20, 30]);
    assert_eq!(trace.edges.len(), 4);
    let turboquant = trace.nodes.iter().find(|node| node.id == 30).unwrap();
    assert_eq!(turboquant.parent, Some(20));
    assert!((turboquant.path_weight - 0.9).abs() < 1e-6);
}

#[test]
fn search_with_slot_mask_k_zero_returns_empty_single_and_batch() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let first = normalized_vectors(1, dim, 0x6A9A_3001);
    let second = normalized_vectors(1, dim, 0x6A9A_3002);
    let mut batch_queries = first.clone();
    batch_queries.extend_from_slice(&second);

    let mask = memory.tag_view_mask("architecture");
    assert!(memory.search_with_slot_mask(&first, 0, &mask).is_empty());

    let batched = memory.search_with_slot_mask_batch(&batch_queries, 0, &mask);
    assert_eq!(batched.len(), 2);
    assert!(batched[0].is_empty());
    assert!(batched[1].is_empty());
}

#[test]
fn prepared_views_support_single_and_batch_queries() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let query_0 = normalized_vectors(1, dim, 0x6A9A_3003);
    let query_1 = normalized_vectors(1, dim, 0x6A9A_3004);
    let mut batch_queries = query_0.clone();
    batch_queries.extend_from_slice(&query_1);

    let prepared = memory.prepare_graph_view(&[10], 2);
    let expected = vec![
        prepared.search(&memory, &query_0, 3),
        prepared.search(&memory, &query_1, 3),
    ];
    let prepared_batch = prepared.search_batch(&memory, &batch_queries, 3);
    assert_eq!(prepared_batch.len(), expected.len());
    for (batched, expected) in prepared_batch.iter().zip(expected.iter()) {
        assert_memory_hits_approx_eq(batched, expected, 1e-6);
    }

    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(2);
    let prepared_policy = memory.prepare_graph_view_with_policy_metadata(
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );
    let rerank_single_0 = prepared_policy
        .search_rerank(&memory, &query_0, 3, rerank)
        .hits;
    let rerank_single_1 = prepared_policy
        .search_rerank(&memory, &query_1, 3, rerank)
        .hits;

    let rerank_batch = prepared_policy.search_rerank_batch(&memory, &batch_queries, 3, rerank);
    assert_eq!(rerank_batch.hits.len(), 2);
    assert_graph_rerank_hits_approx_eq(&rerank_batch.hits[0], &rerank_single_0, 1e-6);
    assert_graph_rerank_hits_approx_eq(&rerank_batch.hits[1], &rerank_single_1, 1e-6);

    let rerank_zero = prepared_policy.search_rerank(&memory, &query_0, 0, rerank);
    assert_eq!(rerank_zero.prefetch_k, 0);
    assert!(rerank_zero.hits.is_empty());

    let rerank_zero_batch = prepared_policy.search_rerank_batch(&memory, &batch_queries, 0, rerank);
    assert_eq!(rerank_zero_batch.prefetch_k, 0);
    assert_eq!(rerank_zero_batch.hits.len(), 2);
    assert!(rerank_zero_batch.hits[0].is_empty());
    assert!(rerank_zero_batch.hits[1].is_empty());
}

#[test]
fn graph_view_mask_with_policy_stats_tracks_cache_hits_and_misses() {
    let mut memory = build_memory();
    let policy = GraphViewPolicy::new(2).with_max_nodes(3);

    let before = memory.cache_stats();
    let _ = memory.graph_view_mask_with_policy_stats(&[10], policy);
    let after_first = memory.cache_stats();
    assert_eq!(
        after_first.policy_view_misses,
        before.policy_view_misses + 1
    );
    assert_eq!(
        after_first.policy_visit_misses,
        before.policy_visit_misses + 1
    );
    assert_eq!(after_first.policy_visit_hits, before.policy_visit_hits);

    let _ = memory.graph_view_mask_with_policy_stats(&[10], policy);
    let after_second = memory.cache_stats();
    assert_eq!(
        after_second.policy_view_hits,
        after_first.policy_view_hits + 1
    );
    assert_eq!(
        after_second.policy_visit_hits,
        after_first.policy_visit_hits
    );
    assert_eq!(
        after_second.policy_view_misses,
        after_first.policy_view_misses
    );
}

#[test]
fn metadata_masks_track_cache_hits_and_misses() {
    let dim = 64;
    let data = normalized_vectors(4, dim, 0x6A9A_3005);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records = vec![
        MemoryRecord::new(1, "alpha", ["architecture"])
            .with_source("kuku.mom")
            .with_timestamp_ms(10),
        MemoryRecord::new(2, "beta", ["architecture"])
            .with_source("liner")
            .with_timestamp_ms(20),
        MemoryRecord::new(3, "gamma", ["other"])
            .with_source("kuku.mom")
            .with_timestamp_ms(30),
        MemoryRecord::new(4, "delta", ["other"])
            .with_source("kuku.mom")
            .with_timestamp_ms(40),
    ];
    memory.add_records(&data, records).unwrap();
    memory.link_directed(1, 2, 1.0).unwrap();
    memory.link_directed(1, 3, 0.8).unwrap();

    let policy = GraphViewPolicy::new(2).with_max_nodes(4);
    let before = memory.cache_stats();
    let (mask_first, first_plan) = memory.graph_view_mask_with_policy_metadata_plan(
        &[1],
        policy,
        &["architecture"],
        &["kuku.mom"],
        Some(5),
        Some(35),
    );
    assert!(mask_first.count() > 0);
    assert!(!first_plan.combined_cache_hit);
    let after_first = memory.cache_stats();
    assert_eq!(after_first.tag_mask_misses, before.tag_mask_misses + 1);
    assert_eq!(
        after_first.source_mask_misses,
        before.source_mask_misses + 1
    );
    assert_eq!(after_first.time_mask_misses, before.time_mask_misses + 1);

    let (_mask_second, second_plan) = memory.graph_view_mask_with_policy_metadata_plan(
        &[1],
        policy,
        &["architecture"],
        &["kuku.mom"],
        Some(5),
        Some(35),
    );
    assert!(second_plan.combined_cache_hit);
    let after_second = memory.cache_stats();
    assert_eq!(after_second.tag_mask_misses, after_first.tag_mask_misses);
    assert_eq!(
        after_second.source_mask_misses,
        after_first.source_mask_misses
    );
    assert_eq!(after_second.time_mask_misses, after_first.time_mask_misses);

    let _ = memory.tag_view_mask("architecture");
    let _ = memory.source_view_mask("kuku.mom");
    let _ = memory.time_range_view_mask(Some(5), Some(35));
    let after_third = memory.cache_stats();
    assert_eq!(after_third.tag_mask_hits, after_second.tag_mask_hits + 1);
    assert_eq!(
        after_third.source_mask_hits,
        after_second.source_mask_hits + 1
    );
    assert_eq!(after_third.time_mask_hits, after_second.time_mask_hits + 1);
}

#[test]
fn cache_ratio_metrics_are_consistent() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_3006);

    let policy = GraphViewPolicy::new(2).with_max_nodes(4);
    let _ = memory.search_graph_view_with_policy_metadata_plan(
        &query,
        4,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );
    let _ = memory.search_graph_view_with_policy_metadata_plan(
        &query,
        4,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );
    let _ = memory.tag_view_mask("architecture");
    let _ = memory.tag_view_mask("architecture");
    let _ = memory.source_view_mask("kuku.mom");
    let _ = memory.source_view_mask("kuku.mom");
    let _ = memory.time_range_view_mask(Some(0), Some(10_000));
    let _ = memory.time_range_view_mask(Some(0), Some(10_000));

    let stats = memory.cache_stats();
    assert_eq!(
        stats.query_cache_hits() + stats.query_cache_misses(),
        stats.query_accesses()
    );
    assert_eq!(
        stats.metadata_cache_hits() + stats.metadata_cache_misses(),
        stats.metadata_accesses()
    );
    assert_eq!(
        stats.cache_accesses(),
        stats.query_accesses() + stats.metadata_accesses()
    );
    assert_eq!(
        stats.query_cache_hits() + stats.metadata_cache_hits(),
        stats.cache_accesses() - (stats.query_cache_misses() + stats.metadata_cache_misses())
    );
    assert_eq!(
        stats.query_cache_misses() + stats.metadata_cache_misses(),
        stats.cache_accesses() - (stats.query_cache_hits() + stats.metadata_cache_hits())
    );
    if stats.query_accesses() > 0 {
        assert!(
            (stats.query_cache_hit_ratio() + stats.query_cache_miss_ratio() - 1.0).abs() < 1e-6
        );
    }
    if stats.metadata_accesses() > 0 {
        assert!(
            (stats.metadata_cache_hit_ratio() + stats.metadata_cache_miss_ratio() - 1.0).abs()
                < 1e-6
        );
    }
    if stats.cache_accesses() > 0 {
        assert!((stats.cache_hit_ratio() + stats.cache_miss_ratio() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn prepared_view_timed_batch_rerank_matches_eager_path() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let query_0 = normalized_vectors(1, dim, 0x6A9A_3007);
    let query_1 = normalized_vectors(1, dim, 0x6A9A_3008);
    let mut batch_queries = query_0.clone();
    batch_queries.extend_from_slice(&query_1);

    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(2);
    let prepared = memory.prepare_graph_view_with_policy_metadata(
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );

    let eager = prepared.search_rerank_batch(&memory, &batch_queries, 3, rerank);
    let timed = prepared.search_rerank_batch_timed(&memory, &batch_queries, 3, rerank);
    assert_eq!(timed.prefetch_k, eager.prefetch_k);
    assert_eq!(timed.plan, eager.plan);
    assert_eq!(timed.hits, eager.hits);
    assert!(timed.telemetry.total_ns >= timed.telemetry.view_build_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.vector_search_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.rerank_ns);

    let zero_k = prepared.search_rerank_batch(&memory, &batch_queries, 0, rerank);
    assert_eq!(zero_k.prefetch_k, 0);
    assert_eq!(zero_k.hits.len(), 2);
    assert!(zero_k.hits[0].is_empty());
    assert!(zero_k.hits[1].is_empty());

    let timed_zero_k = prepared.search_rerank_batch_timed(&memory, &batch_queries, 0, rerank);
    assert_eq!(timed_zero_k.prefetch_k, 0);
    assert_eq!(timed_zero_k.hits.len(), 2);
    assert!(timed_zero_k.hits[0].is_empty());
    assert!(timed_zero_k.hits[1].is_empty());
}

#[test]
fn policy_graph_view_respects_active_block_budget() {
    let dim = 64;
    let n = 96;
    let data = normalized_vectors(n, dim, 0x6A9A_2012);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| MemoryRecord::new(i as u64, format!("node {i}"), ["doc"]))
        .collect();
    memory.add_records(&data, records).unwrap();
    memory.link_directed(0, 40, 0.99).unwrap();
    memory.link_directed(0, 1, 0.90).unwrap();
    memory.link_directed(1, 2, 0.80).unwrap();

    let one_block_policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_max_active_blocks(1);
    let (one_block, one_block_stats) =
        memory.graph_view_mask_with_policy_stats(&[0], one_block_policy);
    assert!(!one_block_stats.cache_hit);
    assert_eq!(one_block.count(), 3);
    assert_eq!(one_block.active_block_count(), 1);
    assert!(one_block.contains(memory.slot_of(0).unwrap()));
    assert!(one_block.contains(memory.slot_of(1).unwrap()));
    assert!(one_block.contains(memory.slot_of(2).unwrap()));
    assert!(!one_block.contains(memory.slot_of(40).unwrap()));

    let two_block_policy = one_block_policy.with_max_active_blocks(2);
    let (two_blocks, first_two_block_stats) =
        memory.graph_view_mask_with_policy_stats(&[0], two_block_policy);
    assert!(
        !first_two_block_stats.cache_hit,
        "active block budget must be part of the cache key"
    );
    assert_eq!(two_blocks.count(), 4);
    assert_eq!(two_blocks.active_block_count(), 2);
    assert!(two_blocks.contains(memory.slot_of(40).unwrap()));

    let (_cached, cached_stats) = memory.graph_view_mask_with_policy_stats(&[0], two_block_policy);
    assert!(cached_stats.cache_hit);
}

#[test]
fn preset_derives_policy_and_rerank_budgets() {
    let dim = 64;
    let n = 96;
    let data = normalized_vectors(n, dim, 0x6A9A_2013);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| MemoryRecord::new(i as u64, format!("node {i}"), ["doc"]))
        .collect();
    memory.add_records(&data, records).unwrap();
    memory.link_directed(0, 40, 0.99).unwrap();
    memory.link_directed(0, 1, 0.90).unwrap();
    memory.link_directed(1, 2, 0.80).unwrap();

    let preset = GraphSearchPreset::low_latency()
        .with_max_hops(2)
        .with_target_active_blocks(1)
        .with_nodes_per_active_block(8)
        .with_min_prefetch(2)
        .with_prefetch_factor(1);
    let tuning = memory.tuning_for_preset(3, preset);
    assert_eq!(tuning.policy.max_hops, 2);
    assert_eq!(tuning.policy.max_active_blocks, 1);
    assert_eq!(tuning.policy.max_nodes, 8);
    assert_eq!(tuning.rerank.min_prefetch, 2);
    assert_eq!(tuning.rerank.prefetch_factor, 1);

    let report = memory.explain_graph_search_with_preset(
        vec_at(&data, dim, 0),
        3,
        &[0],
        preset,
        &["doc"],
        &[],
        None,
        None,
    );
    assert!(report.plan.active_blocks <= 1);
    assert!(report.trace.nodes.iter().any(|node| node.id == 0));
    assert!(report.trace.nodes.iter().any(|node| node.id == 1));
    assert!(!report.trace.nodes.iter().any(|node| node.id == 40));
    assert!(report.prefetch_k <= report.plan.selected_slots);
}

#[test]
fn policy_metadata_search_uses_budgeted_combined_cache() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1007);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.75);

    let first = memory.search_graph_view_with_policy_metadata_plan(
        &query,
        10,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );
    let first_ids: Vec<u64> = first.hits.iter().map(|hit| hit.id).collect();

    assert!(!first.plan.combined_cache_hit);
    assert!(!first.plan.graph_cache_hit);
    assert_eq!(first.plan.graph_slots, 4);
    assert_eq!(first.plan.selected_slots, 2);
    assert_eq!(first.plan.active_blocks, 1);
    assert!(first_ids.contains(&20));
    assert!(first_ids.contains(&30));
    assert!(
        !first_ids.contains(&40),
        "tag filter removes graph/memory note"
    );
    assert!(!first_ids.contains(&50), "weak 0.4 path is below threshold");

    let second = memory.search_graph_view_with_policy_metadata_plan(
        &query,
        10,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
    );
    assert!(second.plan.combined_cache_hit);
    assert!(second.plan.graph_cache_hit);
    assert_eq!(second.plan.selected_slots, first.plan.selected_slots);
    assert_eq!(
        second.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        first_ids
    );
}

#[test]
fn policy_rerank_blends_vector_score_with_graph_path_score() {
    let dim = 64;
    let vector = normalized_vectors(1, dim, 0x6A9A_2008);
    let mut data = Vec::new();
    for _ in 0..3 {
        data.extend_from_slice(&vector);
    }

    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory
        .add_records(
            &data,
            vec![
                MemoryRecord::new(1, "seed", ["seed"]),
                MemoryRecord::new(2, "strong context", ["doc"]),
                MemoryRecord::new(3, "weak context", ["doc"]),
            ],
        )
        .unwrap();
    memory.link_directed(1, 2, 0.9).unwrap();
    memory.link_directed(1, 3, 0.2).unwrap();

    let report = memory.search_graph_view_with_policy_metadata_rerank(
        &vector,
        2,
        &[1],
        GraphViewPolicy::new(1),
        &["doc"],
        &[],
        None,
        None,
        GraphRerankConfig::new(0.0, 1.0)
            .with_prefetch_factor(1)
            .with_min_prefetch(1),
    );

    assert_eq!(report.prefetch_k, 2);
    assert_eq!(report.plan.selected_slots, 2);
    assert_eq!(
        report.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert!((report.hits[0].graph_score - 0.9).abs() < 1e-6);
    assert!((report.hits[1].graph_score - 0.2).abs() < 1e-6);
    assert!(report.hits[0].score > report.hits[1].score);
    assert_eq!(report.hits[0].depth, 1);
    assert_eq!(report.hits[0].parent, Some(1));
}

#[test]
fn timed_policy_rerank_reports_latency_and_mask_skips() {
    let dim = 64;
    let n = 96;
    let data = normalized_vectors(n, dim, 0x6A9A_2009);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| MemoryRecord::new(i as u64, format!("node {i}"), ["doc"]))
        .collect();
    memory.add_records(&data, records).unwrap();
    memory.link_directed(80, 81, 0.8).unwrap();

    let report = memory.search_graph_view_with_policy_metadata_rerank_timed(
        vec_at(&data, dim, 80),
        2,
        &[80],
        GraphViewPolicy::new(1).with_max_nodes(2),
        &["doc"],
        &[],
        None,
        None,
        GraphRerankConfig::new(1.0, 0.1)
            .with_prefetch_factor(1)
            .with_min_prefetch(1),
    );

    assert_eq!(report.plan.total_slots, n);
    assert_eq!(report.plan.selected_slots, 2);
    assert_eq!(report.plan.active_blocks, 1);
    assert_eq!(report.prefetch_k, 2);
    assert_eq!(report.hits.len(), 2);
    assert!(report.telemetry.total_ns >= report.telemetry.view_build_ns);
    assert!(report.telemetry.total_ns >= report.telemetry.vector_search_ns);
    assert!(report.telemetry.total_ns >= report.telemetry.rerank_ns);
    assert!(
        report.telemetry.blocks_skipped_by_mask >= 2,
        "96 slots = 3 SIMD blocks, one active block should skip at least two"
    );

    let cached = memory.search_graph_view_with_policy_metadata_rerank_timed(
        vec_at(&data, dim, 80),
        2,
        &[80],
        GraphViewPolicy::new(1).with_max_nodes(2),
        &["doc"],
        &[],
        None,
        None,
        GraphRerankConfig::new(1.0, 0.1)
            .with_prefetch_factor(1)
            .with_min_prefetch(1),
    );
    assert!(cached.plan.combined_cache_hit);
    assert!(cached.plan.graph_cache_hit);
    assert_eq!(cached.plan.selected_slots, report.plan.selected_slots);
}

#[test]
fn explained_policy_rerank_returns_trace_hits_and_telemetry() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2010);
    let report = memory.explain_graph_search_with_policy_metadata_rerank_timed(
        &query,
        3,
        &[10],
        GraphViewPolicy::new(2)
            .with_max_nodes(4)
            .with_min_path_weight(0.5),
        &["architecture"],
        &[],
        None,
        None,
        GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(4),
    );

    assert_eq!(report.plan.total_slots, 6);
    assert_eq!(report.plan.graph_slots, 4);
    assert_eq!(report.plan.selected_slots, 2);
    assert_eq!(report.prefetch_k, 2);
    assert_eq!(report.trace.nodes.len(), 4);
    assert!(report.trace.nodes.iter().any(|node| node.id == 10));
    assert!(report.trace.nodes.iter().any(|node| node.id == 40));
    assert!(report
        .hits
        .iter()
        .all(|hit| hit.tags.contains(&"architecture".to_string())));
    assert!(report.hits.iter().any(|hit| hit.id == 20));
    assert!(report.hits.iter().any(|hit| hit.id == 30));
    assert!(report.telemetry.trace_build_ns <= report.telemetry.total_ns);
    assert!(report.telemetry.vector_search_ns <= report.telemetry.total_ns);

    let snapshot = report.debug_snapshot();
    assert_eq!(snapshot.summary.total_slots, 6);
    assert_eq!(snapshot.summary.graph_slots, 4);
    assert_eq!(snapshot.summary.selected_slots, 2);
    assert_eq!(snapshot.summary.hit_count, 2);
    assert_eq!(snapshot.summary.trace_node_count, 4);
    assert_eq!(snapshot.summary.trace_edge_count, report.trace.edges.len());
    assert_eq!(snapshot.hits.len(), report.hits.len());
    assert_eq!(snapshot.hits[0].rank, 1);
    let root = snapshot.nodes.iter().find(|node| node.id == 10).unwrap();
    assert_eq!(root.hit_rank, None);
    assert_eq!(root.score, None);
    let hit_node = snapshot
        .nodes
        .iter()
        .find(|node| node.hit_rank == Some(1))
        .unwrap();
    assert!(hit_node.score.is_some());
    assert!(hit_node.vector_score.is_some());
    assert!(hit_node.graph_score.is_some());
    assert_eq!(snapshot.edges.len(), report.trace.edges.len());
}

#[cfg(feature = "serde")]
#[test]
fn debug_snapshot_serializes_with_serde_feature() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2011);
    let report = memory.explain_graph_search_with_policy_metadata_rerank_timed(
        &query,
        3,
        &[10],
        GraphViewPolicy::new(2)
            .with_max_nodes(4)
            .with_min_path_weight(0.5),
        &["architecture"],
        &[],
        None,
        None,
        GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(4),
    );
    let snapshot = report.debug_snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(json.contains("active_block_selectivity"));
    assert!(json.contains("vector_score"));

    let decoded: turbo_graph::GraphSearchDebugSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.summary.hit_count, snapshot.summary.hit_count);
    assert_eq!(
        decoded.summary.trace_node_count,
        snapshot.summary.trace_node_count
    );
    assert_eq!(decoded.hits.len(), snapshot.hits.len());
    assert_eq!(decoded.nodes.len(), snapshot.nodes.len());
    assert_eq!(decoded.edges.len(), snapshot.edges.len());
    assert!(decoded.nodes.iter().any(|node| node.hit_rank == Some(1)));

    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(16)
        .with_max_active_blocks(3)
        .with_min_path_weight(0.25);
    let policy_json = serde_json::to_string(&policy).unwrap();
    let decoded_policy: GraphViewPolicy = serde_json::from_str(&policy_json).unwrap();
    assert_eq!(decoded_policy.max_nodes, 16);
    assert_eq!(decoded_policy.max_active_blocks, 3);
    assert_eq!(decoded_policy.min_path_weight, 0.25);

    let hybrid_config = GraphHybridRerankConfig::new(0.5, 0.25, 0.75)
        .with_candidate_score_normalization(GraphCandidateScoreNormalization::MinMax)
        .with_prefetch_factor(3)
        .with_min_prefetch(11);
    let hybrid_config_json = serde_json::to_string(&hybrid_config).unwrap();
    let decoded_hybrid_config: GraphHybridRerankConfig =
        serde_json::from_str(&hybrid_config_json).unwrap();
    assert_eq!(decoded_hybrid_config, hybrid_config);

    let preset = GraphSearchPreset::balanced()
        .with_target_active_blocks(4)
        .with_nodes_per_active_block(12);
    let preset_json = serde_json::to_string(&preset).unwrap();
    let decoded_preset: GraphSearchPreset = serde_json::from_str(&preset_json).unwrap();
    assert_eq!(decoded_preset.target_active_blocks, 4);
    assert_eq!(decoded_preset.nodes_per_active_block, 12);

    let cache_json = serde_json::to_string(&memory.cache_stats()).unwrap();
    let decoded_cache: turbo_graph::GraphMemoryCacheStats =
        serde_json::from_str(&cache_json).unwrap();
    assert_eq!(
        decoded_cache.total_entries,
        memory.cache_stats().total_entries
    );

    let budget = preset.cache_budget(memory.len());
    let budget_json = serde_json::to_string(&budget).unwrap();
    let decoded_budget: turbo_graph::GraphMemoryCacheBudget =
        serde_json::from_str(&budget_json).unwrap();
    assert_eq!(decoded_budget, budget);
    assert_eq!(
        decoded_budget.total_entries(),
        decoded_budget.query_entries() + decoded_budget.metadata_entries()
    );

    let candidate_plan = turbo_graph::GraphCandidateSearchPlan {
        total_slots: 10,
        graph_slots: 8,
        metadata_slots: 5,
        candidate_input_ids: 5,
        candidate_slots: 3,
        candidate_missing_ids: 1,
        candidate_duplicate_ids: 1,
        selected_slots: 2,
        active_blocks: 1,
        graph_cache_hit: true,
        combined_cache_hit: false,
    };
    let candidate_json = serde_json::to_string(&candidate_plan).unwrap();
    let decoded_candidate: turbo_graph::GraphCandidateSearchPlan =
        serde_json::from_str(&candidate_json).unwrap();
    assert_eq!(decoded_candidate, candidate_plan);

    let candidate_report = memory.search_graph_view_with_policy_metadata_candidates_rerank_timed(
        &query,
        2,
        &[10],
        GraphViewPolicy::new(2)
            .with_max_nodes(4)
            .with_min_path_weight(0.5),
        &["architecture"],
        &[],
        None,
        None,
        &[20, 30],
        GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(1),
    );
    let candidate_report_json = serde_json::to_string(&candidate_report).unwrap();
    let decoded_candidate_report: turbo_graph::GraphCandidateTimedRerankedSearchReport =
        serde_json::from_str(&candidate_report_json).unwrap();
    assert_eq!(
        decoded_candidate_report.plan.selected_slots,
        candidate_report.plan.selected_slots
    );
    assert_eq!(
        decoded_candidate_report.prefetch_k,
        candidate_report.prefetch_k
    );

    let hybrid_report = memory
        .search_graph_view_with_policy_metadata_candidate_scores_hybrid_timed(
            &query,
            2,
            &[10],
            GraphViewPolicy::new(2)
                .with_max_nodes(4)
                .with_min_path_weight(0.5),
            &["architecture"],
            &[],
            None,
            None,
            &[(20, 0.1), (30, 1.0)],
            GraphHybridRerankConfig::new(0.0, 0.0, 1.0).with_min_prefetch(1),
        );
    let hybrid_json = serde_json::to_string(&hybrid_report).unwrap();
    let decoded_hybrid: turbo_graph::GraphCandidateTimedHybridSearchReport =
        serde_json::from_str(&hybrid_json).unwrap();
    assert_eq!(decoded_hybrid.hits.len(), hybrid_report.hits.len());
    assert_eq!(
        decoded_hybrid.hits[0].candidate_score,
        hybrid_report.hits[0].candidate_score
    );

    let hybrid_explained = memory
        .explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed(
            &query,
            2,
            &[10],
            GraphViewPolicy::new(2)
                .with_max_nodes(4)
                .with_min_path_weight(0.5),
            &["architecture"],
            &[],
            None,
            None,
            &[(20, 0.1), (30, 1.0)],
            GraphHybridRerankConfig::new(0.0, 0.0, 1.0).with_min_prefetch(1),
        );
    let hybrid_snapshot = hybrid_explained.debug_snapshot();
    let hybrid_snapshot_json = serde_json::to_string(&hybrid_snapshot).unwrap();
    assert!(hybrid_snapshot_json.contains("candidate_score"));
    let decoded_hybrid_snapshot: turbo_graph::GraphCandidateHybridSearchDebugSnapshot =
        serde_json::from_str(&hybrid_snapshot_json).unwrap();
    assert_eq!(
        decoded_hybrid_snapshot.hits[0].candidate_score,
        hybrid_snapshot.hits[0].candidate_score
    );

    let candidate_explained = memory
        .explain_graph_search_with_policy_metadata_candidates_rerank_timed(
            &query,
            2,
            &[10],
            GraphViewPolicy::new(2)
                .with_max_nodes(4)
                .with_min_path_weight(0.5),
            &["architecture"],
            &[],
            None,
            None,
            &[20, 30],
            GraphRerankConfig::new(1.0, 0.2).with_min_prefetch(1),
        );
    let candidate_snapshot = candidate_explained.debug_snapshot();
    let candidate_snapshot_json = serde_json::to_string(&candidate_snapshot).unwrap();
    assert!(candidate_snapshot_json.contains("candidate_slots"));
    let decoded_candidate_snapshot: turbo_graph::GraphCandidateSearchDebugSnapshot =
        serde_json::from_str(&candidate_snapshot_json).unwrap();
    assert_eq!(
        decoded_candidate_snapshot.summary.candidate_slots,
        candidate_snapshot.summary.candidate_slots
    );
    assert_eq!(
        decoded_candidate_snapshot.summary.hit_count,
        candidate_snapshot.summary.hit_count
    );
}

#[test]
fn search_graph_view_with_trace_keeps_full_trace_and_filtered_stats() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1006);
    let report = memory.search_graph_view_with_trace(&query, 10, &[10], 2, &["architecture"]);
    let hit_ids: Vec<u64> = report.hits.iter().map(|hit| hit.id).collect();

    assert_eq!(report.trace.stats.selected_slots, 5);
    assert_eq!(report.trace.nodes.len(), 5);
    assert_eq!(report.view.selected_slots, 2);
    assert!(hit_ids.contains(&20));
    assert!(hit_ids.contains(&30));
    assert!(!hit_ids.contains(&40));
}

#[test]
fn add_records_batches_vectors_and_dedups_tags() {
    let dim = 64;
    let data = normalized_vectors(3, dim, 0x6A9A_2001);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory
        .add_records(
            &data,
            vec![
                MemoryRecord::new(1, "one", ["b", "a", "a"]),
                MemoryRecord::new(2, "two", ["b"]),
                MemoryRecord::new(3, "three", ["c"]),
            ],
        )
        .unwrap();

    assert_eq!(memory.len(), 3);
    assert_eq!(memory.slot_of(1), Some(0));
    assert_eq!(memory.slot_of(3), Some(2));
    assert_eq!(memory.record(1).unwrap().tags, vec!["a", "b"]);
}

#[test]
fn tag_view_mask_uses_cache_and_invalidates_on_add_remove() {
    let mut memory = build_memory();
    let architecture = memory.tag_view_mask("architecture");
    assert_eq!(architecture.count(), 3);

    let data = normalized_vectors(1, memory.dim(), 0x6A9A_2002);
    let fresh_before = memory.tag_view_mask("fresh");
    assert_eq!(fresh_before.count(), 0);
    memory
        .add_node(70, "fresh note", &data, ["fresh", "architecture"])
        .unwrap();
    let fresh_after = memory.tag_view_mask("fresh");
    assert_eq!(fresh_after.count(), 1);
    assert!(fresh_after.contains(memory.slot_of(70).unwrap()));

    assert!(memory.remove_node(20));
    let architecture_after_remove = memory.tag_view_mask("architecture");
    assert_eq!(architecture_after_remove.count(), 3);
    assert!(!memory.contains(20));
    assert!(architecture_after_remove.contains(memory.slot_of(30).unwrap()));
    assert!(architecture_after_remove.contains(memory.slot_of(60).unwrap()));
    assert!(architecture_after_remove.contains(memory.slot_of(70).unwrap()));
}

#[test]
fn replace_record_metadata_keeps_graph_cache_and_refreshes_metadata_views() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2016);

    let first = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
    );
    assert!(!first.plan.combined_cache_hit);
    assert_eq!(first.plan.selected_slots, 2);

    let cached = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
    );
    assert!(cached.plan.combined_cache_hit);
    assert!(cached.plan.graph_cache_hit);
    assert_eq!(cached.plan.selected_slots, 2);

    memory
        .replace_record_metadata(
            MemoryRecord::new(
                40,
                "promoted graph architecture note",
                ["memory", "architecture"],
            )
            .with_source("liner")
            .with_timestamp_ms(4_000),
        )
        .unwrap();

    let (_graph_mask, graph_stats) = memory.graph_view_mask_with_stats(&[10], 2);
    assert!(
        graph_stats.cache_hit,
        "metadata-only update should keep raw graph-view cache"
    );
    assert_eq!(graph_stats.selected_slots, 5);

    let after = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
    );
    assert!(
        !after.plan.combined_cache_hit,
        "combined graph+metadata cache must be rebuilt after tag changes"
    );
    assert!(after.plan.graph_cache_hit);
    assert_eq!(after.plan.selected_slots, 3);
    assert!(after.hits.iter().any(|hit| hit.id == 40));

    assert_eq!(memory.tag_view_mask("graph").count(), 0);
    assert_eq!(memory.tag_view_mask("architecture").count(), 4);
    assert_eq!(memory.source_view_mask("liner").count(), 1);
    assert_eq!(
        memory
            .time_range_view_mask(Some(3_500), Some(4_500))
            .count(),
        1
    );
    assert_eq!(
        memory.record(40).unwrap().title,
        "promoted graph architecture note"
    );

    let missing = memory
        .replace_record_metadata(MemoryRecord::new(999, "missing", ["doc"]))
        .unwrap_err();
    assert!(matches!(missing, GraphMemoryError::MissingId(999)));
}

#[test]
fn source_and_time_masks_filter_search_and_roundtrip() {
    let dim = 64;
    let data = normalized_vectors(4, dim, 0x6A9A_2003);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory
        .add_records(
            &data,
            vec![
                MemoryRecord::new(1, "kuku root", ["search"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(1_000),
                MemoryRecord::new(2, "kuku fresh", ["search"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(2_000),
                MemoryRecord::new(3, "liner fresh", ["search"])
                    .with_source("liner")
                    .with_timestamp_ms(3_000),
                MemoryRecord::new(4, "kuku other", ["other"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(4_000),
            ],
        )
        .unwrap();
    memory.link_bidirectional(1, 2, 1.0).unwrap();
    memory.link_bidirectional(2, 3, 1.0).unwrap();
    memory.link_bidirectional(3, 4, 1.0).unwrap();

    assert_eq!(memory.source_view_mask("kuku.mom").count(), 3);
    assert_eq!(
        memory
            .time_range_view_mask(Some(1_500), Some(3_500))
            .count(),
        2
    );

    let query = normalized_vectors(1, dim, 0x6A9A_2004);
    let report = memory.search_graph_view_with_metadata(
        &query,
        10,
        &[1],
        3,
        &["search"],
        &["kuku.mom"],
        Some(1_500),
        Some(3_500),
    );
    assert_eq!(report.view.selected_slots, 1);
    assert_eq!(report.hits.len(), 1);
    assert_eq!(report.hits[0].id, 2);
    assert_eq!(report.hits[0].source.as_deref(), Some("kuku.mom"));
    assert_eq!(report.hits[0].timestamp_ms, Some(2_000));

    let trace_report = memory.search_graph_view_with_metadata_trace(
        &query,
        10,
        &[1],
        3,
        &["search"],
        &["kuku.mom"],
        Some(1_500),
        Some(3_500),
    );
    assert_eq!(trace_report.trace.nodes.len(), 4);
    assert_eq!(trace_report.view.selected_slots, 1);

    let (index_path, graph_path) = temp_paths("graph_metadata_roundtrip");
    memory.write(&index_path, &graph_path).unwrap();
    let mut loaded = GraphMemoryIndex::load(&index_path, &graph_path).unwrap();
    fs::remove_file(&index_path).ok();
    fs::remove_file(&graph_path).ok();

    assert_eq!(
        loaded.record(2).unwrap().source.as_deref(),
        Some("kuku.mom")
    );
    assert_eq!(loaded.record(2).unwrap().timestamp_ms, Some(2_000));
    assert_eq!(loaded.source_view_mask("kuku.mom").count(), 3);
    assert_eq!(
        loaded
            .time_range_view_mask(Some(1_500), Some(3_500))
            .count(),
        2
    );

    assert!(loaded.remove_node(2));
    assert_eq!(loaded.source_view_mask("kuku.mom").count(), 2);
    assert_eq!(
        loaded
            .time_range_view_mask(Some(1_500), Some(3_500))
            .count(),
        1
    );
}

#[test]
fn metadata_search_plan_reuses_combined_view_cache() {
    let dim = 64;
    let data = normalized_vectors(5, dim, 0x6A9A_2005);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory
        .add_records(
            &data[..4 * dim],
            vec![
                MemoryRecord::new(1, "root", ["search"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(1_000),
                MemoryRecord::new(2, "fresh kuku", ["search", "architecture"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(2_000),
                MemoryRecord::new(3, "fresh liner", ["search", "architecture"])
                    .with_source("liner")
                    .with_timestamp_ms(2_500),
                MemoryRecord::new(4, "other kuku", ["other"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(3_000),
            ],
        )
        .unwrap();
    memory.link_bidirectional(1, 2, 1.0).unwrap();
    memory.link_bidirectional(2, 3, 1.0).unwrap();
    memory.link_bidirectional(3, 4, 1.0).unwrap();

    let query = normalized_vectors(1, dim, 0x6A9A_2006);
    let first = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[1],
        3,
        &["architecture", "search"],
        &["kuku.mom"],
        Some(1_500),
        Some(2_500),
    );
    assert!(!first.plan.combined_cache_hit);
    assert!(!first.plan.graph_cache_hit);
    assert_eq!(first.plan.total_slots, 4);
    assert_eq!(first.plan.graph_slots, 4);
    assert_eq!(first.plan.selected_slots, 1);
    assert_eq!(first.plan.active_blocks, 1);
    assert_eq!(first.plan.selectivity(), 0.25);
    assert_eq!(first.plan.graph_selectivity(), 1.0);
    assert_eq!(first.plan.active_block_selectivity(), 1.0);
    assert_eq!(first.hits.len(), 1);
    assert_eq!(first.hits[0].id, 2);

    let second = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[1],
        3,
        &["search", "architecture"],
        &["kuku.mom", "kuku.mom"],
        Some(1_500),
        Some(2_500),
    );
    assert!(second.plan.combined_cache_hit);
    assert!(second.plan.graph_cache_hit);
    assert_eq!(second.plan.selected_slots, first.plan.selected_slots);
    assert_eq!(second.plan.active_blocks, first.plan.active_blocks);
    assert_eq!(second.hits[0].id, 2);

    memory
        .add_records(
            &data[4 * dim..],
            vec![
                MemoryRecord::new(5, "later kuku", ["search", "architecture"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(4_000),
            ],
        )
        .unwrap();
    let after_invalidation = memory.search_graph_view_with_metadata_plan(
        &query,
        10,
        &[1],
        3,
        &["architecture", "search"],
        &["kuku.mom"],
        Some(1_500),
        Some(2_500),
    );
    assert!(!after_invalidation.plan.combined_cache_hit);
    assert!(!after_invalidation.plan.graph_cache_hit);
    assert_eq!(after_invalidation.hits[0].id, 2);
}

#[test]
fn metadata_batch_search_reuses_one_compiled_view_for_many_queries() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let queries = normalized_vectors(3, dim, 0x6A9A_2016);

    let batch = memory.search_graph_view_with_metadata_batch_plan(
        &queries,
        2,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
    );

    assert!(!batch.plan.combined_cache_hit);
    assert_eq!(batch.plan.total_slots, 6);
    assert_eq!(batch.plan.graph_slots, 5);
    assert_eq!(batch.plan.selected_slots, 2);
    assert_eq!(batch.hits.len(), 3);
    assert!(batch.hits.iter().all(|row| row.len() == 2));

    for qi in 0..3 {
        let single = memory.search_graph_view_with_metadata_plan(
            vec_at(&queries, dim, qi),
            2,
            &[10],
            2,
            &["architecture"],
            &[],
            None,
            None,
        );
        assert!(single.plan.combined_cache_hit);
        assert_eq!(single.plan.selected_slots, batch.plan.selected_slots);
        assert_eq!(single.hits.len(), batch.hits[qi].len());
        for (batched, single) in batch.hits[qi].iter().zip(single.hits.iter()) {
            assert_eq!(batched.id, single.id);
            assert!(
                (batched.score - single.score).abs()
                    <= 1e-4 * batched.score.abs().max(single.score.abs()).max(1.0),
                "query {qi}: batch score {} != single score {}",
                batched.score,
                single.score
            );
        }
    }
}

#[test]
fn policy_rerank_batch_reuses_view_and_matches_single_query_rows() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let queries = normalized_vectors(3, dim, 0x6A9A_201A);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphRerankConfig::new(0.7, 0.3)
        .with_prefetch_factor(1)
        .with_min_prefetch(2);

    let batch = memory.search_graph_view_with_policy_metadata_rerank_batch(
        &queries,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        rerank,
    );

    assert!(!batch.plan.combined_cache_hit);
    assert_eq!(batch.plan.selected_slots, 2);
    assert_eq!(batch.prefetch_k, 2);
    assert_eq!(batch.hits.len(), 3);
    assert!(batch.hits.iter().all(|row| row.len() == 2));

    for qi in 0..3 {
        let single = memory.search_graph_view_with_policy_metadata_rerank(
            vec_at(&queries, dim, qi),
            2,
            &[10],
            policy,
            &["architecture"],
            &[],
            None,
            None,
            rerank,
        );
        assert!(single.plan.combined_cache_hit);
        assert_eq!(single.plan.selected_slots, batch.plan.selected_slots);
        assert_eq!(single.prefetch_k, batch.prefetch_k);
        assert_eq!(single.hits.len(), batch.hits[qi].len());
        for (batched, single) in batch.hits[qi].iter().zip(single.hits.iter()) {
            assert_eq!(batched.id, single.id);
            assert_eq!(batched.depth, single.depth);
            assert_eq!(batched.parent, single.parent);
            assert!((batched.graph_score - single.graph_score).abs() < 1e-6);
            assert!(
                (batched.score - single.score).abs()
                    <= 1e-4 * batched.score.abs().max(single.score.abs()).max(1.0),
                "query {qi}: batch rerank score {} != single score {}",
                batched.score,
                single.score
            );
        }
    }
}

#[test]
fn candidate_policy_rerank_batch_matches_single_query_rows() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let queries = normalized_vectors(3, dim, 0x6A9A_2022);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphRerankConfig::new(0.7, 0.3)
        .with_prefetch_factor(1)
        .with_min_prefetch(2);
    let candidate_ids = [20, 30, 40, 999, 30];

    let batch = memory.search_graph_view_with_policy_metadata_candidates_rerank_batch(
        &queries,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &candidate_ids,
        rerank,
    );

    assert!(!batch.plan.combined_cache_hit);
    assert_eq!(batch.plan.candidate_input_ids, 5);
    assert_eq!(batch.plan.candidate_slots, 3);
    assert_eq!(batch.plan.candidate_missing_ids, 1);
    assert_eq!(batch.plan.candidate_duplicate_ids, 1);
    assert_eq!(batch.plan.selected_slots, 2);
    assert_eq!(batch.prefetch_k, 2);
    assert_eq!(batch.hits.len(), 3);
    assert!(batch.hits.iter().all(|row| row.len() == 2));

    for qi in 0..3 {
        let single = memory.search_graph_view_with_policy_metadata_candidates_rerank(
            vec_at(&queries, dim, qi),
            2,
            &[10],
            policy,
            &["architecture"],
            &[],
            None,
            None,
            &candidate_ids,
            rerank,
        );
        assert!(single.plan.combined_cache_hit);
        assert_eq!(single.plan.selected_slots, batch.plan.selected_slots);
        assert_eq!(single.prefetch_k, batch.prefetch_k);
        assert_eq!(single.hits.len(), batch.hits[qi].len());
        for (batched, single) in batch.hits[qi].iter().zip(single.hits.iter()) {
            assert_eq!(batched.id, single.id);
            assert_eq!(batched.depth, single.depth);
            assert_eq!(batched.parent, single.parent);
            assert!((batched.graph_score - single.graph_score).abs() < 1e-6);
            assert!(
                (batched.score - single.score).abs()
                    <= 1e-4 * batched.score.abs().max(single.score.abs()).max(1.0),
                "query {qi}: candidate batch score {} != single score {}",
                batched.score,
                single.score
            );
        }
    }
}

#[test]
fn candidate_id_masks_intersect_graph_metadata_views_without_new_cache_key() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2017);

    let report = memory.search_graph_view_with_metadata_candidates_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
        &[30, 40, 60, 999, 30],
    );
    assert!(!report.plan.combined_cache_hit);
    assert_eq!(report.plan.total_slots, 6);
    assert_eq!(report.plan.graph_slots, 5);
    assert_eq!(report.plan.metadata_slots, 2);
    assert_eq!(report.plan.candidate_input_ids, 5);
    assert_eq!(report.plan.candidate_slots, 3);
    assert_eq!(report.plan.candidate_missing_ids, 1);
    assert_eq!(report.plan.candidate_duplicate_ids, 1);
    assert_eq!(report.plan.selected_slots, 1);
    assert_eq!(report.plan.active_blocks, 1);
    assert_eq!(report.plan.view_stats().selected_slots, 1);
    assert_eq!(report.plan.selectivity(), 1.0 / 6.0);
    assert_eq!(report.plan.graph_selectivity(), 5.0 / 6.0);
    assert_eq!(report.plan.metadata_selectivity(), 2.0 / 6.0);
    assert_eq!(report.plan.candidate_selectivity(), 3.0 / 6.0);
    assert_eq!(report.plan.candidate_live_ratio(), 3.0 / 5.0);
    assert_eq!(report.plan.candidate_missing_ratio(), 1.0 / 5.0);
    assert_eq!(report.plan.candidate_duplicate_ratio(), 1.0 / 5.0);
    assert_eq!(report.hits.len(), 1);
    assert_eq!(report.hits[0].id, 30);

    let second = memory.search_graph_view_with_metadata_candidates_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
        &[20],
    );
    assert!(
        second.plan.combined_cache_hit,
        "candidate ids should reuse the cached graph+metadata base view"
    );
    assert!(second.plan.graph_cache_hit);
    assert_eq!(second.plan.candidate_input_ids, 1);
    assert_eq!(second.plan.candidate_slots, 1);
    assert_eq!(second.plan.candidate_missing_ids, 0);
    assert_eq!(second.plan.candidate_duplicate_ids, 0);
    assert_eq!(second.plan.selected_slots, 1);
    assert_eq!(second.hits[0].id, 20);

    let empty = memory.search_graph_view_with_metadata_candidates_plan(
        &query,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
        &[],
    );
    assert_eq!(empty.plan.candidate_input_ids, 0);
    assert_eq!(empty.plan.candidate_slots, 0);
    assert_eq!(empty.plan.candidate_missing_ids, 0);
    assert_eq!(empty.plan.candidate_duplicate_ids, 0);
    assert_eq!(empty.plan.selected_slots, 0);
    assert!(empty.hits.is_empty());

    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let policy_report = memory.search_graph_view_with_policy_metadata_candidates_plan(
        &query,
        10,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[20, 40],
    );
    assert_eq!(policy_report.plan.metadata_slots, 2);
    assert_eq!(policy_report.plan.candidate_input_ids, 2);
    assert_eq!(policy_report.plan.candidate_slots, 2);
    assert_eq!(policy_report.plan.selected_slots, 1);
    assert_eq!(policy_report.hits[0].id, 20);

    let batch_queries = [
        query.clone(),
        normalized_vectors(1, memory.dim(), 0x6A9A_2018),
    ]
    .concat();
    let batch = memory.search_graph_view_with_metadata_candidates_batch_plan(
        &batch_queries,
        10,
        &[10],
        2,
        &["architecture"],
        &[],
        None,
        None,
        &[20, 30, 40, 999, 30],
    );
    assert!(batch.plan.combined_cache_hit);
    assert_eq!(batch.plan.candidate_input_ids, 5);
    assert_eq!(batch.plan.candidate_slots, 3);
    assert_eq!(batch.plan.candidate_missing_ids, 1);
    assert_eq!(batch.plan.candidate_duplicate_ids, 1);
    assert_eq!(batch.plan.selected_slots, 2);
    assert_eq!(batch.hits.len(), 2);
    assert_eq!(batch.hits[0].len(), 2);
    assert_eq!(batch.hits[1].len(), 2);

    for (row_idx, query_row) in batch_queries.chunks(memory.dim()).enumerate() {
        let single = memory.search_graph_view_with_metadata_candidates_plan(
            query_row,
            10,
            &[10],
            2,
            &["architecture"],
            &[],
            None,
            None,
            &[20, 30, 40, 999, 30],
        );
        assert_eq!(
            batch.hits[row_idx]
                .iter()
                .map(|hit| hit.id)
                .collect::<Vec<_>>(),
            single.hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
        );
    }
}

#[test]
fn candidate_policy_rerank_blends_and_reports_timing() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2018);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphRerankConfig::new(0.0, 1.0)
        .with_prefetch_factor(1)
        .with_min_prefetch(1);

    let report = memory.search_graph_view_with_policy_metadata_candidates_rerank(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[20, 30, 40],
        rerank,
    );
    assert_eq!(report.plan.metadata_slots, 2);
    assert_eq!(report.plan.candidate_slots, 3);
    assert_eq!(report.plan.selected_slots, 2);
    assert_eq!(report.prefetch_k, 2);
    assert_eq!(
        report.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![20, 30]
    );
    assert!((report.hits[0].graph_score - 1.0).abs() < 1e-6);
    assert!((report.hits[1].graph_score - 0.9).abs() < 1e-6);
    assert!(report.hits[0].score > report.hits[1].score);

    let timed = memory.search_graph_view_with_policy_metadata_candidates_rerank_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[30],
        rerank,
    );
    assert!(timed.plan.combined_cache_hit);
    assert!(timed.plan.graph_cache_hit);
    assert_eq!(timed.plan.metadata_slots, 2);
    assert_eq!(timed.plan.candidate_slots, 1);
    assert_eq!(timed.plan.selected_slots, 1);
    assert_eq!(timed.prefetch_k, 1);
    assert_eq!(timed.hits.len(), 1);
    assert_eq!(timed.hits[0].id, 30);
    assert!(timed.telemetry.total_ns >= timed.telemetry.view_build_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.vector_search_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.rerank_ns);

    let empty = memory.search_graph_view_with_policy_metadata_candidates_rerank_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[],
        rerank,
    );
    assert_eq!(empty.plan.selected_slots, 0);
    assert_eq!(empty.prefetch_k, 0);
    assert!(empty.hits.is_empty());
    assert_eq!(empty.telemetry.vector_search_ns, 0);
    assert_eq!(empty.telemetry.rerank_ns, 0);

    let explained = memory.explain_graph_search_with_policy_metadata_candidates_rerank_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[20, 30],
        rerank,
    );
    assert_eq!(explained.plan.metadata_slots, 2);
    assert_eq!(explained.plan.candidate_slots, 2);
    assert_eq!(explained.plan.selected_slots, 2);
    assert_eq!(explained.prefetch_k, 2);
    assert_eq!(explained.hits.len(), 2);
    assert_eq!(explained.trace.nodes.len(), 4);
    assert!(explained.telemetry.trace_build_ns <= explained.telemetry.total_ns);

    let snapshot = explained.debug_snapshot();
    assert_eq!(snapshot.summary.metadata_slots, 2);
    assert_eq!(snapshot.summary.candidate_input_ids, 2);
    assert_eq!(snapshot.summary.candidate_slots, 2);
    assert_eq!(snapshot.summary.candidate_missing_ids, 0);
    assert_eq!(snapshot.summary.candidate_duplicate_ids, 0);
    assert_eq!(snapshot.summary.selected_slots, 2);
    assert_eq!(snapshot.summary.hit_count, 2);
    assert_eq!(
        snapshot.summary.trace_node_count,
        explained.trace.nodes.len()
    );
    assert_eq!(snapshot.hits[0].rank, 1);
    assert!(snapshot.nodes.iter().any(|node| node.hit_rank == Some(1)));
    assert!(snapshot
        .nodes
        .iter()
        .any(|node| node.id == 10 && node.hit_rank.is_none()));
}

#[test]
fn candidate_score_hybrid_rerank_uses_external_scores() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_2019);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphHybridRerankConfig::new(0.0, 0.0, 1.0)
        .with_prefetch_factor(1)
        .with_min_prefetch(1);

    let report = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, 0.2), (30, 0.7), (30, 1.0), (40, 10.0), (999, 100.0)],
        rerank,
    );
    assert_eq!(report.plan.metadata_slots, 2);
    assert_eq!(report.plan.candidate_input_ids, 5);
    assert_eq!(report.plan.candidate_slots, 3);
    assert_eq!(report.plan.candidate_missing_ids, 1);
    assert_eq!(report.plan.candidate_duplicate_ids, 1);
    assert_eq!(report.plan.selected_slots, 2);
    assert_eq!(report.prefetch_k, 2);
    assert_eq!(
        report.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![30, 20]
    );
    assert_eq!(report.hits[0].candidate_score, 1.0);
    assert_eq!(report.hits[1].candidate_score, 0.2);
    assert!(report.hits[0].score > report.hits[1].score);

    let minmax = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, 10.0), (30, 20.0), (30, 15.0), (40, 1_000.0)],
        rerank.with_candidate_score_normalization(GraphCandidateScoreNormalization::MinMax),
    );
    assert_eq!(minmax.plan.candidate_slots, 3);
    assert_eq!(minmax.plan.selected_slots, 2);
    assert_eq!(
        minmax.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![30, 20]
    );
    assert_eq!(minmax.hits[0].candidate_score, 1.0);
    assert_eq!(minmax.hits[1].candidate_score, 0.0);

    let max_abs = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, -5.0), (30, 10.0), (40, -1_000.0)],
        rerank.with_candidate_score_normalization(GraphCandidateScoreNormalization::MaxAbs),
    );
    assert_eq!(max_abs.plan.candidate_slots, 3);
    assert_eq!(max_abs.plan.selected_slots, 2);
    assert_eq!(max_abs.hits[0].id, 30);
    assert_eq!(max_abs.hits[0].candidate_score, 1.0);
    assert_eq!(max_abs.hits[1].candidate_score, -0.5);

    let constant = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, 5.0), (30, 5.0)],
        rerank.with_candidate_score_normalization(GraphCandidateScoreNormalization::MinMax),
    );
    assert!(constant.hits.iter().all(|hit| hit.candidate_score == 1.0));

    let timed = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, 0.2), (30, 1.0)],
        GraphHybridRerankConfig::from_graph_rerank(
            GraphRerankConfig::new(0.0, 0.0)
                .with_prefetch_factor(1)
                .with_min_prefetch(1),
            1.0,
        ),
    );
    assert!(timed.plan.combined_cache_hit);
    assert_eq!(timed.plan.candidate_input_ids, 2);
    assert_eq!(timed.plan.candidate_slots, 2);
    assert_eq!(timed.plan.candidate_missing_ids, 0);
    assert_eq!(timed.plan.candidate_duplicate_ids, 0);
    assert_eq!(timed.plan.selected_slots, 2);
    assert_eq!(timed.hits[0].id, 30);
    assert!(timed.telemetry.total_ns >= timed.telemetry.view_build_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.vector_search_ns);
    assert!(timed.telemetry.total_ns >= timed.telemetry.rerank_ns);

    let empty = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[],
        rerank,
    );
    assert_eq!(empty.plan.candidate_input_ids, 0);
    assert_eq!(empty.plan.candidate_slots, 0);
    assert_eq!(empty.plan.candidate_missing_ids, 0);
    assert_eq!(empty.plan.candidate_duplicate_ids, 0);
    assert_eq!(empty.plan.selected_slots, 0);
    assert_eq!(empty.prefetch_k, 0);
    assert!(empty.hits.is_empty());
    assert_eq!(empty.telemetry.vector_search_ns, 0);
    assert_eq!(empty.telemetry.rerank_ns, 0);

    let explained = memory.explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed(
        &query,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &[(20, 0.2), (30, 1.0)],
        rerank,
    );
    assert_eq!(explained.plan.candidate_input_ids, 2);
    assert_eq!(explained.plan.candidate_slots, 2);
    assert_eq!(explained.plan.selected_slots, 2);
    assert_eq!(explained.hits[0].id, 30);
    assert_eq!(explained.hits[0].candidate_score, 1.0);
    assert!(explained.telemetry.trace_build_ns <= explained.telemetry.total_ns);

    let snapshot = explained.debug_snapshot();
    assert_eq!(snapshot.summary.candidate_slots, 2);
    assert_eq!(snapshot.summary.selected_slots, 2);
    assert_eq!(snapshot.hits[0].candidate_score, 1.0);
    let hit_node = snapshot
        .nodes
        .iter()
        .find(|node| node.hit_rank == Some(1))
        .unwrap();
    assert_eq!(hit_node.candidate_score, Some(1.0));
    assert!(snapshot
        .nodes
        .iter()
        .any(|node| node.id == 10 && node.candidate_score.is_none()));
}

#[test]
fn candidate_score_hybrid_batch_matches_single_query_rows() {
    let mut memory = build_memory();
    let dim = memory.dim();
    let queries = normalized_vectors(3, dim, 0x6A9A_2021);
    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(4)
        .with_min_path_weight(0.5);
    let rerank = GraphHybridRerankConfig::new(0.6, 0.2, 0.4)
        .with_prefetch_factor(1)
        .with_min_prefetch(2)
        .with_candidate_score_normalization(GraphCandidateScoreNormalization::MinMax);
    let candidate_scores = [
        (20, 10.0),
        (30, 20.0),
        (30, 15.0),
        (40, 1_000.0),
        (999, 5.0),
    ];

    let batch = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid_batch(
        &queries,
        2,
        &[10],
        policy,
        &["architecture"],
        &[],
        None,
        None,
        &candidate_scores,
        rerank,
    );

    assert!(!batch.plan.combined_cache_hit);
    assert_eq!(batch.plan.candidate_input_ids, 5);
    assert_eq!(batch.plan.candidate_slots, 3);
    assert_eq!(batch.plan.candidate_missing_ids, 1);
    assert_eq!(batch.plan.candidate_duplicate_ids, 1);
    assert_eq!(batch.plan.selected_slots, 2);
    assert_eq!(batch.prefetch_k, 2);
    assert_eq!(batch.hits.len(), 3);
    assert!(batch.hits.iter().all(|row| row.len() == 2));

    for qi in 0..3 {
        let single = memory.search_graph_view_with_policy_metadata_candidate_scores_hybrid(
            vec_at(&queries, dim, qi),
            2,
            &[10],
            policy,
            &["architecture"],
            &[],
            None,
            None,
            &candidate_scores,
            rerank,
        );
        assert!(single.plan.combined_cache_hit);
        assert_eq!(single.plan.selected_slots, batch.plan.selected_slots);
        assert_eq!(single.prefetch_k, batch.prefetch_k);
        assert_eq!(single.hits.len(), batch.hits[qi].len());
        for (batched, single) in batch.hits[qi].iter().zip(single.hits.iter()) {
            assert_eq!(batched.id, single.id);
            assert_eq!(batched.depth, single.depth);
            assert_eq!(batched.parent, single.parent);
            assert!((batched.vector_score - single.vector_score).abs() < 1e-4);
            assert!((batched.graph_score - single.graph_score).abs() < 1e-6);
            assert!((batched.candidate_score - single.candidate_score).abs() < 1e-6);
            assert!(
                (batched.score - single.score).abs()
                    <= 1e-4 * batched.score.abs().max(single.score.abs()).max(1.0),
                "query {qi}: batch hybrid score {} != single score {}",
                batched.score,
                single.score
            );
        }
    }
}

#[test]
fn cache_stats_trim_and_clear_operational_caches() {
    let dim = 64;
    let n = 96;
    let data = normalized_vectors(n, dim, 0x6A9A_2014);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| {
            let mut record = if i % 2 == 0 {
                MemoryRecord::new(
                    i as u64,
                    format!("architecture node {i}"),
                    ["doc", "architecture"],
                )
            } else {
                MemoryRecord::new(i as u64, format!("web node {i}"), ["doc", "web"])
            };
            record = record.with_source(if i % 3 == 0 { "kuku.mom" } else { "liner" });
            record.with_timestamp_ms(1_000 + i as i64 * 100)
        })
        .collect();
    memory.add_records(&data, records).unwrap();
    memory.link_directed(0, 1, 0.9).unwrap();
    memory.link_directed(0, 2, 0.8).unwrap();
    memory.link_directed(1, 3, 0.7).unwrap();
    memory.link_directed(0, 40, 0.95).unwrap();
    memory.link_directed(40, 41, 0.9).unwrap();

    assert!(memory.cache_stats().is_empty());

    let policy = GraphViewPolicy::new(2)
        .with_max_nodes(6)
        .with_max_active_blocks(2);
    let query = vec_at(&data, dim, 0);

    memory.graph_view_mask_with_stats(&[0], 1);
    memory.graph_view_mask_with_stats(&[0], 2);
    memory.graph_view_mask_with_stats(&[40], 1);
    memory.graph_view_mask_with_policy_stats(&[0], policy);
    memory.graph_view_mask_with_policy_stats(&[40], policy);
    memory.tag_view_mask("doc");
    memory.tag_view_mask("architecture");
    memory.source_view_mask("kuku.mom");
    memory.source_view_mask("liner");
    memory.time_range_view_mask(Some(1_000), Some(2_000));
    memory.time_range_view_mask(Some(2_000), Some(4_000));
    memory.search_graph_view_with_metadata_plan(
        query,
        3,
        &[0],
        2,
        &["doc"],
        &["kuku.mom"],
        Some(1_000),
        Some(4_000),
    );
    memory.search_graph_view_with_metadata_plan(
        query,
        3,
        &[40],
        1,
        &["architecture"],
        &["liner"],
        Some(2_000),
        Some(6_000),
    );
    memory.search_graph_view_with_policy_metadata_plan(
        query,
        3,
        &[0],
        policy,
        &["doc"],
        &["kuku.mom"],
        Some(1_000),
        Some(4_000),
    );
    memory.search_graph_view_with_policy_metadata_plan(
        query,
        3,
        &[40],
        policy,
        &["architecture"],
        &["liner"],
        Some(2_000),
        Some(6_000),
    );

    let stats = memory.cache_stats();
    assert!(stats.graph_views >= 3);
    assert!(stats.policy_visits >= 2);
    assert!(stats.policy_views >= 2);
    assert!(stats.combined_views >= 2);
    assert!(stats.combined_policy_views >= 2);
    assert!(stats.tag_masks >= 2);
    assert!(stats.source_masks >= 2);
    assert!(stats.time_masks >= 2);
    assert_eq!(
        stats.total_entries,
        stats.query_entries() + stats.metadata_entries()
    );

    memory.trim_all_caches(1);
    let trimmed = memory.cache_stats();
    assert!(trimmed.graph_views <= 1);
    assert!(trimmed.policy_visits <= 1);
    assert!(trimmed.policy_views <= 1);
    assert!(trimmed.combined_views <= 1);
    assert!(trimmed.combined_policy_views <= 1);
    assert!(trimmed.tag_masks <= 1);
    assert!(trimmed.source_masks <= 1);
    assert!(trimmed.time_masks <= 1);
    assert!(!trimmed.is_empty());
    assert_eq!(
        trimmed.total_entries,
        trimmed.query_entries() + trimmed.metadata_entries()
    );

    memory.clear_query_caches();
    let metadata_only = memory.cache_stats();
    assert_eq!(metadata_only.query_entries(), 0);
    assert!(metadata_only.metadata_entries() > 0);

    memory.clear_metadata_caches();
    assert!(memory.cache_stats().is_empty());
}

#[test]
fn cache_trimming_is_deterministic_for_metadata_masks() {
    let dim = 64;
    let n = 6;
    let data = normalized_vectors(n, dim, 0x6A9A_2019);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| {
            let source = match i % 3 {
                0 => "alpha",
                1 => "beta",
                _ => "gamma",
            };
            MemoryRecord::new(i as u64, format!("trim node {i}"), ["doc"]).with_source(source)
        })
        .collect();
    memory.add_records(&data, records).unwrap();

    memory.source_view_mask("alpha");
    memory.source_view_mask("beta");
    memory.source_view_mask("gamma");
    assert_eq!(memory.cache_stats().source_masks, 3);

    memory.trim_metadata_caches(1);
    let trimmed = memory.cache_stats();
    assert_eq!(trimmed.source_masks, 1);

    let before_gamma = memory.cache_stats();
    memory.source_view_mask("gamma");
    let after_gamma = memory.cache_stats();
    assert_eq!(
        after_gamma.source_mask_hits,
        before_gamma.source_mask_hits + 1,
        "deterministic trim should retain the lexicographically last source key"
    );

    memory.source_view_mask("alpha");
    let after_alpha = memory.cache_stats();
    assert_eq!(
        after_alpha.source_mask_misses,
        after_gamma.source_mask_misses + 1
    );
}

#[test]
fn preset_cache_budgets_scale_and_trim_individual_caches() {
    let dim = 64;
    let n = 256;
    let data = normalized_vectors(n, dim, 0x6A9A_2015);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| {
            MemoryRecord::new(i as u64, format!("cache node {i}"), ["doc"])
                .with_source(if i % 2 == 0 { "kuku.mom" } else { "liner" })
                .with_timestamp_ms(i as i64)
        })
        .collect();
    memory.add_records(&data, records).unwrap();
    for i in 0..(n - 1) {
        memory.link_directed(i as u64, (i + 1) as u64, 1.0).unwrap();
    }

    let low = memory.cache_budget_for_preset(GraphSearchPreset::low_latency());
    let balanced = memory.cache_budget_for_preset(GraphSearchPreset::balanced());
    let broad = memory.cache_budget_for_preset(GraphSearchPreset::broad());
    assert!(low.query_entries() < balanced.query_entries());
    assert!(balanced.query_entries() < broad.query_entries());
    assert!(low.metadata_entries() < balanced.metadata_entries());
    assert!(balanced.metadata_entries() < broad.metadata_entries());

    let query = vec_at(&data, dim, 0);
    for seed in 0..4u64 {
        memory.graph_view_mask_with_stats(&[seed], 2);
        memory.graph_view_mask_with_policy_stats(
            &[seed],
            GraphViewPolicy::new(2)
                .with_max_nodes(8)
                .with_max_active_blocks(2),
        );
        memory.search_graph_view_with_metadata_plan(
            query,
            3,
            &[seed],
            2,
            &["doc"],
            if seed % 2 == 0 {
                &["kuku.mom"]
            } else {
                &["liner"]
            },
            Some(seed as i64),
            Some(seed as i64 + 64),
        );
        memory.search_graph_view_with_policy_metadata_plan(
            query,
            3,
            &[seed],
            GraphViewPolicy::new(2)
                .with_max_nodes(8)
                .with_max_active_blocks(2),
            &["doc"],
            if seed % 2 == 0 {
                &["kuku.mom"]
            } else {
                &["liner"]
            },
            Some(seed as i64),
            Some(seed as i64 + 64),
        );
    }
    assert!(memory.cache_stats().total_entries > 0);

    let tight = turbo_graph::GraphMemoryCacheBudget::fixed(1);
    assert_eq!(tight.total_entries(), 8);
    memory.trim_caches_to_budget(tight);
    let trimmed = memory.cache_stats();
    assert!(trimmed.graph_views <= tight.graph_views);
    assert!(trimmed.policy_visits <= tight.policy_visits);
    assert!(trimmed.policy_views <= tight.policy_views);
    assert!(trimmed.combined_views <= tight.combined_views);
    assert!(trimmed.combined_policy_views <= tight.combined_policy_views);
    assert!(trimmed.tag_masks <= tight.tag_masks);
    assert!(trimmed.source_masks <= tight.source_masks);
    assert!(trimmed.time_masks <= tight.time_masks);

    let applied = memory.trim_caches_for_preset(GraphSearchPreset::low_latency());
    assert_eq!(applied, low);
}

#[test]
fn remove_node_repairs_slot_tables_after_swap_remove() {
    let mut memory = build_memory();
    let moved_before = memory.slot_of(60).unwrap();
    assert_eq!(moved_before, 5);

    assert!(memory.remove_node(20));

    assert!(!memory.contains(20));
    assert_eq!(memory.len(), 5);
    assert_eq!(memory.slot_of(60), Some(1));
    assert!(!memory.neighbors(10).iter().any(|edge| edge.to == 20));
    assert!(!memory.neighbors(30).iter().any(|edge| edge.to == 20));

    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1002);
    let hits = memory.search_global(&query, 10);
    assert!(hits.iter().all(|hit| hit.id != 20));
}

#[test]
fn replace_embedding_preserves_edges_and_slot() {
    let mut memory = build_memory();
    let slot_before = memory.slot_of(30).unwrap();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1003);

    memory.replace_embedding(30, &query).unwrap();

    assert_eq!(memory.slot_of(30), Some(slot_before));
    assert!(memory.neighbors(30).iter().any(|edge| edge.to == 20));
    let hits = memory.search_graph_view(&query, 3, &[10], 2, &["architecture"]);
    assert_eq!(hits.first().map(|hit| hit.id), Some(30));
}

#[test]
fn write_load_roundtrips_graph_sidecar() {
    let mut memory = build_memory();
    let query = normalized_vectors(1, memory.dim(), 0x6A9A_1004);
    let before = memory.search_graph_view(&query, 10, &[10], 2, &["architecture"]);
    let (index_path, graph_path) = temp_paths("graph_roundtrip");

    memory.write(&index_path, &graph_path).unwrap();
    let mut loaded = GraphMemoryIndex::load(&index_path, &graph_path).unwrap();
    let after = loaded.search_graph_view(&query, 10, &[10], 2, &["architecture"]);

    fs::remove_file(&index_path).ok();
    fs::remove_file(&graph_path).ok();

    assert_eq!(loaded.len(), memory.len());
    assert_eq!(loaded.record(20).unwrap().title, "architecture note");
    assert_eq!(loaded.neighbors(20).len(), 3);
    assert_eq!(loaded.tag_view_mask("architecture").count(), 3);
    assert_eq!(
        before.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        after.iter().map(|hit| hit.id).collect::<Vec<_>>()
    );
}

#[test]
fn duplicate_and_missing_ids_return_errors() {
    let dim = 64;
    let data = normalized_vectors(1, dim, 0x6A9A_1005);
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    memory.add_node(1, "one", &data, ["a"]).unwrap();

    let duplicate = memory.add_node(1, "dupe", &data, ["a"]).unwrap_err();
    assert!(matches!(
        duplicate,
        GraphMemoryError::Add(turbo_graph::AddError::IdAlreadyPresent(1))
    ));

    let missing = memory.link_bidirectional(1, 99, 1.0).unwrap_err();
    assert!(matches!(missing, GraphMemoryError::MissingId(99)));
}
