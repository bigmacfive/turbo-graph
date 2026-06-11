//! Graph-view local memory layer on top of TurboQuant.
//!
//! Run:
//!   cargo run -p turbo-graph --example graph_memory_layer

use turbo_graph::{
    GraphMemoryIndex, GraphRerankConfig, GraphRerankedHit, GraphSearchPreset, MemoryHit,
    MemoryRecord,
};

fn toy_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut vector = vec![0.0f32; dim];
    for x in &mut vector {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        *x = unit * 2.0 - 1.0;
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut vector {
        *x /= norm.max(1e-6);
    }
    vector
}

fn print_hits(title: &str, hits: &[MemoryHit]) {
    println!("{title}");
    for (rank, hit) in hits.iter().enumerate() {
        println!(
            "  rank={:>2} id={} score={:.4} title={}",
            rank + 1,
            hit.id,
            hit.score,
            hit.title,
        );
    }
}

fn print_reranked_hits(title: &str, hits: &[GraphRerankedHit]) {
    println!("{title}");
    for (rank, hit) in hits.iter().enumerate() {
        println!(
            "  rank={:>2} id={} score={:.4} vec={:.4} graph={:.4} depth={} parent={:?} title={}",
            rank + 1,
            hit.id,
            hit.score,
            hit.vector_score,
            hit.graph_score,
            hit.depth,
            hit.parent,
            hit.title,
        );
    }
}

fn main() {
    let dim = 64;
    let mut memory = GraphMemoryIndex::new(dim, 4).expect("valid graph memory config");

    let embeddings: Vec<f32> = [10, 20, 30, 40, 50]
        .into_iter()
        .flat_map(|seed| toy_embedding(dim, seed))
        .collect();
    memory
        .add_records(
            &embeddings,
            vec![
                MemoryRecord::new(10, "kuku.mom search product brief", ["product", "search"])
                    .with_source("kuku.mom")
                    .with_timestamp_ms(1_700_000_000_000),
                MemoryRecord::new(
                    20,
                    "local context memory architecture",
                    ["architecture", "memory"],
                )
                .with_source("kuku.mom")
                .with_timestamp_ms(1_700_100_000_000),
                MemoryRecord::new(
                    30,
                    "TurboQuant compressed vector index",
                    ["architecture", "vector"],
                )
                .with_source("kuku.mom")
                .with_timestamp_ms(1_700_200_000_000),
                MemoryRecord::new(40, "graph view candidate pruning", ["graph", "memory"])
                    .with_source("liner")
                    .with_timestamp_ms(1_700_300_000_000),
                MemoryRecord::new(50, "external web ranking notes", ["ranking", "web"])
                    .with_source("web")
                    .with_timestamp_ms(1_700_400_000_000),
            ],
        )
        .unwrap();

    memory.link_bidirectional(10, 20, 1.0).unwrap();
    memory.link_bidirectional(20, 30, 0.9).unwrap();
    memory.link_bidirectional(20, 40, 0.8).unwrap();
    memory.link_bidirectional(10, 50, 0.4).unwrap();

    let preset = GraphSearchPreset::balanced()
        .with_target_active_blocks(1)
        .with_nodes_per_active_block(4)
        .with_min_path_weight(0.5)
        .with_min_prefetch(8);
    let tuning = memory.tuning_for_preset(3, preset);

    let query = toy_embedding(dim, 42);
    let batch_queries = [43, 44, 45]
        .into_iter()
        .flat_map(|seed| toy_embedding(dim, seed))
        .collect::<Vec<_>>();

    let prepared_view = memory.prepare_graph_view(&[10], 2);
    println!(
        "prepared view: total_slots={} selected_slots={} active_blocks={}",
        prepared_view.plan.total_slots,
        prepared_view.plan.selected_slots,
        prepared_view.plan.active_blocks
    );
    let prepared_hits = prepared_view.search(&memory, &query, 3);
    print_hits("prepared search (graph only):", &prepared_hits);

    let prepared_batch = prepared_view.search_batch(&memory, &batch_queries, 2);
    println!(
        "prepared batch (graph only): {} query rows",
        prepared_batch.len()
    );
    for (qi, row) in prepared_batch.iter().enumerate() {
        println!(
            "  batch {qi} -> {:?}",
            row.iter().map(|hit| hit.id).collect::<Vec<_>>()
        );
    }

    let prepared_policy = memory.prepare_graph_view_with_policy_metadata(
        &[10],
        tuning.policy,
        &["architecture"],
        &["kuku.mom"],
        Some(1_700_000_000_000),
        Some(1_700_300_000_000),
    );
    let policy_single = prepared_policy.search_rerank(&memory, &query, 3, tuning.rerank);
    print_reranked_hits(
        "prepared policy+metadata rerank (single):",
        &policy_single.hits,
    );

    let policy_batch =
        prepared_policy.search_rerank_batch(&memory, &batch_queries, 3, tuning.rerank);
    println!(
        "prepared policy+metadata rerank batch: rows={} prefetch_k={}",
        policy_batch.hits.len(),
        policy_batch.prefetch_k,
    );
    for (qi, row) in policy_batch.hits.iter().enumerate() {
        println!(
            "  batch {qi} -> {:?}",
            row.iter().map(|hit| hit.id).collect::<Vec<_>>()
        );
    }

    let legacy = memory.search_graph_view_with_policy_metadata_rerank(
        &query,
        3,
        &[10],
        tuning.policy,
        &["architecture"],
        &["kuku.mom"],
        Some(1_700_000_000_000),
        Some(1_700_300_000_000),
        tuning.rerank,
    );
    println!(
        "legacy rerank first ids: {:?}",
        legacy.hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
    );

    let candidate = memory.search_graph_view_with_policy_metadata_candidates_rerank_timed(
        &query,
        3,
        &[10],
        tuning.policy,
        &["architecture"],
        &["kuku.mom"],
        Some(1_700_000_000_000),
        Some(1_700_300_000_000),
        &[20, 30],
        GraphRerankConfig::new(1.0, 0.2)
            .with_prefetch_factor(1)
            .with_min_prefetch(2),
    );
    println!(
        "candidate rerank first ids: {:?}",
        candidate.hits.iter().map(|hit| hit.id).collect::<Vec<_>>()
    );

    let cache = memory.cache_stats();
    println!("cache entries={}", cache.total_entries);
    println!(
        "cache ratio query={} metadata={} overall={}",
        cache.query_cache_hit_ratio(),
        cache.metadata_cache_hit_ratio(),
        cache.cache_hit_ratio(),
    );
}
