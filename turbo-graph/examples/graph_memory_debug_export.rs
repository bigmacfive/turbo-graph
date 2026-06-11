//! Export a graph-memory debug snapshot as JSON.
//!
//! Run:
//!   cargo run -p turbo-graph --features serde --example graph_memory_debug_export
//!   cargo run -p turbo-graph --features serde --example graph_memory_debug_export -- \
//!     --nodes 5000 --avg-degree 6 --output /tmp/graph-memory-snapshot.json

#[cfg(feature = "serde")]
use std::{env, fs, path::PathBuf};

#[cfg(feature = "serde")]
use turbo_graph::{
    GraphCandidateScoreNormalization, GraphCandidateSearchDebugSummary, GraphHybridRerankConfig,
    GraphMemoryIndex, GraphViewPolicy, MemoryRecord,
};

#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
enum ExportMode {
    Search,
    RawGraph,
}

#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
struct SyntheticConfig {
    nodes: usize,
    max_trace_nodes: usize,
    avg_degree: usize,
    dim: usize,
    seed: u64,
    prefetch: usize,
    top_k: usize,
    seed_count: usize,
    max_hops: usize,
    min_path_weight: f32,
}

#[cfg(feature = "serde")]
impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            nodes: 6_000,
            max_trace_nodes: 0,
            avg_degree: 5,
            dim: 64,
            seed: 0xBEEFu64,
            prefetch: 64,
            top_k: 80,
            seed_count: 24,
            max_hops: 6,
            min_path_weight: 0.18,
        }
    }
}

#[cfg(feature = "serde")]
struct ExportConfig {
    output: Option<PathBuf>,
    pretty: bool,
    synthetic: SyntheticConfig,
    mode: ExportMode,
}

#[cfg(feature = "serde")]
fn usage() -> &'static str {
    "usage:
  cargo run -p turbo-graph --features serde --example graph_memory_debug_export
  cargo run -p turbo-graph --features serde --example graph_memory_debug_export -- \\
    --raw \\
    --nodes <usize> --avg-degree <usize> --seed <u64> --seed-count <usize> \\
    --max-trace-nodes <usize> \\
    --dim <usize> --top-k <usize> --prefetch <usize> --max-hops <usize> \\
    --min-path-weight <f32> --output <path> [--compact]"
}

#[cfg(feature = "serde")]
fn parse_u64(raw: &str, field: &str) -> Result<u64, String> {
    raw.parse::<u64>()
        .map_err(|error| format!("invalid value for {field}: {raw} ({error})"))
}

#[cfg(feature = "serde")]
fn parse_usize(raw: &str, field: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|error| format!("invalid value for {field}: {raw} ({error})"))
}

#[cfg(feature = "serde")]
fn parse_f32(raw: &str, field: &str) -> Result<f32, String> {
    raw.parse::<f32>()
        .map_err(|error| format!("invalid value for {field}: {raw} ({error})"))
}

#[cfg(feature = "serde")]
fn parse_config() -> Result<Option<ExportConfig>, String> {
    let mut output = None;
    let mut pretty = true;
    let mut mode = ExportMode::Search;

    let mut synthetic = SyntheticConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--output" | "-o" => {
                output = Some(PathBuf::from(args.next().ok_or_else(|| {
                    "--output requires a path; e.g. --output /tmp/graph-memory-snapshot.json"
                        .to_string()
                })?));
            }
            "--compact" => {
                pretty = false;
            }
            "--raw" => {
                mode = ExportMode::RawGraph;
            }
            "--nodes" => {
                synthetic.nodes = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--nodes requires a value; e.g. --nodes 10000".to_string()
                    })?,
                    "--nodes",
                )?;
            }
            "--max-trace-nodes" => {
                synthetic.max_trace_nodes = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--max-trace-nodes requires a value; e.g. --max-trace-nodes 18000"
                            .to_string()
                    })?,
                    "--max-trace-nodes",
                )?;
            }
            "--avg-degree" => {
                synthetic.avg_degree = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--avg-degree requires a value; e.g. --avg-degree 6".to_string()
                    })?,
                    "--avg-degree",
                )?;
            }
            "--seed" => {
                synthetic.seed = parse_u64(
                    &args
                        .next()
                        .ok_or_else(|| "--seed requires a value; e.g. --seed 42".to_string())?,
                    "--seed",
                )?;
            }
            "--dim" => {
                synthetic.dim = parse_usize(
                    &args
                        .next()
                        .ok_or_else(|| "--dim requires a value; e.g. --dim 64".to_string())?,
                    "--dim",
                )?;
            }
            "--top-k" => {
                synthetic.top_k = parse_usize(
                    &args
                        .next()
                        .ok_or_else(|| "--top-k requires a value; e.g. --top-k 80".to_string())?,
                    "--top-k",
                )?;
            }
            "--seed-count" => {
                synthetic.seed_count = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--seed-count requires a value; e.g. --seed-count 24".to_string()
                    })?,
                    "--seed-count",
                )?;
            }
            "--prefetch" => {
                synthetic.prefetch = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--prefetch requires a value; e.g. --prefetch 64".to_string()
                    })?,
                    "--prefetch",
                )?;
            }
            "--max-hops" => {
                synthetic.max_hops = parse_usize(
                    &args.next().ok_or_else(|| {
                        "--max-hops requires a value; e.g. --max-hops 3".to_string()
                    })?,
                    "--max-hops",
                )?;
            }
            "--min-path-weight" => {
                synthetic.min_path_weight = parse_f32(
                    &args.next().ok_or_else(|| {
                        "--min-path-weight requires a value; e.g. --min-path-weight 0.20"
                            .to_string()
                    })?,
                    "--min-path-weight",
                )?;
            }
            other => return Err(format!("unknown argument {other:?}\n\n{}", usage())),
        }
    }

    if synthetic.nodes == 0 {
        return Err("--nodes must be >= 1".to_string());
    }
    if synthetic.avg_degree == 0 {
        return Err("--avg-degree must be >= 1".to_string());
    }
    if synthetic.dim == 0 {
        return Err("--dim must be >= 1".to_string());
    }
    if !(0.0..=1.0).contains(&synthetic.min_path_weight) {
        return Err("--min-path-weight should be in [0.0, 1.0]".to_string());
    }
    if synthetic.max_trace_nodes != 0 && synthetic.max_trace_nodes < 1 {
        return Err("--max-trace-nodes must be >= 1".to_string());
    }

    Ok(Some(ExportConfig {
        output,
        pretty,
        synthetic,
        mode,
    }))
}

#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

#[cfg(feature = "serde")]
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            0
        } else {
            (self.next_u64() as usize) % max
        }
    }
}

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
fn build_raw_snapshot(
    config: &SyntheticConfig,
) -> turbo_graph::GraphCandidateHybridSearchDebugSnapshot {
    let mut rng = SplitMix64::new(config.seed ^ 0xBADD_BEEF_CAFE_u64);
    let sources = ["kuku.mom", "liner", "planner", "archive", "search", "graph"];
    let tags = [
        "graph",
        "search",
        "ranking",
        "vector",
        "memory",
        "product",
        "index",
        "paper",
        "candidate",
        "policy",
        "context",
    ];

    let mut nodes = Vec::with_capacity(config.nodes);
    for i in 0..config.nodes {
        let id = 10_000_u64 + i as u64;
        let depth = rng.next_usize(9);
        let slot = i;
        let source = sources[rng.next_usize(sources.len())];
        let first = rng.next_usize(tags.len());
        let second = (first + 1 + rng.next_usize(tags.len())) % tags.len();
        nodes.push(turbo_graph::GraphHybridSearchDebugNode {
            id,
            slot,
            depth,
            parent: if i == 0 {
                None
            } else {
                Some(10_000_u64 + rng.next_usize(i) as u64)
            },
            via_weight: (rng.next_f32() * 0.8 + 0.2).clamp(0.2, 1.0),
            path_weight: (rng.next_f32() * 0.8 + 0.2).clamp(0.2, 1.0),
            hit_rank: None,
            score: None,
            vector_score: None,
            graph_score: None,
            candidate_score: Some((0.25 + rng.next_f32() * 0.75).clamp(0.25, 1.0)),
            title: format!("raw node {id}"),
            tags: vec![tags[first].to_string(), tags[second].to_string()],
            source: Some(source.to_string()),
            timestamp_ms: Some(1_700_000_000_000 + (i as i64 * 7)),
        });
    }

    let mut edge_set = std::collections::HashSet::new();
    let avg_degree = config.avg_degree.max(1);
    let mut edges = Vec::with_capacity(config.nodes.saturating_mul(avg_degree));
    for i in 0..config.nodes {
        let from = 10_000_u64 + i as u64;
        for _ in 0..avg_degree {
            let to = 10_000_u64 + rng.next_usize(config.nodes) as u64;
            if from == to {
                continue;
            }
            let edge_id = format!("{from}:{to}");
            if edge_set.insert(edge_id) {
                edges.push(turbo_graph::GraphSearchDebugEdge {
                    from,
                    to,
                    weight: (rng.next_f32() * 0.9 + 0.1).clamp(0.1, 1.0),
                });
            }
        }
    }

    let hit_count = config.nodes.min(5000);
    let mut hits = Vec::with_capacity(hit_count);
    for rank in 1..=hit_count {
        let id = 10_000_u64 + (rank as u64 - 1);
        let node = &nodes[(rank - 1) % nodes.len()];
        hits.push(turbo_graph::GraphHybridSearchDebugHit {
            rank,
            id,
            score: 1.0 - rank as f32 / (hit_count as f32 + 1.0),
            vector_score: 1.0 - rank as f32 / (hit_count as f32 + 2.0),
            graph_score: 1.0 - rank as f32 / (hit_count as f32 + 3.0),
            candidate_score: 1.0 - rank as f32 / (hit_count as f32 + 4.0),
            depth: node.depth,
            parent: node.parent,
            title: node.title.clone(),
            tags: node.tags.clone(),
            source: node.source.clone(),
            timestamp_ms: node.timestamp_ms,
        });
    }

    let selected_slots = nodes.len().min(1200);
    let summary = GraphCandidateSearchDebugSummary {
        total_slots: nodes.len(),
        graph_slots: nodes.len(),
        metadata_slots: nodes.len(),
        candidate_input_ids: nodes.len(),
        candidate_slots: nodes.len(),
        candidate_missing_ids: 0,
        candidate_duplicate_ids: 0,
        selected_slots,
        active_blocks: 1,
        graph_cache_hit: false,
        combined_cache_hit: false,
        selectivity: selected_slots as f32 / nodes.len().max(1) as f32,
        graph_selectivity: 1.0,
        metadata_selectivity: 1.0,
        candidate_selectivity: 1.0,
        active_block_selectivity: 1.0,
        prefetch_k: config.prefetch,
        hit_count: hits.len(),
        trace_node_count: nodes.len(),
        trace_edge_count: edges.len(),
    };

    turbo_graph::GraphCandidateHybridSearchDebugSnapshot {
        summary,
        telemetry: turbo_graph::GraphSearchTelemetry {
            view_build_ns: 0,
            vector_search_ns: 0,
            rerank_ns: 0,
            trace_build_ns: 0,
            total_ns: 0,
            blocks_skipped_by_mask: 0,
        },
        hits,
        nodes,
        edges,
    }
}

#[cfg(feature = "serde")]
fn build_synthetic_snapshot(
    config: &SyntheticConfig,
) -> turbo_graph::GraphCandidateHybridSearchDebugSnapshot {
    let mut rng = SplitMix64::new(config.seed ^ 0xBADC0FFEEu64);
    let avg_degree = config.avg_degree.max(1);
    let max_edges = config.nodes.saturating_mul(avg_degree);

    let mut records = Vec::with_capacity(config.nodes);
    let sources = ["kuku.mom", "liner", "planner", "archive", "search", "graph"];
    let tags = [
        "graph",
        "search",
        "ranking",
        "vector",
        "memory",
        "product",
        "index",
        "paper",
        "candidate",
        "policy",
        "context",
    ];

    let embeddings: Vec<f32> = (0..config.nodes)
        .flat_map(|i| toy_embedding(config.dim, config.seed.wrapping_add(i as u64 + 1)).into_iter())
        .collect();

    for i in 0..config.nodes {
        let id = 10_000_u64 + i as u64;
        let first = rng.next_usize(tags.len());
        let second = (first + 1 + rng.next_usize(tags.len())) % tags.len();
        let source = sources[rng.next_usize(sources.len())];
        let ts = 1_700_000_000_000_u64 + (i as u64 % 86_400) * 7 + (rng.next_u64() % 10_000);

        records.push(
            MemoryRecord::new(
                id,
                format!("synthetic node {id}"),
                [tags[first], tags[second]],
            )
            .with_source(source)
            .with_timestamp_ms(ts as i64),
        );
    }

    let mut memory = GraphMemoryIndex::new(config.dim, 4).expect("valid graph memory config");
    memory
        .add_records(&embeddings, records)
        .expect("records added");

    for from_offset in 0..config.nodes {
        let from = 10_000_u64 + from_offset as u64;
        let target_degree = avg_degree + (rng.next_usize(3) as usize);
        for _ in 0..target_degree {
            let to = 10_000_u64 + rng.next_usize(config.nodes) as u64;
            if to == from {
                continue;
            }
            let weight = (0.2 + rng.next_f32() * 0.8).clamp(0.05, 1.0);
            memory
                .link_bidirectional(from, to, weight)
                .expect("graph edge linked");
        }
    }

    // Ensure a connected scaffold so the search remains coherent at large scale.
    for i in 1..config.nodes {
        let from = 10_000_u64 + i as u64 - 1;
        let to = 10_000_u64 + i as u64;
        memory
            .link_bidirectional(
                from,
                to,
                (0.80 - (i as f32 / max_edges.max(1) as f32 * 0.30)).clamp(0.35, 0.8),
            )
            .expect("chain link");
    }

    let query = toy_embedding(config.dim, config.seed.wrapping_add(0xD1E));
    let max_nodes = if config.max_trace_nodes == 0 {
        config.nodes
    } else {
        config.max_trace_nodes.min(config.nodes)
    };
    let policy = GraphViewPolicy::new(config.max_hops)
        .with_max_nodes(max_nodes.max(1))
        .with_max_active_blocks((config.nodes / 350).max(4).min(24))
        .with_min_path_weight(config.min_path_weight);

    let candidate_count = config.nodes.min((config.nodes * 4) / 5).max(1);
    let mut candidate_scores = Vec::with_capacity(candidate_count);
    for i in 0..candidate_count {
        let id = 10_000_u64 + i as u64;
        candidate_scores.push((
            id,
            0.03 + ((candidate_count - i) as f32 / candidate_count as f32),
        ));
    }

    let rerank = GraphHybridRerankConfig::new(0.64, 0.21, 0.15)
        .with_candidate_score_normalization(GraphCandidateScoreNormalization::MinMax)
        .with_prefetch_factor(3)
        .with_min_prefetch(config.prefetch);

    let seed_count = config.seed_count.max(1).min(config.nodes);
    let mut seed_ids = Vec::with_capacity(seed_count);
    for i in 0..seed_count {
        seed_ids.push(10_000_u64 + i as u64);
    }

    let report = memory.explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed(
        &query,
        config.top_k,
        &seed_ids,
        policy,
        &[],
        &[],
        Some(1_700_000_000_000),
        None,
        &candidate_scores,
        rerank,
    );
    report.debug_snapshot()
}

#[cfg(feature = "serde")]
fn main() {
    let config = match parse_config() {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!("{}", usage());
            return;
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    let snapshot = match config.mode {
        ExportMode::Search => build_synthetic_snapshot(&config.synthetic),
        ExportMode::RawGraph => build_raw_snapshot(&config.synthetic),
    };
    let json = if config.pretty {
        serde_json::to_string_pretty(&snapshot).unwrap()
    } else {
        serde_json::to_string(&snapshot).unwrap()
    };

    if let Some(path) = config.output {
        fs::write(&path, json).unwrap_or_else(|err| {
            eprintln!("failed to write {}: {err}", path.display());
            std::process::exit(2);
        });
        eprintln!("wrote {}", path.display());
    } else {
        println!("{json}");
    }
}

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!("enable the `serde` feature to export graph-memory debug snapshots as JSON");
}
