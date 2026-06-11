//! Benchmark graph-view candidate pruning.
//!
//! Run with optimizations:
//!   cargo run -p turbo-graph --release --example graph_view_bench
//!   cargo run -p turbo-graph --release --example graph_view_bench -- \
//!     --vectors-f32 embeddings.f32 --dim 768 --query-row 0 --csv bench.csv
//!
//! This intentionally stays dependency-free. It prints a compact table for
//! global search, caller bool masks, cached SlotMask, GraphMemoryIndex
//! graph-view cache hits, and preset-driven graph+metadata+rerank workloads.

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use turbo_graph::search::{blocks_skipped_by_mask, reset_blocks_skipped_by_mask};
use turbo_graph::{GraphMemoryIndex, GraphSearchPreset, MemoryRecord, SlotMask, TurboQuantIndex};

const DEFAULT_DIM: usize = 64;
const DEFAULT_N: usize = 16_384;
const K: usize = 10;
const DEFAULT_ITERS: usize = 50;
const BATCH_QUERY_ROWS: usize = 8;
const SELECTIVITY: &[f32] = &[0.001, 0.01, 0.05, 0.20, 1.0];
const WORKLOAD_FANOUT: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

#[derive(Clone, Copy)]
struct Timing {
    elapsed: Duration,
    skipped_blocks: u64,
}

struct BenchConfig {
    source: BenchSource,
    iters: usize,
    csv_path: Option<PathBuf>,
}

enum BenchSource {
    Synthetic,
    F32 {
        vectors_path: PathBuf,
        queries_path: Option<PathBuf>,
        dim: usize,
        query_row: usize,
    },
}

struct CsvRecorder {
    rows: Vec<String>,
}

fn toy_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
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

fn usage() -> &'static str {
    "usage:
  cargo run -p turbo-graph --release --example graph_view_bench
  cargo run -p turbo-graph --release --example graph_view_bench -- \\
    --vectors-f32 embeddings.f32 --dim 768 [--queries-f32 queries.f32] [--query-row 0] [--iters 50] [--csv bench.csv]

input format:
  *.f32 files are raw little-endian f32 values in row-major order.
  --vectors-f32 length must be a multiple of --dim.
  --queries-f32, when present, uses every row as a benchmark query batch.
  otherwise --query-row selects one vector row from --vectors-f32 as the query.
  single-query modes are expanded to 8 rows so batch search paths are exercised."
}

fn parse_config() -> Result<Option<BenchConfig>, String> {
    let mut vectors_path = None;
    let mut queries_path = None;
    let mut dim = None;
    let mut query_row = 0usize;
    let mut iters = DEFAULT_ITERS;
    let mut csv_path = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--vectors-f32" => {
                vectors_path =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--vectors-f32 requires a path".to_string()
                    })?));
            }
            "--queries-f32" => {
                queries_path =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--queries-f32 requires a path".to_string()
                    })?));
            }
            "--dim" => {
                dim = Some(parse_usize_arg("--dim", args.next())?);
            }
            "--query-row" => {
                query_row = parse_usize_arg("--query-row", args.next())?;
            }
            "--iters" => {
                iters = parse_usize_arg("--iters", args.next())?.max(1);
            }
            "--csv" => {
                csv_path = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--csv requires a path".to_string())?,
                ));
            }
            other => {
                return Err(format!("unknown argument {other:?}\n\n{}", usage()));
            }
        }
    }

    let source = match vectors_path {
        Some(vectors_path) => BenchSource::F32 {
            vectors_path,
            queries_path,
            dim: dim.ok_or_else(|| "--dim is required with --vectors-f32".to_string())?,
            query_row,
        },
        None => {
            if dim.is_some() || queries_path.is_some() || query_row != 0 {
                return Err("--vectors-f32 is required for corpus benchmark options".to_string());
            }
            BenchSource::Synthetic
        }
    };

    Ok(Some(BenchConfig {
        source,
        iters,
        csv_path,
    }))
}

fn parse_usize_arg(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<usize>()
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn read_f32_matrix(path: &PathBuf, dim: usize) -> Result<Vec<f32>, String> {
    if dim == 0 {
        return Err("--dim must be greater than zero".to_string());
    }
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} has {} bytes, not a multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if values.is_empty() {
        return Err(format!("{} contains no f32 values", path.display()));
    }
    if values.len() % dim != 0 {
        return Err(format!(
            "{} has {} f32 values, not a multiple of dim={dim}",
            path.display(),
            values.len()
        ));
    }
    Ok(values)
}

fn query_from_corpus(data: &[f32], dim: usize, query_row: usize) -> Result<Vec<f32>, String> {
    let n = data.len() / dim;
    if query_row >= n {
        return Err(format!(
            "--query-row {query_row} is out of bounds for {n} vector rows"
        ));
    }
    Ok(data[query_row * dim..(query_row + 1) * dim].to_vec())
}

fn post_filter_fetches(n: usize, k: usize) -> Vec<usize> {
    let mut values = vec![k, k * 4, k * 16, k * 64, k * 256, n / 2, n];
    for value in &mut values {
        *value = (*value).clamp(1, n);
    }
    values.sort_unstable();
    values.dedup();
    values
}

fn expand_queries_for_batch(queries: &[f32], dim: usize) -> Vec<f32> {
    let n_query = queries.len() / dim;
    if dim == 0 || n_query == 0 || n_query >= 2 {
        return queries.to_vec();
    }
    let first = queries[..dim].to_vec();
    let mut expanded = Vec::with_capacity(dim * BATCH_QUERY_ROWS);
    for _ in 0..BATCH_QUERY_ROWS {
        expanded.extend_from_slice(&first);
    }
    expanded
}

impl CsvRecorder {
    fn new() -> Self {
        Self {
            rows: vec![[
                "mode",
                "phase",
                "case",
                "selectivity",
                "fetch_k",
                "vectors",
                "dim",
                "queries",
                "k",
                "iters",
                "allowed",
                "selected_slots",
                "active_blocks",
                "prefetch_k",
                "ms_per_iter",
                "ratio_to_global",
                "blocks_skipped",
                "cache_entries",
                "recall",
                "post_hits",
                "direct_mask_ms",
            ]
            .join(",")],
        }
    }

    fn push(&mut self, fields: Vec<String>) {
        assert_eq!(fields.len(), 21, "CSV row must match header width");
        self.rows.push(
            fields
                .into_iter()
                .map(csv_escape)
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    fn finish(self) -> String {
        let mut out = self.rows.join("\n");
        out.push('\n');
        out
    }
}

fn csv_escape(value: String) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value
    }
}

fn record_csv(csv: &mut Option<CsvRecorder>, fields: Vec<String>) {
    if let Some(csv) = csv.as_mut() {
        csv.push(fields);
    }
}

fn blank() -> String {
    String::new()
}

fn fmt_f64(value: f64) -> String {
    format!("{value:.6}")
}

fn time_it(iters: usize, mut f: impl FnMut()) -> Timing {
    reset_blocks_skipped_by_mask();
    let before = blocks_skipped_by_mask();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let after = blocks_skipped_by_mask();
    Timing {
        elapsed,
        skipped_blocks: after - before,
    }
}

fn ms_per_iter(timing: Timing, iters: usize) -> f64 {
    timing.elapsed.as_secs_f64() * 1000.0 / iters as f64
}

fn filtered_global_slots(
    index: &TurboQuantIndex,
    queries: &[f32],
    fetch_k: usize,
    mask: &SlotMask,
    n: usize,
    k: usize,
) -> Vec<i64> {
    let results = index.search(queries, fetch_k.min(n));
    let mut filtered = Vec::new();
    for &slot in results.indices_for_query(0) {
        if slot >= 0 && mask.contains(slot as usize) {
            filtered.push(slot);
            if filtered.len() == k {
                break;
            }
        }
    }
    filtered
}

fn recall_at_k(reference: &[i64], candidates: &[i64]) -> f64 {
    if reference.is_empty() {
        return 1.0;
    }
    let hits = reference
        .iter()
        .filter(|&&slot| candidates.contains(&slot))
        .count();
    hits as f64 / reference.len() as f64
}

fn build_graph_memory(data: &[f32], dim: usize) -> GraphMemoryIndex {
    let n = data.len() / dim;
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| {
            MemoryRecord::new(i as u64, format!("bench node {i}"), ["bench"])
                .with_source("bench")
                .with_timestamp_ms(i as i64)
        })
        .collect();
    memory.add_records(data, records).unwrap();
    for i in 0..n.saturating_sub(1) {
        memory.link_directed(i as u64, (i + 1) as u64, 1.0).unwrap();
    }
    memory.prepare();
    memory
}

fn build_workload_memory(data: &[f32], dim: usize) -> GraphMemoryIndex {
    let n = data.len() / dim;
    let mut memory = GraphMemoryIndex::new(dim, 4).unwrap();
    let records: Vec<MemoryRecord> = (0..n)
        .map(|i| {
            let tags = match i % 4 {
                0 => vec!["bench", "architecture"],
                1 => vec!["bench", "product"],
                2 => vec!["bench", "ranking"],
                _ => vec!["bench", "archive"],
            };
            let source = match i % 4 {
                0 => "kuku.mom",
                1 => "liner",
                2 => "web",
                _ => "archive",
            };
            MemoryRecord::new(i as u64, format!("workload node {i}"), tags)
                .with_source(source)
                .with_timestamp_ms(i as i64)
        })
        .collect();
    memory.add_records(data, records).unwrap();

    for i in 0..n {
        for (rank, &step) in WORKLOAD_FANOUT.iter().enumerate() {
            let to = i + step;
            if to >= n {
                continue;
            }
            let weight = 1.0 - rank as f32 * 0.06;
            memory
                .link_directed(i as u64, to as u64, weight.max(0.25))
                .unwrap();
        }
    }
    memory.prepare();
    memory
}

fn run_preset_workload(
    label: &str,
    index: &TurboQuantIndex,
    data: &[f32],
    dim: usize,
    queries: &[f32],
    global: Timing,
    n: usize,
    nq: usize,
    k: usize,
    iters: usize,
    csv: &mut Option<CsvRecorder>,
) {
    let mut memory = build_workload_memory(data, dim);
    let workload_seed = (n / 3) as u64;
    let query = &queries[..dim];
    let global_batch_denominator_ms = (global.elapsed.as_secs_f64() * nq.max(1) as f64).max(1e-18);
    let cases = [
        ("low_latency", GraphSearchPreset::low_latency()),
        ("balanced", GraphSearchPreset::balanced()),
        ("broad", GraphSearchPreset::broad()),
    ];
    let mut balanced_mask = None;

    println!();
    println!(
        "preset workload seed={workload_seed} sources=kuku.mom|liner tag=bench iterations={iters}"
    );
    println!(
        "{:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>11} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "preset",
        "blk_cap",
        "node_cap",
        "selected",
        "act_blk",
        "prefetch",
        "packed_ms",
        "graph_ms",
        "batch_ms",
        "packed/glob",
        "graph/glob",
        "batch/glob",
        "batch/graph",
        "blk_skip",
        "cache"
    );

    for (name, preset) in cases {
        let tuning = memory.tuning_for_preset(k, preset);
        let (mask, cold_plan) = memory.graph_view_mask_with_policy_metadata_plan(
            &[workload_seed],
            tuning.policy,
            &["bench"],
            &["kuku.mom", "liner"],
            Some(0),
            Some(n as i64),
        );
        black_box(cold_plan.combined_cache_hit);

        let prepared = memory.prepare_graph_view_with_policy_metadata(
            &[workload_seed],
            tuning.policy,
            &["bench"],
            &["kuku.mom", "liner"],
            Some(0),
            Some(n as i64),
        );

        let warm = memory.search_graph_view_with_policy_metadata_rerank_timed(
            query,
            k,
            &[workload_seed],
            tuning.policy,
            &["bench"],
            &["kuku.mom", "liner"],
            Some(0),
            Some(n as i64),
            tuning.rerank,
        );
        assert!(warm.plan.combined_cache_hit);

        let packed = time_it(iters, || {
            let res = index.search_with_slot_mask(black_box(queries), k, &mask);
            black_box(res.scores.len());
        });
        let graph = time_it(iters, || {
            for qi in 0..nq {
                let q = &queries[(qi * dim)..((qi + 1) * dim)];
                let report = memory.search_graph_view_with_policy_metadata_rerank_timed(
                    black_box(q),
                    k,
                    &[workload_seed],
                    tuning.policy,
                    &["bench"],
                    &["kuku.mom", "liner"],
                    Some(0),
                    Some(n as i64),
                    tuning.rerank,
                );
                black_box(report.hits.len());
                black_box(report.plan.combined_cache_hit);
                black_box(report.telemetry.blocks_skipped_by_mask);
            }
        });
        let batch = time_it(iters, || {
            let report =
                prepared.search_rerank_batch_timed(&memory, black_box(queries), k, tuning.rerank);
            black_box(report.hits.len());
            black_box(report.plan.combined_cache_hit);
            black_box(report.prefetch_k);
        });
        let cache = memory.cache_stats();
        if name == "balanced" {
            balanced_mask = Some((
                mask.clone(),
                warm.plan.selected_slots,
                warm.plan.active_blocks,
            ));
        }

        println!(
            "{:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>11.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12} {:>10}",
            name,
            tuning.policy.max_active_blocks,
            tuning.policy.max_nodes,
            warm.plan.selected_slots,
            warm.plan.active_blocks,
            warm.prefetch_k,
            ms_per_iter(packed, iters),
            ms_per_iter(graph, iters),
            ms_per_iter(batch, iters),
            packed.elapsed.as_secs_f64() / global_batch_denominator_ms,
            graph.elapsed.as_secs_f64() / global_batch_denominator_ms,
            batch.elapsed.as_secs_f64() / global_batch_denominator_ms,
            batch.elapsed.as_secs_f64() / graph.elapsed.as_secs_f64().max(1e-12),
            graph.skipped_blocks,
            cache.total_entries,
        );
        record_csv(
            csv,
            vec![
                label.to_string(),
                "preset".to_string(),
                format!("{name}:packed"),
                blank(),
                blank(),
                n.to_string(),
                dim.to_string(),
                nq.to_string(),
                k.to_string(),
                iters.to_string(),
                blank(),
                warm.plan.selected_slots.to_string(),
                warm.plan.active_blocks.to_string(),
                warm.prefetch_k.to_string(),
                fmt_f64(ms_per_iter(packed, iters)),
                fmt_f64(packed.elapsed.as_secs_f64() / global_batch_denominator_ms),
                packed.skipped_blocks.to_string(),
                cache.total_entries.to_string(),
                blank(),
                blank(),
                blank(),
            ],
        );
        record_csv(
            csv,
            vec![
                label.to_string(),
                "preset".to_string(),
                format!("{name}:graph"),
                blank(),
                blank(),
                n.to_string(),
                dim.to_string(),
                nq.to_string(),
                k.to_string(),
                iters.to_string(),
                blank(),
                warm.plan.selected_slots.to_string(),
                warm.plan.active_blocks.to_string(),
                warm.prefetch_k.to_string(),
                fmt_f64(ms_per_iter(graph, iters)),
                fmt_f64(graph.elapsed.as_secs_f64() / global_batch_denominator_ms),
                graph.skipped_blocks.to_string(),
                cache.total_entries.to_string(),
                blank(),
                blank(),
                blank(),
            ],
        );
        record_csv(
            csv,
            vec![
                label.to_string(),
                "preset".to_string(),
                format!("{name}:batch-rerank"),
                blank(),
                blank(),
                n.to_string(),
                dim.to_string(),
                nq.to_string(),
                k.to_string(),
                iters.to_string(),
                blank(),
                warm.plan.selected_slots.to_string(),
                warm.plan.active_blocks.to_string(),
                warm.prefetch_k.to_string(),
                fmt_f64(ms_per_iter(batch, iters)),
                fmt_f64(batch.elapsed.as_secs_f64() / global_batch_denominator_ms),
                batch.skipped_blocks.to_string(),
                cache.total_entries.to_string(),
                blank(),
                blank(),
                blank(),
            ],
        );
    }

    let (mask, selected_slots, active_blocks) = balanced_mask.expect("balanced preset ran");
    run_post_filter_recall(
        label,
        index,
        query,
        &mask,
        selected_slots,
        active_blocks,
        n,
        dim,
        1,
        k,
        iters,
        csv,
    );
}

fn run_post_filter_recall(
    label: &str,
    index: &TurboQuantIndex,
    query: &[f32],
    mask: &SlotMask,
    selected_slots: usize,
    active_blocks: usize,
    n: usize,
    dim: usize,
    nq: usize,
    k: usize,
    iters: usize,
    csv: &mut Option<CsvRecorder>,
) {
    let direct = index.search_with_slot_mask(query, k, mask);
    let direct_slots: Vec<i64> = direct.indices_for_query(0).to_vec();
    let direct_timing = time_it(iters, || {
        let results = index.search_with_slot_mask(black_box(query), k, mask);
        black_box(results.indices_for_query(0).len());
    });

    println!();
    println!(
        "post-filter recall selected={selected_slots} active_blocks={active_blocks} direct_mask_ms={:.3}",
        ms_per_iter(direct_timing, iters),
    );
    println!(
        "{:>8} {:>9} {:>10} {:>11} {:>13}",
        "fetch_k",
        "post_hits",
        format!("recall@{k}"),
        "post_ms",
        "post/direct"
    );
    let direct_ms = ms_per_iter(direct_timing, iters);
    record_csv(
        csv,
        vec![
            label.to_string(),
            "post_filter".to_string(),
            "direct_mask".to_string(),
            blank(),
            blank(),
            n.to_string(),
            dim.to_string(),
            nq.to_string(),
            k.to_string(),
            iters.to_string(),
            blank(),
            selected_slots.to_string(),
            active_blocks.to_string(),
            blank(),
            fmt_f64(direct_ms),
            "1.000000".to_string(),
            direct_timing.skipped_blocks.to_string(),
            blank(),
            "1.000000".to_string(),
            direct_slots.len().to_string(),
            fmt_f64(direct_ms),
        ],
    );

    for fetch_k in post_filter_fetches(n, k) {
        let filtered = filtered_global_slots(index, query, fetch_k, mask, n, k);
        let recall = recall_at_k(&direct_slots, &filtered);
        let post_timing = time_it(iters, || {
            let filtered = filtered_global_slots(index, black_box(query), fetch_k, mask, n, k);
            black_box(filtered.len());
        });

        println!(
            "{:>8} {:>9} {:>10.2} {:>11.3} {:>13.3}",
            fetch_k,
            filtered.len(),
            recall,
            ms_per_iter(post_timing, iters),
            post_timing.elapsed.as_secs_f64() / direct_timing.elapsed.as_secs_f64(),
        );
        record_csv(
            csv,
            vec![
                label.to_string(),
                "post_filter".to_string(),
                "global_overfetch".to_string(),
                blank(),
                fetch_k.to_string(),
                n.to_string(),
                dim.to_string(),
                nq.to_string(),
                k.to_string(),
                iters.to_string(),
                blank(),
                selected_slots.to_string(),
                active_blocks.to_string(),
                blank(),
                fmt_f64(ms_per_iter(post_timing, iters)),
                fmt_f64(post_timing.elapsed.as_secs_f64() / direct_timing.elapsed.as_secs_f64()),
                post_timing.skipped_blocks.to_string(),
                blank(),
                fmt_f64(recall),
                filtered.len().to_string(),
                fmt_f64(direct_ms),
            ],
        );
    }
}

fn run_benchmark(
    label: &str,
    data: Vec<f32>,
    dim: usize,
    queries: Vec<f32>,
    iters: usize,
    csv_path: Option<PathBuf>,
) -> Result<(), String> {
    assert!(dim > 0, "benchmark dim must be positive");
    assert!(
        data.len() % dim == 0,
        "data length must be a multiple of dim"
    );
    assert!(
        queries.len() % dim == 0,
        "query length must be a multiple of dim"
    );
    let n = data.len() / dim;
    let nq = queries.len() / dim;
    let query = queries[..dim].to_vec();
    let batch_queries = expand_queries_for_batch(&queries, dim);
    let batch_nq = batch_queries.len() / dim;
    assert!(n > 0, "benchmark needs at least one vector");
    assert!(nq > 0, "benchmark needs at least one query");
    let k = K.min(n);
    let mut csv = csv_path.as_ref().map(|_| CsvRecorder::new());

    let mut index = TurboQuantIndex::new(dim, 4).unwrap();
    index.add(&data);
    index.prepare();

    let mut memory = build_graph_memory(&data, dim);

    let global = time_it(iters, || {
        let res = index.search(black_box(&queries), k);
        black_box(res.scores.len());
    });

    let global_ms = ms_per_iter(global, iters);
    println!("mode={label} vectors={n} dim={dim} queries={nq} k={k} iterations={iters}");
    println!("global_ms_per_iter={:.3}", global_ms);
    record_csv(
        &mut csv,
        vec![
            label.to_string(),
            "global".to_string(),
            "global".to_string(),
            blank(),
            blank(),
            n.to_string(),
            dim.to_string(),
            nq.to_string(),
            k.to_string(),
            iters.to_string(),
            blank(),
            blank(),
            blank(),
            blank(),
            fmt_f64(global_ms),
            "1.000000".to_string(),
            global.skipped_blocks.to_string(),
            blank(),
            blank(),
            blank(),
            blank(),
        ],
    );
    println!(
        "{:>8} {:>8} {:>8} {:>11} {:>11} {:>12} {:>12} {:>12} {:>12}",
        "sel%",
        "allowed",
        "act_blk",
        "bool_ms",
        "slot_ms",
        "graph_ms",
        "slot/global",
        "graph/glob",
        "blk_skip"
    );

    for &selectivity in SELECTIVITY {
        let allowed = ((n as f32 * selectivity).round() as usize).clamp(1, n);
        let start_slot = n - allowed;
        let allowed_slots: Vec<usize> = (start_slot..n).collect();
        let slot_mask = SlotMask::from_slots(n, allowed_slots.iter().copied());
        let mut bool_mask = vec![false; n];
        for &slot in &allowed_slots {
            bool_mask[slot] = true;
        }

        let graph_seed = start_slot as u64;
        let graph_hops = allowed.saturating_sub(1);
        let (first_mask, first_view) = memory.graph_view_mask_with_stats(&[graph_seed], graph_hops);
        assert_eq!(first_mask.count(), allowed);
        black_box(first_view.cache_hit);
        let (_second_mask, second_view) =
            memory.graph_view_mask_with_stats(&[graph_seed], graph_hops);
        assert!(second_view.cache_hit);

        let bool_filtered = time_it(iters, || {
            let res = index.search_with_mask(black_box(&queries), k, Some(&bool_mask));
            black_box(res.scores.len());
        });
        let packed_filtered = time_it(iters, || {
            let res = index.search_with_slot_mask(black_box(&queries), k, &slot_mask);
            black_box(res.scores.len());
        });
        let graph_filtered = time_it(iters, || {
            let report = memory.search_graph_view_with_stats(
                black_box(&query),
                k,
                &[graph_seed],
                graph_hops,
                &[],
            );
            black_box(report.hits.len());
            black_box(report.view.cache_hit);
        });

        println!(
            "{:>8.2} {:>8} {:>8} {:>11.3} {:>11.3} {:>12.3} {:>12.3} {:>12.3} {:>12}",
            selectivity * 100.0,
            allowed,
            slot_mask.active_block_count(),
            ms_per_iter(bool_filtered, iters),
            ms_per_iter(packed_filtered, iters),
            ms_per_iter(graph_filtered, iters),
            packed_filtered.elapsed.as_secs_f64() / global.elapsed.as_secs_f64(),
            graph_filtered.elapsed.as_secs_f64() / global.elapsed.as_secs_f64(),
            packed_filtered.skipped_blocks,
        );
        for (case, timing) in [
            ("bool_mask", bool_filtered),
            ("slot_mask", packed_filtered),
            ("graph_memory", graph_filtered),
        ] {
            record_csv(
                &mut csv,
                vec![
                    label.to_string(),
                    "selectivity".to_string(),
                    case.to_string(),
                    fmt_f64(selectivity as f64),
                    blank(),
                    n.to_string(),
                    dim.to_string(),
                    nq.to_string(),
                    k.to_string(),
                    iters.to_string(),
                    allowed.to_string(),
                    allowed.to_string(),
                    slot_mask.active_block_count().to_string(),
                    blank(),
                    fmt_f64(ms_per_iter(timing, iters)),
                    fmt_f64(timing.elapsed.as_secs_f64() / global.elapsed.as_secs_f64()),
                    timing.skipped_blocks.to_string(),
                    blank(),
                    blank(),
                    blank(),
                    blank(),
                ],
            );
        }
    }

    run_preset_workload(
        label,
        &index,
        &data,
        dim,
        &batch_queries,
        global,
        n,
        batch_nq,
        k,
        iters,
        &mut csv,
    );

    if let (Some(path), Some(csv)) = (csv_path, csv) {
        fs::write(&path, csv.finish())
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

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

    let BenchConfig {
        source,
        iters,
        csv_path,
    } = config;

    match source {
        BenchSource::Synthetic => {
            let data = toy_vectors(DEFAULT_N, DEFAULT_DIM, 0xA11C_E001);
            let queries = toy_vectors(1, DEFAULT_DIM, 0xA11C_E002);
            if let Err(err) =
                run_benchmark("synthetic", data, DEFAULT_DIM, queries, iters, csv_path)
            {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
        BenchSource::F32 {
            vectors_path,
            queries_path,
            dim,
            query_row,
        } => {
            let data = read_f32_matrix(&vectors_path, dim).unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(2);
            });
            let queries = match queries_path {
                Some(path) => read_f32_matrix(&path, dim).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    std::process::exit(2);
                }),
                None => query_from_corpus(&data, dim, query_row).unwrap_or_else(|err| {
                    eprintln!("{err}");
                    std::process::exit(2);
                }),
            };
            if let Err(err) = run_benchmark(
                &format!("corpus:{}", vectors_path.display()),
                data,
                dim,
                queries,
                iters,
                csv_path,
            ) {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    }
}
