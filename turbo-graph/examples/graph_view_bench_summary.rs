//! Summarize `graph_view_bench --csv` output.
//!
//! Run:
//!   cargo run -p turbo-graph --example graph_view_bench_summary -- graph-view-bench.csv
//!   cargo run -p turbo-graph --example graph_view_bench_summary -- before.csv after.csv

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;

struct BenchCsv {
    path: PathBuf,
    headers: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

#[derive(Clone)]
struct ComparisonRow {
    file: String,
    mode: String,
    global_ms: Option<f64>,
    balanced_graph_ratio: Option<f64>,
    balanced_batch_ratio: Option<f64>,
    full_recall_fetch: Option<usize>,
    full_recall_post_direct: Option<f64>,
}

fn usage() -> &'static str {
    "usage:
  cargo run -p turbo-graph --example graph_view_bench_summary -- graph-view-bench.csv
  cargo run -p turbo-graph --example graph_view_bench_summary -- before.csv after.csv"
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return;
    }

    let mut comparisons = Vec::new();
    for arg in args {
        match BenchCsv::read(PathBuf::from(&arg)) {
            Ok(csv) => {
                let comparison = print_summary(&csv);
                comparisons.push(comparison);
            }
            Err(err) => {
                eprintln!("{arg}: {err}");
                std::process::exit(2);
            }
        }
    }

    if comparisons.len() > 1 {
        print_comparison(&comparisons);
    }
}

impl BenchCsv {
    fn read(path: PathBuf) -> Result<Self, String> {
        let text = fs::read_to_string(&path).map_err(|err| format!("failed to read CSV: {err}"))?;
        let mut records = parse_csv_records(&text)?;
        if records.is_empty() {
            return Err("CSV has no header".to_string());
        }
        let header = records.remove(0);
        let headers = header
            .into_iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect();
        Ok(Self {
            path,
            headers,
            rows: records,
        })
    }

    fn value<'a>(&self, row: &'a [String], name: &str) -> &'a str {
        self.headers
            .get(name)
            .and_then(|&idx| row.get(idx))
            .map(String::as_str)
            .unwrap_or("")
    }

    fn number(&self, row: &[String], name: &str) -> Option<f64> {
        let value = self.value(row, name);
        if value.is_empty() {
            None
        } else {
            value.parse::<f64>().ok()
        }
    }

    fn integer(&self, row: &[String], name: &str) -> Option<usize> {
        let value = self.value(row, name);
        if value.is_empty() {
            None
        } else {
            value.parse::<usize>().ok()
        }
    }

    fn rows_for<'a>(&'a self, phase: &str) -> Vec<&'a Vec<String>> {
        self.rows
            .iter()
            .filter(|row| self.value(row, "phase") == phase)
            .collect()
    }

    fn find<'a>(&'a self, phase: &str, case: &str) -> Option<&'a Vec<String>> {
        self.rows
            .iter()
            .find(|row| self.value(row, "phase") == phase && self.value(row, "case") == case)
    }
}

fn print_summary(csv: &BenchCsv) -> ComparisonRow {
    let global = csv.find("global", "global");
    let mode = global
        .map(|row| csv.value(row, "mode").to_string())
        .or_else(|| {
            csv.rows
                .first()
                .map(|row| csv.value(row, "mode").to_string())
        })
        .unwrap_or_default();
    let global_ms = global.and_then(|row| csv.number(row, "ms_per_iter"));

    println!();
    println!("file={}", csv.path.display());
    println!(
        "mode={} vectors={} dim={} queries={} k={} iters={} global_ms={}",
        value_or(global.map(|row| csv.value(row, "mode")), "-"),
        value_or(global.map(|row| csv.value(row, "vectors")), "-"),
        value_or(global.map(|row| csv.value(row, "dim")), "-"),
        value_or(global.map(|row| csv.value(row, "queries")), "-"),
        value_or(global.map(|row| csv.value(row, "k")), "-"),
        value_or(global.map(|row| csv.value(row, "iters")), "-"),
        fmt_opt(global_ms),
    );

    print_selectivity(csv);
    print_build_and_compile(csv);
    print_presets(csv);
    print_constrained(csv);
    let (full_recall_fetch, full_recall_post_direct) = print_post_filter(csv);

    ComparisonRow {
        file: csv.path.display().to_string(),
        mode,
        global_ms,
        balanced_graph_ratio: csv
            .find("preset", "balanced:graph")
            .and_then(|row| csv.number(row, "ratio_to_global")),
        balanced_batch_ratio: csv
            .find("preset", "balanced:batch-rerank")
            .and_then(|row| csv.number(row, "ratio_to_global")),
        full_recall_fetch,
        full_recall_post_direct,
    }
}

fn print_selectivity(csv: &BenchCsv) {
    let mut grouped: BTreeMap<String, Vec<&Vec<String>>> = BTreeMap::new();
    for row in csv.rows_for("selectivity") {
        grouped
            .entry(csv.value(row, "selectivity").to_string())
            .or_default()
            .push(row);
    }
    if grouped.is_empty() {
        return;
    }

    println!("selectivity:");
    println!(
        "{:>8} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "sel%", "allowed", "blocks", "slot/global", "graph/global", "graph/slot"
    );
    for (selectivity, rows) in grouped {
        let slot = find_case(csv, &rows, "slot_mask");
        let graph = find_case(csv, &rows, "graph_memory");
        let allowed = slot.or(graph).and_then(|row| csv.integer(row, "allowed"));
        let blocks = slot
            .or(graph)
            .and_then(|row| csv.integer(row, "active_blocks"));
        let slot_ratio = slot.and_then(|row| csv.number(row, "ratio_to_global"));
        let graph_ratio = graph.and_then(|row| csv.number(row, "ratio_to_global"));
        let graph_over_slot = match (slot_ratio, graph_ratio) {
            (Some(slot_ratio), Some(graph_ratio)) if slot_ratio > 0.0 => {
                Some(graph_ratio / slot_ratio)
            }
            _ => None,
        };
        println!(
            "{:>8} {:>8} {:>8} {:>12} {:>12} {:>12}",
            pct_str(&selectivity),
            fmt_usize(allowed),
            fmt_usize(blocks),
            fmt_opt(slot_ratio),
            fmt_opt(graph_ratio),
            fmt_opt(graph_over_slot),
        );
    }
}

fn print_build_and_compile(csv: &BenchCsv) {
    let mut selectivities = BTreeMap::new();
    for row in csv.rows_for("mask_build") {
        selectivities
            .entry(csv.value(row, "selectivity").to_string())
            .or_insert_with(Vec::new)
            .push(row);
    }
    for row in csv.rows_for("view_compile") {
        selectivities
            .entry(csv.value(row, "selectivity").to_string())
            .or_insert_with(Vec::new)
            .push(row);
    }
    if selectivities.is_empty() {
        return;
    }

    println!("build_compile:");
    println!(
        "{:>8} {:>8} {:>8} {:>12} {:>12} {:>12}",
        "sel%", "allowed", "blocks", "mask_ms", "cold_ms", "warm_ms"
    );
    for (selectivity, rows) in selectivities {
        let mask = find_case(csv, &rows, "slot_mask_from_slots");
        let cold = find_case(csv, &rows, "graph_view_cold");
        let warm = find_case(csv, &rows, "graph_view_warm");
        let representative = mask.or(cold).or(warm);
        println!(
            "{:>8} {:>8} {:>8} {:>12} {:>12} {:>12}",
            pct_str(&selectivity),
            fmt_usize(representative.and_then(|row| csv.integer(row, "allowed"))),
            fmt_usize(representative.and_then(|row| csv.integer(row, "active_blocks"))),
            fmt_opt(mask.and_then(|row| csv.number(row, "ms_per_iter"))),
            fmt_opt(cold.and_then(|row| csv.number(row, "ms_per_iter"))),
            fmt_opt(warm.and_then(|row| csv.number(row, "ms_per_iter"))),
        );
    }
}

fn print_presets(csv: &BenchCsv) {
    let presets = ["low_latency", "balanced", "broad"];
    if !presets
        .iter()
        .any(|preset| csv.find("preset", &format!("{preset}:graph")).is_some())
    {
        return;
    }

    println!("presets:");
    println!(
        "{:>12} {:>8} {:>8} {:>8} {:>14} {:>12} {:>12} {:>12} {:>12}",
        "preset",
        "selected",
        "blocks",
        "prefetch",
        "packed/glob",
        "graph/glob",
        "batch/glob",
        "graph/packed",
        "batch/graph"
    );
    for preset in presets {
        let packed = csv.find("preset", &format!("{preset}:packed"));
        let graph = csv.find("preset", &format!("{preset}:graph"));
        let batch = csv.find("preset", &format!("{preset}:batch-rerank"));
        let selected = graph
            .or(packed)
            .and_then(|row| csv.integer(row, "selected_slots"));
        let blocks = graph
            .or(packed)
            .and_then(|row| csv.integer(row, "active_blocks"));
        let prefetch = graph
            .or(packed)
            .and_then(|row| csv.integer(row, "prefetch_k"));
        let packed_ratio = packed.and_then(|row| csv.number(row, "ratio_to_global"));
        let graph_ratio = graph.and_then(|row| csv.number(row, "ratio_to_global"));
        let batch_ratio = batch.and_then(|row| csv.number(row, "ratio_to_global"));
        let overhead = match (packed_ratio, graph_ratio) {
            (Some(packed_ratio), Some(graph_ratio)) if packed_ratio > 0.0 => {
                Some(graph_ratio / packed_ratio)
            }
            _ => None,
        };
        let batch_over_graph = match (batch_ratio, graph_ratio) {
            (Some(batch_ratio), Some(graph_ratio)) if graph_ratio > 0.0 => {
                Some(batch_ratio / graph_ratio)
            }
            _ => None,
        };
        println!(
            "{:>12} {:>8} {:>8} {:>8} {:>14} {:>12} {:>12} {:>12} {:>12}",
            preset,
            fmt_usize(selected),
            fmt_usize(blocks),
            fmt_usize(prefetch),
            fmt_opt(packed_ratio),
            fmt_opt(graph_ratio),
            fmt_opt(batch_ratio),
            fmt_opt(overhead),
            fmt_opt(batch_over_graph),
        );
    }
}

fn print_constrained(csv: &BenchCsv) {
    let rows = csv.rows_for("constrained");
    if rows.is_empty() {
        return;
    }

    println!("constrained:");
    println!(
        "{:>22} {:>12} {:>12} {:>12}",
        "case", "ms/iter", "vs_cached", "blocks_skip"
    );
    for case in ["cached_mask_search", "rebuild_view", "candidate_intersect"] {
        if let Some(row) = rows
            .iter()
            .copied()
            .find(|row| csv.value(row, "case") == case)
        {
            println!(
                "{:>22} {:>12} {:>12} {:>12}",
                case,
                fmt_opt(csv.number(row, "ms_per_iter")),
                fmt_opt(csv.number(row, "ratio_to_global")),
                fmt_usize(csv.integer(row, "blocks_skipped")),
            );
        }
    }
}

fn print_post_filter(csv: &BenchCsv) -> (Option<usize>, Option<f64>) {
    let direct = csv.find("post_filter", "direct_mask");
    let overfetch: Vec<&Vec<String>> = csv
        .rows_for("post_filter")
        .into_iter()
        .filter(|row| csv.value(row, "case") == "global_overfetch")
        .collect();
    if direct.is_none() && overfetch.is_empty() {
        return (None, None);
    }

    let direct_ms = direct.and_then(|row| csv.number(row, "ms_per_iter"));
    let full = overfetch
        .iter()
        .filter_map(|row| {
            let recall = csv.number(row, "recall")?;
            let fetch = csv.integer(row, "fetch_k")?;
            let ratio = csv.number(row, "ratio_to_global")?;
            Some((recall, fetch, ratio))
        })
        .filter(|(recall, _, _)| *recall >= 0.999)
        .min_by_key(|(_, fetch, _)| *fetch);

    println!("post_filter:");
    println!(
        "  direct_mask_ms={} first_full_recall_fetch={} first_full_recall_post/direct={}",
        fmt_opt(direct_ms),
        full.map(|(_, fetch, _)| fetch.to_string())
            .unwrap_or_else(|| "-".to_string()),
        full.map(|(_, _, ratio)| fmt_opt(Some(ratio)))
            .unwrap_or_else(|| "-".to_string()),
    );

    (
        full.map(|(_, fetch, _)| fetch),
        full.map(|(_, _, ratio)| ratio),
    )
}

fn print_comparison(rows: &[ComparisonRow]) {
    println!();
    println!("comparison:");
    println!(
        "{:>28} {:>16} {:>10} {:>16} {:>16} {:>16} {:>16}",
        "file",
        "mode",
        "global_ms",
        "balanced_graph",
        "balanced_batch",
        "full_recall_k",
        "full_post/direct"
    );
    for row in rows {
        println!(
            "{:>28} {:>16} {:>10} {:>16} {:>16} {:>16} {:>16}",
            trim_start(&row.file, 28),
            trim_start(&row.mode, 16),
            fmt_opt(row.global_ms),
            fmt_opt(row.balanced_graph_ratio),
            fmt_opt(row.balanced_batch_ratio),
            row.full_recall_fetch
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_string()),
            fmt_opt(row.full_recall_post_direct),
        );
    }
}

fn parse_csv_records(input: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' if field.is_empty() => {
                in_quotes = true;
            }
            ',' if !in_quotes => {
                row.push(std::mem::take(&mut field));
            }
            '\n' if !in_quotes => {
                row.push(std::mem::take(&mut field));
                if row.iter().any(|value| !value.is_empty()) {
                    records.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            '\r' if !in_quotes => {}
            other => field.push(other),
        }
    }

    if in_quotes {
        return Err("unterminated quoted CSV field".to_string());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        if row.iter().any(|value| !value.is_empty()) {
            records.push(row);
        }
    }
    Ok(records)
}

fn find_case<'a>(
    csv: &BenchCsv,
    rows: &'a [&'a Vec<String>],
    case: &str,
) -> Option<&'a Vec<String>> {
    rows.iter()
        .copied()
        .find(|row| csv.value(row, "case") == case)
}

fn value_or<'a>(value: Option<&'a str>, fallback: &'a str) -> &'a str {
    value.filter(|value| !value.is_empty()).unwrap_or(fallback)
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn pct_str(value: &str) -> String {
    value
        .parse::<f64>()
        .map(|value| format!("{:.2}", value * 100.0))
        .unwrap_or_else(|_| "-".to_string())
}

fn trim_start(value: &str, max_len: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_len {
        value.to_string()
    } else {
        value
            .chars()
            .skip(char_count - max_len + 3)
            .collect::<String>()
            .chars()
            .fold(String::from("..."), |mut out, ch| {
                out.push(ch);
                out
            })
    }
}
