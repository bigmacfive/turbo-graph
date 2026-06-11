# turbo-graph

`turbo-graph` is a Rust crate for turbovec-compatible TurboQuant vector search
plus graph and metadata memory for constrained RAG workloads.

Use the plain vector APIs when you need the shared TurboQuant core:

- 2-4 bit vector compression.
- Train-free ingest.
- SIMD search with optional kernel-level slot masks.
- `.tv` / `.tvim` persistence.

Use `GraphMemoryIndex` when retrieval constraints are part of the product:

- Weighted graph neighborhood expansion before vector search.
- Tag, source, and time-window views.
- Reusable `SlotMask` compilation for repeated constrained queries.
- Candidate-list intersection for upstream BM25, SQL, or ACL systems.
- Graph-aware rerank and explain telemetry.
- Cache stats and bounded cache trimming for long-running services.

## Install

```bash
cargo add turbo-graph
```

## Vector Search

```rust,no_run
use turbo_graph::TurboQuantIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vectors = vec![0.0_f32; 1536 * 1000];
    let queries = vec![0.0_f32; 1536];

    let mut index = TurboQuantIndex::new(1536, 4)?;
    index.add(&vectors);
    index.prepare();

    let results = index.search(&queries, 10);
    println!("effective_k={}", results.k);
    Ok(())
}
```

## Graph Memory

```rust,no_run
use turbo_graph::{GraphMemoryIndex, GraphSearchPreset, MemoryRecord};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let flat_vectors = vec![0.0_f32; 1536 * 2];
    let query = vec![0.0_f32; 1536];

    let mut memory = GraphMemoryIndex::new(1536, 4)?;
    memory.add_records(
        &flat_vectors,
        vec![
            MemoryRecord::new(1001, "Architecture note", ["architecture"])
                .with_source("docs")
                .with_timestamp_ms(1_700_000_000_000),
            MemoryRecord::new(1002, "Retrieval cache note", ["architecture", "cache"])
                .with_source("docs")
                .with_timestamp_ms(1_700_000_010_000),
        ],
    )?;
    memory.link_bidirectional(1001, 1002, 0.8)?;

    let report = memory.explain_graph_search_with_preset(
        &query,
        10,
        &[1001],
        GraphSearchPreset::balanced(),
        &["architecture"],
        &["docs"],
        None,
        None,
    );
    println!("hits={} selected_slots={}", report.hits.len(), report.plan.selected_slots);
    Ok(())
}
```

## When to Choose turbo-graph

Choose `turbovec` or the plain `TurboQuantIndex`/`IdMapIndex` surface when
filters are light and cheap to construct.

Choose `turbo-graph` when the expensive work is repeatedly compiling and
explaining `graph ∩ tag ∩ source ∩ time ∩ candidates` before vector search.

## Project

- Repository: <https://github.com/bigmacfive/turbo-graph>
- Python package: `pip install turbo-graph`
- API docs and comparison notes live in the repository `docs/` directory.

## License

MIT.
