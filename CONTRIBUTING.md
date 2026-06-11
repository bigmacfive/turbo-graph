# Contributing

Thanks for your interest in turbo-graph. The project is open to issues,
documentation fixes, benchmarks, integrations, and code contributions.

## Workflow

1. **Open an issue first for non-trivial work.** Describe the bug, feature,
   documentation gap, or performance question with enough context to reproduce
   or evaluate it.
2. **Use a pull request for focused changes.** Small documentation fixes can go
   straight to PR. Larger API, performance, packaging, or integration changes
   should link to an issue so reviewers can understand the intent.
3. **Keep the PR narrow.** One logical change per PR is much easier to review
   and benchmark.

Maintainers review and merge to `main`. Please be patient with review on
performance-sensitive changes; this project has a small core and correctness is
more important than fast merges.

## What Makes A Good Contribution

The contributions that move turbo-graph forward are clear and evidence-backed:
a sharp bug report, a reproducible benchmark, a failing test, a precise docs
gap, or an implementation that preserves TurboQuant core parity while improving
the graph/metadata layer.

When proposing changes, call out whether they affect:

- The turbovec-compatible TurboQuant core.
- Python bindings or package metadata.
- Framework integrations.
- Graph memory, mask planning, cache behavior, or explain telemetry.
- Documentation, release, or repository hygiene.

## Commit and PR conventions

- **One logical change per PR.** Refactors get their own PR, separate from feature work.
- **Commit messages:** short imperative title, body explaining *why* (the *what* is in the diff). Multi-line bodies should preserve formatting — use a HEREDOC if writing from the shell.
- **PRs reference their issue** with `Closes #N` and include a test plan.
- **AI-assisted commits are welcome** when the final PR is reviewed by a human.
  If your commit uses `Co-Authored-By:` trailers, leave them intact.

## Integration contributions

If you're adding or modifying an integration (LangChain, LlamaIndex, Haystack, Agno, or a new framework), structurally compare against the canonical in-tree reference store (`InMemoryVectorStore`, `SimpleVectorStore`, `InMemoryDocumentStore`, etc.) for that framework. The wrappers should match the reference's surface and idioms — that's the bar for a drop-in replacement.

## Build, test, bench

See the [Install](README.md#install) and [Run benchmarks](README.md#run-benchmarks) sections of the README.

Before opening a PR that touches Rust core, Python bindings, packaging, docs, or graph-memory behavior, run the closest useful subset of this release gate:

```bash
scripts/release_check.sh --quick
```

For integration or packaging-heavy changes, run the full gate:

```bash
scripts/release_check.sh --full
```

The script does not publish, tag, or mutate git history. If you need to run the commands manually, use the same sequence:

```bash
cargo fmt --check
cargo clippy -p turbo-graph --all-targets -- -D warnings
cargo clippy -p turbo-graph-python --all-targets -- -D warnings
cargo test -p turbo-graph --release
cargo package -p turbo-graph --allow-dirty
cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml

cd turbo-graph-python
python -m maturin build --release --out dist-test
python -m pytest tests/test_index.py tests/test_id_map.py tests/test_filtering.py tests/test_graph_memory.py -q
```

If you changed framework integrations, run the full Python suite with the extras
installed. From the repository root:

```bash
python -m pip install "langchain-core>=0.3" "llama-index-core>=0.11" "haystack-ai>=2.0" "agno>=2.0"
python -m pytest turbo-graph-python/tests/ -q
```

If you changed graph-memory planning, candidate handling, cache policy, or explain telemetry, include at least one of:

- `cargo run -p turbo-graph --release --example graph_view_bench -- --iters 3 --csv /tmp/graph-view-bench.csv`
- `python turbo-graph-python/examples/graph_memory_rag.py`

For recall or speed changes, attach before/after numbers and describe whether the change affects TurboQuant core parity, the graph/metadata layer, or both.
