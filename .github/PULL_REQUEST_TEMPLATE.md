<!--
Thanks for the contribution!

The expected workflow is: open an issue describing the change and your
proposed approach, get a 👍 or design discussion, then open this PR
referencing the issue. See CONTRIBUTING.md for the narrow exceptions
(typos, one-line obvious bug fixes, docs-only changes).
-->

## Related issue

Closes #<!-- issue number -->

## Summary

<!-- Bullets describing what changed. Short is good. -->

## Motivation

<!-- Why this change? If the issue already covers this, a one-line pointer is fine. -->

## Test plan

<!--
What you ran to verify. Check the ones you actually verified; add others as relevant.
For recall or speed changes, include before/after numbers.
-->

- [ ] `cargo test -p turbo-graph --release` passes
- [ ] `cargo clippy -p turbo-graph --all-targets -- -D warnings` passes
- [ ] `cargo clippy -p turbo-graph-python --all-targets -- -D warnings` passes
- [ ] `cargo package -p turbo-graph --allow-dirty` passes
- [ ] downstream smoke passes: `cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml`
- [ ] Python wheel builds: `cd turbo-graph-python && python -m maturin build --release --out dist-test`
- [ ] `pytest turbo-graph-python/tests/` passes
- [ ] or `scripts/release_check.sh --quick` / `--full` passes
- [ ] GraphMemory changes include candidate/cache/explain evidence or `graph_view_bench` numbers
