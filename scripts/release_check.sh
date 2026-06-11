#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: scripts/release_check.sh [--quick|--full] [--skip-bench]

Runs the local release-readiness checks used for turbo-graph 0.1.x.

Modes:
  --quick      Rust gate + wheel build + GraphMemory/core Python tests.
  --full       Quick gate plus full Python integration tests with extras.

Options:
  --skip-bench Skip graph_view_bench smoke.

Environment:
  PYTHON       Python executable used to create the check venv (default: python3).

The script does not publish, tag, or mutate git history.
EOF
}

mode="quick"
skip_bench=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      mode="quick"
      ;;
    --full)
      mode="full"
      ;;
    --skip-bench)
      skip_bench=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$mode" != "quick" && "$mode" != "full" ]]; then
  echo "invalid mode: $mode" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
wheelhouse="$repo_root/turbo-graph-python/dist-test"
venv_dir="${TMPDIR:-/tmp}/turbo-graph-release-check-venv"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run cargo fmt --check
run cargo clippy -p turbo-graph --all-targets -- -D warnings
run cargo clippy -p turbo-graph-python --all-targets -- -D warnings
run cargo test -p turbo-graph --release
run cargo package -p turbo-graph --allow-dirty
run cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml

if [[ "$skip_bench" -eq 0 ]]; then
  bench_csv="${TMPDIR:-/tmp}/turbo-graph-view-bench-smoke.csv"
  run cargo run -p turbo-graph --release --example graph_view_bench -- --iters 1 --csv "$bench_csv"
  run cargo run -p turbo-graph --release --example graph_view_bench_summary -- "$bench_csv"
fi

run rm -rf "$venv_dir"
run "$python_bin" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"
run "$venv_python" -m pip install --upgrade pip
run "$venv_python" -m pip install "maturin>=1.12,<2.0" pytest numpy

run rm -rf "$wheelhouse"
run "$venv_python" -m maturin build --release --out "$wheelhouse" --manifest-path turbo-graph-python/Cargo.toml
run "$venv_python" -m maturin sdist --out "${TMPDIR:-/tmp}/turbo-graph-sdist-check" --manifest-path turbo-graph-python/Cargo.toml

wheel_file="$(find "$wheelhouse" -maxdepth 1 -name 'turbo_graph-*.whl' -print -quit)"
if [[ -z "$wheel_file" ]]; then
  echo "wheel build did not produce a turbo_graph wheel" >&2
  exit 1
fi
run "$venv_python" -m pip install --force-reinstall "$wheel_file"

if [[ "$mode" == "quick" ]]; then
  run "$venv_python" -m pytest \
    turbo-graph-python/tests/test_index.py \
    turbo-graph-python/tests/test_id_map.py \
    turbo-graph-python/tests/test_filtering.py \
    turbo-graph-python/tests/test_graph_memory.py \
    -q
else
  run "$venv_python" -m pip install \
    "langchain-core>=0.3" \
    "llama-index-core>=0.11" \
    "haystack-ai>=2.0" \
    "agno>=2.0"
  run "$venv_python" -m pytest turbo-graph-python/tests/ -q
fi

if git ls-files | rg '(^|/)\.DS_Store$|\.whl$|dist-test|(^|/)dist/|\.pytest_cache|target/'; then
  echo "tracked generated artifacts found" >&2
  exit 1
fi

echo
echo "release check passed ($mode)"
