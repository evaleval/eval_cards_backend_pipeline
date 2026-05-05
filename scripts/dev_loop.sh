#!/usr/bin/env bash
# Local end-to-end dev loop: run the pipeline against a small CONFIGS subset,
# then start the frontend (general-eval-card) pointed at output/.
#
# Usage:
#   scripts/dev_loop.sh                # run pipeline, then start frontend
#   scripts/dev_loop.sh pipeline       # run pipeline only
#   scripts/dev_loop.sh frontend       # start frontend against existing output/
#
# Env overrides:
#   CONFIGS                # comma-separated EEE configs (default exercises the
#                          # known-bug surface: GPQA, Tau2, SciCode, Vals AI)
#   PORT                   # frontend port (default 3002)
#   FRONTEND_DIR           # path to general-eval-card repo (default sibling dir)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${FRONTEND_DIR:-$REPO_ROOT/../general-eval-card}"
OUTPUT_DIR="$REPO_ROOT/output"
PORT="${PORT:-3002}"

# Default CONFIGS picks suites that exercise the most-frequent bug surfaces:
#   - llm_stats: GPQA slices, Tau2/Tau-Bench variants, Finance Agent
#   - artificial_analysis_llms: SciCode, Tau2-Bench
#   - openeval: GPQA, broad small-suite coverage
# Override via CONFIGS= for targeted runs.
CONFIGS="${CONFIGS:-llm-stats,artificial-analysis-llms,openeval,hfopenllm_v2}"

run_pipeline() {
  echo "==> Running pipeline (CONFIGS=$CONFIGS, dry-run)"
  cd "$REPO_ROOT"
  CONFIGS="$CONFIGS" \
  uv run \
    --with huggingface_hub --with datasets --with pandas --with pyarrow \
    --with 'eval-entity-resolver @ git+https://github.com/evaleval/evalcard-registry.git#subdirectory=packages/eval-entity-resolver' \
    --no-project python -m scripts.pipeline --dry-run
  echo "==> Pipeline done. Output at $OUTPUT_DIR"
}

run_frontend() {
  if [[ ! -d "$FRONTEND_DIR" ]]; then
    echo "Frontend repo not found at $FRONTEND_DIR" >&2
    echo "Set FRONTEND_DIR to the general-eval-card checkout path." >&2
    exit 1
  fi
  if [[ ! -f "$OUTPUT_DIR/manifest.json" ]]; then
    echo "No output/manifest.json — run the pipeline first" >&2
    exit 1
  fi
  echo "==> Starting frontend on http://localhost:$PORT"
  echo "    LOCAL_PIPELINE_OUTPUT=$OUTPUT_DIR"
  cd "$FRONTEND_DIR"
  exec env \
    DATA_BACKEND=duckdb \
    LOCAL_PIPELINE_OUTPUT="$OUTPUT_DIR" \
    HF_DATA_LOCAL_DIR="$OUTPUT_DIR" \
    HF_DATA_OFFLINE=1 \
    PORT="$PORT" \
    pnpm dev
}

case "${1:-all}" in
  pipeline) run_pipeline ;;
  frontend) run_frontend ;;
  all)      run_pipeline && run_frontend ;;
  *)        echo "Usage: $0 [pipeline|frontend|all]" >&2; exit 2 ;;
esac
