#!/usr/bin/env bash
#
# Run the meeting-assistant benchmark against a LOCAL Ollama server instead of a
# hosted vLLM instance. Ollama exposes an OpenAI-compatible API, which is exactly
# what the agents client expects ({BASE_URL}/v1/chat/completions).
#
# Notes:
#   * Ollama has no vLLM /metrics endpoint, so we pass --no-server-metrics
#     (host CPU/RAM are still sampled via psutil).
#   * Ollama does not report usage.prompt_tokens_details.cached_tokens, so the
#     per-request prefix-cache ratio is 0 and the runner prints a P1.2 warning.
#     This is expected on Ollama; use a vLLM host to exercise prefix caching.
#   * Output is written to results/ollama/runs/ so a local sweep never clobbers
#     the canonical (TPU) corpus in results/runs/.
#
# Usage:
#   scripts/run_ollama.sh                       # small, 3 modes, 3 repeats, warmup 1
#   SIZES="small medium" REPEATS=2 scripts/run_ollama.sh
#   OLLAMA_MODEL=qwen3.5:9b scripts/run_ollama.sh
#
set -euo pipefail

# --- config (override via env) ------------------------------------------------
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
SIZES="${SIZES:-small}"
MODES="${MODES:-sequential parallel langgraph}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
PYTHON="${PYTHON:-python3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-$ROOT/results/ollama/runs}"

# --- point the client at Ollama's OpenAI-compatible API -----------------------
export VLLM_BASE_URL="$OLLAMA_HOST"
export MODEL_NAME="$OLLAMA_MODEL"
# Localhost must bypass any corporate proxy, or httpx will try to tunnel it.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true

echo "==> Ollama:  $OLLAMA_HOST   model: $OLLAMA_MODEL"
echo "==> Sweep:   sizes=[$SIZES] modes=[$MODES] repeats=$REPEATS warmup=$WARMUP"
echo "==> Output:  $RUNS_DIR"

# --- preflight: is the model available? --------------------------------------
if ! curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "ERROR: cannot reach Ollama at $OLLAMA_HOST. Is 'ollama serve' running?" >&2
  exit 1
fi
if ! curl -fsS "$OLLAMA_HOST/api/tags" | grep -q "\"$OLLAMA_MODEL\""; then
  echo "NOTE: model '$OLLAMA_MODEL' not found in 'ollama list'." >&2
  echo "      Pull it first:  ollama pull $OLLAMA_MODEL" >&2
  exit 1
fi

cd "$ROOT"

# --- generate transcripts (offline synthetic) for the requested sizes --------
# shellcheck disable=SC2086
"$PYTHON" scripts/generate_transcripts.py --offline --sizes $SIZES

# --- run the sweep ------------------------------------------------------------
# shellcheck disable=SC2086
"$PYTHON" -m bench.runner \
  --sizes $SIZES \
  --modes $MODES \
  --repeats "$REPEATS" \
  --warmup "$WARMUP" \
  --no-server-metrics \
  --runs-dir "$RUNS_DIR"

# --- print a quick summary (mean + 95% CI per mode) --------------------------
"$PYTHON" - "$RUNS_DIR" <<'PY'
import sys
from bench.metrics import aggregate, summarize
runs_dir = sys.argv[1]
df = aggregate(runs_dir)
if df.empty:
    print("No runs to summarize."); sys.exit(0)
s = summarize(df)
cols = ["transcript_size", "mode", "n_runs", "e2e_wall_s", "e2e_wall_s_ci95",
        "total_tokens", "revisions"]
cols = [c for c in cols if c in s.columns]
print("\n=== summary (mean over repeats; e2e_wall_s_ci95 = 95% CI half-width) ===")
print(s[cols].to_string(index=False))
PY
