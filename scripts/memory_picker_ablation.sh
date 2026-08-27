#!/usr/bin/env bash
# Soft vs ranked memory picker ablation (same prompt).
# A: soft clear-mint  B: soft reload  C: ranked reload
# Expect: C logs memory_force_mode=ranked and memory_ranked true on some steps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/memory_picker_${STAMP}"
mkdir -p "$OUT"

[[ -x "$BIN" ]] || cargo build --release --bin hydrodynamic-swarm
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 1; }

run() {
  local id="$1" cfg="$2"; shift 2
  echo "========== $id =========="
  "$BIN" --config "$cfg" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --tokens "$TOKENS" --no-endocrine --test "$@" \
    2>&1 | tee "$OUT/${id}.log" | tail -25
}

run A_soft_mint config.example.toml --clear-memory
run B_soft_reload config.example.toml
run C_ranked_reload configs/profiles/config.memory_ranked.toml

echo
echo "Logs: $OUT"
echo "JSONL: newest under logs/*gemma3*.jsonl — jq memory_ranked / memory_force_mode"
echo "Done."
