#!/usr/bin/env bash
# Near-vanilla A/B: Unsloth vs bartowski Gemma 4 31B IT Q4_K_M
# Same config, tokenizer, prompt — compare surface personality / decode.
#
# Usage:
#   ./scripts/ab_gemma4_unsloth_vs_bart.sh
#   PROMPT="..." TOKENS=48 ./scripts/ab_gemma4_unsloth_vs_bart.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# NVIDIA: hydro generation requires Device::new_cuda(0)
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — need NVIDIA driver for this A/B." >&2
  exit 1
fi
echo "[*] NVIDIA:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || nvidia-smi | head -12
echo "[*] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_HOME=$CUDA_HOME"

BIN="${BIN:-./target/release/hydrodynamic-swarm}"
CONFIG="${CONFIG:-configs/gemma4/config.gemma4_greedy.toml}"
TOKENIZER="${TOKENIZER:-data/google/gemma4_assets/tokenizer.json}"
PROMPT="${PROMPT:-Say hi in one short sentence.}"
TOKENS="${TOKENS:-40}"
OUT_DIR="${OUT_DIR:-logs/ab_g4_31b_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

UNSLOTH="${UNSLOTH:-data/google/unsloth_gemma-4-31B-it-Q4_K_M.gguf}"
BART="${BART:-data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf}"

run_one() {
  local label="$1"
  local model="$2"
  local out="$OUT_DIR/${label}.txt"
  echo "=== A/B arm: $label (CUDA) ==="
  echo "model=$model"
  if [[ ! -e "$model" ]]; then
    echo "MISSING model: $model" | tee "$out"
    return 1
  fi
  # Fail the arm if CUDA did not bind (hydro prints this line on success)
  if ! "$BIN" \
    --config "$CONFIG" \
    --model "$model" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --clear-memory \
    --no-endocrine \
    2>&1 | tee "$out" | tee /dev/stderr | grep -q 'Using CUDA GPU'; then
    # still have full log in $out; check after
    :
  fi
  if ! grep -q 'Using CUDA GPU' "$out"; then
    echo "ERROR: $label did not report CUDA GPU — abort A/B." | tee -a "$out" >&2
    return 1
  fi
  echo "--- decoded extract ($label) ---"
  sed -n '/Full Decoded Output/,/Phase 5/p' "$out" | head -20 || true
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
}

echo "A/B out dir: $OUT_DIR"
echo "prompt: $PROMPT"
echo "config: $CONFIG tokens: $TOKENS"
run_one unsloth "$UNSLOTH"
run_one bartowski "$BART"

{
  echo "# Gemma 4 31B Unsloth vs bartowski (near-vanilla)"
  echo "date_utc: $(date -u -Iseconds)"
  echo "prompt: $PROMPT"
  echo "tokens: $TOKENS"
  echo "config: $CONFIG"
  echo
  echo "## Unsloth decoded"
  sed -n '/Full Decoded Output/,/Phase 5/p' "$OUT_DIR/unsloth.txt" 2>/dev/null | head -30
  echo
  echo "## bartowski decoded"
  sed -n '/Full Decoded Output/,/Phase 5/p' "$OUT_DIR/bartowski.txt" 2>/dev/null | head -30
} | tee "$OUT_DIR/COMPARE.md"

echo "Done. Compare: $OUT_DIR/COMPARE.md"
