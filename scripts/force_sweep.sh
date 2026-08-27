#!/usr/bin/env bash
# Force-sweep ablation: test multiple force_scale values
# Measures output quality vs physics metrics to find sweet spot
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/force_sweep_${STAMP}"
mkdir -p "$OUT"

source "$ROOT/scripts/cuda_env.sh"

if [[ ! -x "$BIN" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi
[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

# Sweep values for force_scale
FORCE_VALUES=(0.0 0.1 0.25 0.5 0.75 1.0 1.5 2.0)
echo "Force sweep values: ${FORCE_VALUES[*]}"

# Run baseline first (force=0, no physics)
echo "=== BASELINE (force=0, no physics) ==="
run_one() {
  local id="$1"; shift
  local force="$1"; shift
  echo
  echo "========== RUN $id (force=$force) =========="
  echo "  args: $*"
  set +e
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --config "$ROOT/configs/ablation/config_isolation_baseline.toml" \
    --force-scale "$force" \
    2>&1 | tee "$OUT/${id}.stdout"
  local ec=${PIPESTATUS[0]}
  set -e
  # Capture jsonl if available
  local jl
  jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "$jl" ]]; then
    cp -f "$jl" "$OUT/${id}.jsonl"
  fi
  echo "  exit=$ec → $OUT/${id}.*"
}

# Run baseline
run_one BASELINE 0.0

# Run sweep
for force in "${FORCE_VALUES[@]}"; do
  run_one "F${force//./_}" "$force"
done

echo
echo "=== SWEEP COMPLETE ==="
echo "Output dir: $OUT"
echo "Files: $(ls -1 "$OUT" | wc -l) files"
