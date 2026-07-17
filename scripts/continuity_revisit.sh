#!/usr/bin/env bash
# Continuity-first revisit: NO --clear-memory. Load existing store, re-run bridge prompts,
# emit CONT cards. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/continuity_revisit_${STAMP}"
mkdir -p "$OUT"

source "$ROOT/scripts/cuda_env.sh"
# +fp16 helps some aarch64 hosts; leave unset elsewhere (x86_64, etc.)
if [[ -z "${RUSTFLAGS:-}" ]] && [[ "$(uname -m)" == "aarch64" ]]; then
  export RUSTFLAGS="-C target-feature=+fp16"
fi

[[ -x "$BIN" ]] || { echo "missing $BIN — cargo build --release"; exit 1; }
[[ -f "$MODEL" ]] || { echo "missing $MODEL"; exit 1; }
[[ -f data/splat_memory.tct ]] || { echo "missing data/splat_memory.tct — mint once with prefill bridge first"; exit 1; }

echo "out=$OUT"
echo "store: $(ls -la data/splat_memory.tct data/splat_memory.tct.json 2>/dev/null | tr '\n' ' ')"
python3 scripts/list_bridges.py | tee "$OUT/bridges_before.txt"

run_prompt() {
  local id="$1"
  local prompt="$2"
  echo
  echo "========== $id =========="
  echo "prompt: $prompt"
  set +e
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$prompt" \
    --tokens "$TOKENS" \
    --export-tct data/splat_memory.tct \
    2>&1 | tee "$OUT/${id}.stdout"
  local ec=${PIPESTATUS[0]}
  set -e
  local jl
  jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "$jl" ]]; then
    cp -f "$jl" "$OUT/${id}.jsonl"
  fi
  cp -f data/splat_memory.tct.json "$OUT/${id}.tct.json" 2>/dev/null || true
  echo "--- CONT card $id ---"
  python3 scripts/continuity_card.py "$OUT/${id}.jsonl" \
    --label "$id: $prompt" \
    --out "$OUT/${id}.kpi.json" \
    --tct-json data/splat_memory.tct.json | tee -a "$OUT/cards.txt"
  echo "exit=$ec"
  return 0
}

# Bridge revisits (warm expected if residual coupling works)
run_prompt R1 "Explain the Physics of Friendship in one short paragraph."
run_prompt R2 "Write three short tips for debugging a CUDA kernel."
# Novel prompt (cold expected on first visit)
run_prompt COLD "List three prime numbers greater than 50."

python3 scripts/list_bridges.py | tee "$OUT/bridges_after.txt"
cp -f data/splat_memory.tct "$OUT/splat_memory.tct" 2>/dev/null || true
cp -f data/splat_memory.tct.json "$OUT/splat_memory.tct.json" 2>/dev/null || true

{
  echo "# Continuity revisit $STAMP"
  echo
  echo "## Cards"
  cat "$OUT/cards.txt" 2>/dev/null || true
  echo
  echo "## Bridges after"
  cat "$OUT/bridges_after.txt" 2>/dev/null || true
} >"$OUT/CONTINUITY_CARD.md"

echo
echo "======== SUMMARY ========"
cat "$OUT/cards.txt" 2>/dev/null || true
echo "done: $OUT"
echo "card: $OUT/CONTINUITY_CARD.md"
