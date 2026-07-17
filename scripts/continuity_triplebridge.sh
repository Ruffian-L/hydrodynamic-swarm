#!/usr/bin/env bash
# Three-basin continuity: A → B → C → A without --clear-memory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/continuity_triplebridge_${STAMP}"
mkdir -p "$OUT"

export PATH="${CUDA_HOME:-/usr/local/cuda-13.3}/bin:${PATH:-}"
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13010}"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda-13.3}/lib64:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS="${RUSTFLAGS:--C target-feature=+fp16}"

A="${PROMPT_A:-Explain the Physics of Friendship in one short paragraph.}"
B="${PROMPT_B:-Write three short tips for debugging a CUDA kernel.}"
C="${PROMPT_C:-List three prime numbers greater than 50.}"

[[ -x "$BIN" ]] || { echo "missing $BIN"; exit 1; }
[[ -f data/splat_memory.tct ]] || { echo "missing data/splat_memory.tct"; exit 1; }

echo "out=$OUT"
python3 scripts/list_bridges.py | tee "$OUT/bridges_before.txt"
: >"$OUT/cards.txt"

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
  [[ -n "$jl" ]] && cp -f "$jl" "$OUT/${id}.jsonl"
  cp -f data/splat_memory.tct.json "$OUT/${id}.tct.json" 2>/dev/null || true
  python3 scripts/continuity_card.py "$OUT/${id}.jsonl" \
    --label "$id" \
    --out "$OUT/${id}.kpi.json" | tee -a "$OUT/cards.txt"
  echo "exit=$ec"
}

run_prompt A1 "$A"
run_prompt B1 "$B"
run_prompt C1 "$C"
run_prompt A2 "$A"

python3 scripts/list_bridges.py | tee "$OUT/bridges_after.txt"
cp -f data/splat_memory.tct "$OUT/splat_memory.tct"
cp -f data/splat_memory.tct.json "$OUT/splat_memory.tct.json" 2>/dev/null || true

python3 - <<PY | tee "$OUT/return_verdict.txt"
import json
from pathlib import Path
out = Path("$OUT")

def load(i):
    p = out / f"{i}.kpi.json"
    return json.loads(p.read_text()) if p.exists() else {}

a1, b1, c1, a2 = load("A1"), load("B1"), load("C1"), load("A2")

def fmt(d, name):
    return f"{name}: status={d.get('status')} nearest={d.get('nearest_min')} pot={d.get('pot_max')} gain_max={d.get('gain_max')}"

ok = a2.get("status") == "WARM" and (a2.get("nearest_min") or 999) < 200
if isinstance(a2.get("pot_max"), (int, float)) and a2["pot_max"] < 0.1:
    ok = ok and (a2.get("nearest_min") or 999) < 80
gains = a2.get("bridge_gains") or {}
# all pleasure bridges should keep weight
pleasure = [g for g in gains.values() if isinstance(g, (int, float)) and g > 0]
ok = ok and len(pleasure) >= 3 and min(pleasure) >= 0.5

verdict = "PASS_TRIPLE_RETURN" if ok else "WEAK_TRIPLE_RETURN"
print(verdict)
for name, d in [("A1", a1), ("B1", b1), ("C1", c1), ("A2", a2)]:
    print(fmt(d, name))
print(f"bridges={a2.get('n_prefill_bridges')} gains={gains}")
PY

{
  echo "# Triple-bridge continuity $STAMP"
  echo
  echo "## Cards"
  cat "$OUT/cards.txt"
  echo
  echo "## Verdict"
  cat "$OUT/return_verdict.txt"
  echo
  echo "## Bridges"
  cat "$OUT/bridges_after.txt"
} >"$OUT/CONTINUITY_CARD.md"

echo
echo "======== SUMMARY ========"
cat "$OUT/cards.txt"
echo
cat "$OUT/return_verdict.txt"
echo "done: $OUT"
