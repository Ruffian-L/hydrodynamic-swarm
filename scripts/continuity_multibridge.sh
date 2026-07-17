#!/usr/bin/env bash
# Multi-bridge continuity: A → B → A without --clear-memory.
# Expect: A warm, B warm/near, A-return warm (pot/nearest recover).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/continuity_multibridge_${STAMP}"
mkdir -p "$OUT"

source "$ROOT/scripts/cuda_env.sh"
# +fp16 helps some aarch64 hosts; leave unset elsewhere (x86_64, etc.)
if [[ -z "${RUSTFLAGS:-}" ]] && [[ "$(uname -m)" == "aarch64" ]]; then
  export RUSTFLAGS="-C target-feature=+fp16"
fi

[[ -x "$BIN" ]] || { echo "missing $BIN"; exit 1; }
[[ -f data/splat_memory.tct ]] || { echo "missing data/splat_memory.tct"; exit 1; }

A="${PROMPT_A:-Explain the Physics of Friendship in one short paragraph.}"
B="${PROMPT_B:-Write three short tips for debugging a CUDA kernel.}"

echo "out=$OUT"
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
  [[ -n "$jl" ]] && cp -f "$jl" "$OUT/${id}.jsonl"
  cp -f data/splat_memory.tct.json "$OUT/${id}.tct.json" 2>/dev/null || true
  python3 - <<PY | tee "$OUT/${id}.prefill.txt"
import json
from pathlib import Path
p = Path("$OUT/${id}.jsonl")
for line in p.read_text().splitlines():
    o = json.loads(line)

    def find(d, depth=0):
        if not isinstance(d, dict) or depth > 5:
            return None
        if "nearest_scar_dist" in d and "scar_potential_at_prefill" in d:
            return {
                k: d.get(k)
                for k in (
                    "nearest_scar_dist",
                    "nearest_scar_sigma",
                    "scar_potential_at_prefill",
                    "n_prefill_bridges",
                    "scars_at_start",
                    "mean_scar_dist",
                )
            }
        for v in d.values():
            if isinstance(v, dict):
                r = find(v, depth + 1)
                if r:
                    return r
        return None

    r = find(o)
    if r:
        print(r)
        break
PY
  python3 scripts/continuity_card.py "$OUT/${id}.jsonl" \
    --label "$id" \
    --out "$OUT/${id}.kpi.json" | tee -a "$OUT/cards.txt"
  echo "exit=$ec"
}

: >"$OUT/cards.txt"
run_prompt A1 "$A"
run_prompt B1 "$B"
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

a1, b1, a2 = load("A1"), load("B1"), load("A2")

def fmt(d, name):
    g = d.get("gain_max")
    gs = f"{g:.3f}" if isinstance(g, (int, float)) and g == g else "n/a"
    return (
        f"{name}: status={d.get('status')} nearest={d.get('nearest_min')} "
        f"pot={d.get('pot_max')} gain_max={gs}"
    )

ok = a2.get("status") == "WARM"
if a2.get("nearest_min") is not None:
    ok = ok and a2["nearest_min"] < 200.0
if a1.get("pot_max") and a1["pot_max"] > 0.2 and a2.get("pot_max") is not None:
    ok = ok and (
        a2["pot_max"] >= 0.5 * a1["pot_max"] or (a2.get("nearest_min") or 999) < 80
    )
# weight floor: strongest bridge should not vanish after multi-topic
g2 = a2.get("gain_max")
if isinstance(g2, (int, float)) and g2 == g2:
    ok = ok and g2 >= 0.2

verdict = "PASS_RETURN" if ok else "WEAK_RETURN"
print(verdict)
print(fmt(a1, "A1"))
print(fmt(b1, "B1"))
print(fmt(a2, "A2"))
print(f"bridges={a2.get('n_prefill_bridges')} fps={a2.get('bridge_prompt_fps')}")
if a2.get("bridge_gains"):
    print(f"bridge_gains={a2.get('bridge_gains')}")
PY

{
  echo "# Multi-bridge continuity $STAMP"
  echo
  echo "## Cards"
  cat "$OUT/cards.txt"
  echo
  echo "## Return verdict"
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
echo "card: $OUT/CONTINUITY_CARD.md"
