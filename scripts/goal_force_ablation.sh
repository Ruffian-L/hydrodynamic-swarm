#!/bin/bash
# Goal Force Ablation: ON vs OFF
# Tests whether the attractor goal_force is driving the oscillation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BIN="$PROJECT_ROOT/target/release/hydrodynamic-swarm"
MODEL="data/google/gemma-3-4b-it-Q4_K_M.gguf"
PROMPT="Explain the Physics of Friendship in one short paragraph."
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/goal_force_ablation_${STAMP}"
mkdir -p "$OUT"

source "$SCRIPT_DIR/cuda_env.sh"

if [[ ! -x "$BIN" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi
[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

echo "=== GOAL FORCE ABLATION ==="
echo "Output dir: $OUT"
echo "Acceptance: residual < 5000 for 50+ steps, no growing oscillation"
echo ""

# Variant A: goal_force ON (default: scale=0.15, max=60)
echo "======== A_goal_ON: scale=0.15 max=60 ========"
"$BIN" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --tokens 65 \
  --force_cap 3.1 \
  --temperature 0.88 \
  --clear-memory \
  > "$OUT/A_goal_ON.stdout" 2>&1 || true

jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
if [[ -n "$jl" ]]; then
  cp -f "$jl" "$OUT/A_goal_ON.jsonl"
  steps=$(python3 -c "
import json
steps = []
try:
    with open('$jl') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('entry_type') == 'step':
                steps.append(obj['step'].get('residual_norm', 0))
except:
    pass
if not steps:
    print('0|0|0|0|FAIL')
else:
    total = len(steps)
    peak = max(steps)
    last = steps[-1]
    max_consec = 0
    cur_consec = 0
    for i in range(1, len(steps)):
        if steps[i] > steps[i-1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    below_5k = sum(1 for s in steps if s < 5000)
    print(f'{total}|{peak:.0f}|{last:.0f}|{max_consec}|{\"PASS\" if below_5k >= 50 and max_consec < 3 else \"FAIL\"}')
" 2>&1)
  IFS='|' read -r total peak last consec status <<<"$steps"
  echo "  steps=$total peak=$peak last=$last consec_increases=$consec → $status"
fi

# Variant B: goal_force OFF (scale=0)
echo ""
echo "======== B_goal_OFF: scale=0.0 max=60 ========"
"$BIN" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --tokens 65 \
  --force_cap 3.1 \
  --temperature 0.88 \
  --goal_force_scale 0.0 \
  --clear-memory \
  > "$OUT/B_goal_OFF.stdout" 2>&1 || true

jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
if [[ -n "$jl" ]]; then
  cp -f "$jl" "$OUT/B_goal_OFF.jsonl"
  steps=$(python3 -c "
import json
steps = []
try:
    with open('$jl') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('entry_type') == 'step':
                steps.append(obj['step'].get('residual_norm', 0))
except:
    pass
if not steps:
    print('0|0|0|0|FAIL')
else:
    total = len(steps)
    peak = max(steps)
    last = steps[-1]
    max_consec = 0
    cur_consec = 0
    for i in range(1, len(steps)):
        if steps[i] > steps[i-1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    below_5k = sum(1 for s in steps if s < 5000)
    print(f'{total}|{peak:.0f}|{last:.0f}|{max_consec}|{\"PASS\" if below_5k >= 50 and max_consec < 3 else \"FAIL\"}')
" 2>&1)
  IFS='|' read -r total peak last consec status <<<"$steps"
  echo "  steps=$total peak=$peak last=$last consec_increases=$consec → $status"
fi

echo ""
echo "=== ABLATION COMPLETE ==="
echo "Results in: $OUT"
ls -la "$OUT/"
