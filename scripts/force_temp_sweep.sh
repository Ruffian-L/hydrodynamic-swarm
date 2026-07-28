#!/bin/bash
# Force Cap × Temperature Factorial Sweep
# Tests whether oscillation is driven by force_cap, temperature, or both
# 
# Acceptance criterion: residual < 5000 for 50+ steps, no growing oscillation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BIN="$PROJECT_ROOT/target/release/hydrodynamic-swarm"
MODEL="data/google/gemma-3-4b-it-Q4_K_M.gguf"
PROMPT="Explain the Physics of Friendship in one short paragraph."
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/force_temp_sweep_${STAMP}"
mkdir -p "$OUT"

source "$SCRIPT_DIR/cuda_env.sh"

if [[ ! -x "$BIN" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi
[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

# Variants: name | force_cap | temperature
VARIANTS=(
  "A_baseline|3.1|0.88"
  "B_lowtemp|3.1|0.5"
  "C_current|5.0|0.5"
  "D_lowforce|2.0|0.5"
  "E_verylowtemp|3.1|0.3"
  "F_gentle|2.0|0.3"
)

echo "=== FORCE CAP × TEMPERATURE SWEEP ==="
echo "Output dir: $OUT"
echo "Variants: ${#VARIANTS[@]}"
echo "Acceptance: residual < 5000 for 50+ steps, no growing oscillation"
echo ""

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r name force_cap temperature <<<"$spec"
  echo "======== $name: force_cap=$force_cap T=$temperature ========"
  
  # Run with explicit overrides
  "$BIN" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --tokens 65 \
    --force_cap "$force_cap" \
    --temperature "$temperature" \
    --clear-memory \
    > "$OUT/${name}.stdout" 2>&1 || true
  
  # Extract metrics
  jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "$jl" ]]; then
    cp -f "$jl" "$OUT/${name}.jsonl"
  fi
  
  # Quick analysis
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
    # Count consecutive increases (growing oscillation)
    max_consec = 0
    cur_consec = 0
    for i in range(1, len(steps)):
        if steps[i] > steps[i-1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0
    # Check if residual stays below 5000 for 50+ steps
    below_5k = sum(1 for s in steps if s < 5000)
    print(f'{total}|{peak:.0f}|{last:.0f}|{max_consec}|{\"PASS\" if below_5k >= 50 and max_consec < 3 else \"FAIL\"}')
" 2>&1)
  
  IFS='|' read -r total peak last consec status <<<"$steps"
  echo "  steps=$total peak=$peak last=$last consec_increases=$consec → $status"
  echo ""
done

echo "=== SWEEP COMPLETE ==="
echo "Results in: $OUT"
ls -la "$OUT/"
