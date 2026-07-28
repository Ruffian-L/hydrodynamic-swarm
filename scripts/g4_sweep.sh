#!/bin/bash
# G4 Parameter Sweep: decay × splat_force × consolidation
# Each variant runs mint → save → reload → compare
# Usage: ./g4_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BIN="$PROJECT_ROOT/target/release/hydrodynamic-swarm"
MODEL="data/google/gemma-3-4b-it-Q4_K_M.gguf"
PROMPT="Explain the Physics of Friendship in one paragraph."
LOGDIR="$PROJECT_ROOT/logs/g4_sweep_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Variants: name | decay | splat_scale | consol_thresh | dream_bonus
VARIANTS=(
  "V1_lowdecay|0.95|1.0|5|1.25"
  "V2_highsplat|0.99|2.0|5|1.25"
  "V3_lowconsol|0.99|1.0|2|1.25"
  "V4_highdream|0.99|1.0|5|3.0"
  "V5_combo|0.95|2.0|2|3.0"
)

echo "=== G4 PARAMETER SWEEP ==="
echo "Log dir: $LOGDIR"
echo "Variants: ${#VARIANTS[@]}"
echo ""

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r name decay splat_scale consol_thresh dream_bonus <<<"$spec"
  echo "======== $name: decay=$decay splat=$splat_scale consol=$consol_thresh dream=$dream_bonus ========"
  
  # Write variant config
  cat > "$LOGDIR/${name}_config.toml" <<EOF
[model]
path = "$MODEL"
variant = "gemma3"

[physics]
sigma = 7.59
field_wake_scale = 3.0
goal_force_scale = 0.5
splat_force_scale = $splat_scale
field_logit_bias_alpha = 0.15

[steering]
force_scale = 0.5
attractor_scale = 0.3
force_ramp_tokens = 6

[memory]
splat_memory_path = "$LOGDIR/${name}_memory.safetensors"
tct_path = "$LOGDIR/${name}_memory.tct"
consolidation_threshold = $consol_thresh
dream_replay_bonus = $dream_bonus
decay_rate = $decay

[ocean]
enabled = true
dim = 2560
deposit_every = 4
force_scale = 0.12

[generation]
max_tokens = 150
temperature = 0.9
top_k = 1024

[logging]
log_dir = "$LOGDIR"
taco_db = "$LOGDIR/taco_${name}.db"
EOF

  # Run A: mint (clear memory)
  echo "  [A] Mint run..."
  "$BIN" \
    --config "$LOGDIR/${name}_config.toml" \
    --clear-memory \
    --prompt "$PROMPT" \
    > "$LOGDIR/${name}_A_mint.stdout" 2>&1 || true
  
  # Run B: reload (no clear, loads memory from A)
  echo "  [B] Reload run..."
  "$BIN" \
    --config "$LOGDIR/${name}_config.toml" \
    --prompt "$PROMPT" \
    > "$LOGDIR/${name}_B_reload.stdout" 2>&1 || true
  
  echo "  DONE $name"
  echo ""
done

echo "=== SWEEP COMPLETE ==="
echo "Results in: $LOGDIR"
echo ""

# Quick summary: extract key metrics from each run
echo "=== SUMMARY ==="
echo "run\tvariant\tscars_mint\tscars_reload\tpain_mint\tpain_reload\tdelta_mint\tdelta_reload"

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r name decay splat_scale consol_thresh dream_bonus <<<"$spec"
  
  # Extract from stdout
  mint_scars=$(grep -oP 'n_scars=\K\d+' "$LOGDIR/${name}_A_mint.stdout" 2>/dev/null || echo "0")
  reload_scars=$(grep -oP 'n_scars=\K\d+' "$LOGDIR/${name}_B_reload.stdout" 2>/dev/null || echo "0")
  mint_pain=$(grep -oP 'pain_tokens=\K\d+' "$LOGDIR/${name}_A_mint.stdout" 2>/dev/null || echo "0")
  reload_pain=$(grep -oP 'pain_tokens=\K\d+' "$LOGDIR/${name}_B_reload.stdout" 2>/dev/null || echo "0")
  mint_delta=$(grep -oP 'avg_delta=\K[\d.]+' "$LOGDIR/${name}_A_mint.stdout" 2>/dev/null || echo "0")
  reload_delta=$(grep -oP 'avg_delta=\K[\d.]+' "$LOGDIR/${name}_B_reload.stdout" 2>/dev/null || echo "0")
  
  echo "${name}_mint\t${name}\t${mint_scars}\t${reload_scars}\t${mint_pain}\t${reload_pain}\t${mint_delta}\t${reload_delta}"
  echo "${name}_reload\t${name}\t${mint_scars}\t${reload_scars}\t${mint_pain}\t${reload_pain}\t${mint_delta}\t${reload_delta}"
done
