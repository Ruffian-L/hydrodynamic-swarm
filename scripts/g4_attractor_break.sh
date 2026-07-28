#!/usr/bin/env bash
# G4 — Break the 5-splat consolidation attractor
# Tests: consolidation_threshold=3, dream_replay_bonus=3.0, combined, full-break
# Run from repo root
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/g4_attractor_break_${STAMP}"
mkdir -p "$OUT"

source "$ROOT/scripts/cuda_env.sh"

if [[ ! -x "$BIN" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi
[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

run_one() {
  local id="$1"; shift
  local config="$1"; shift
  echo
  echo "========== RUN $id (config=$config) =========="
  echo "  args: $*"
  set +e
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --config "$config" \
    "$@" 2>&1 | tee "$OUT/${id}.stdout"
  local ec=${PIPESTATUS[0]}
  set -e
  local jl
  jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "$jl" ]]; then
    cp -f "$jl" "$OUT/${id}.jsonl"
  fi
  if [[ -f data/splat_memory.tct.json ]]; then
    cp -f data/splat_memory.tct.json "$OUT/${id}.tct.json" 2>/dev/null || true
  fi
  echo "  exit=$ec → $OUT/${id}.*"
  return 0
}

echo "Output dir: $OUT"
echo "Prompt: $PROMPT ($TOKENS tok)"

# Arm IDs = descriptive slugs (not GA/GB letter codes).
# === Low consolidation threshold (3) ===
echo ""
echo "=== g4_lowconsol (threshold=3) ==="
cat > configs/gates/config.g4_lowconsol.toml <<'EOF'
[model]
path = "data/google/gemma-3-4b-it-Q4_K_M.gguf"
variant = "gemma3"

[physics]
sigma = 7.59
field_wake_scale = 3.0
goal_force_scale = 0.5
splat_force_scale = 1.0
field_logit_bias_alpha = 0.15

[steering]
force_scale = 0.5
attractor_scale = 0.3
force_ramp_tokens = 6

[memory]
splat_memory_path = "data/splat_memory.safetensors"
tct_path = "data/splat_memory.tct"
consolidation_threshold = 3
dream_replay_bonus = 1.25
decay_rate = 0.99

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
log_dir = "logs/g4_attractor_break"
taco_db = "logs/g4_attractor_break/taco_lowconsol.db"
EOF
run_one g4_lowconsol --clear-memory configs/gates/config.g4_lowconsol.toml

# === High dream replay bonus (3.0) ===
echo ""
echo "=== g4_highdream (3.0) ==="
cat > configs/gates/config.g4_highdream.toml <<'EOF'
[model]
path = "data/google/gemma-3-4b-it-Q4_K_M.gguf"
variant = "gemma3"

[physics]
sigma = 7.59
field_wake_scale = 3.0
goal_force_scale = 0.5
splat_force_scale = 1.0
field_logit_bias_alpha = 0.15

[steering]
force_scale = 0.5
attractor_scale = 0.3
force_ramp_tokens = 6

[memory]
splat_memory_path = "data/splat_memory.safetensors"
tct_path = "data/splat_memory.tct"
consolidation_threshold = 5
dream_replay_bonus = 3.0
decay_rate = 0.99

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
log_dir = "logs/g4_attractor_break"
taco_db = "logs/g4_attractor_break/taco_highdream.db"
EOF
run_one g4_highdream --clear-memory configs/gates/config.g4_highdream.toml

# === Combined low threshold + high dream ===
echo ""
echo "=== g4_combined (low threshold + high dream) ==="
cat > configs/gates/config.g4_combined.toml <<'EOF'
[model]
path = "data/google/gemma-3-4b-it-Q4_K_M.gguf"
variant = "gemma3"

[physics]
sigma = 7.59
field_wake_scale = 3.0
goal_force_scale = 0.5
splat_force_scale = 1.0
field_logit_bias_alpha = 0.15

[steering]
force_scale = 0.5
attractor_scale = 0.3
force_ramp_tokens = 6

[memory]
splat_memory_path = "data/splat_memory.safetensors"
tct_path = "data/splat_memory.tct"
consolidation_threshold = 3
dream_replay_bonus = 3.0
decay_rate = 0.99

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
log_dir = "logs/g4_attractor_break"
taco_db = "logs/g4_attractor_break/taco_combined.db"
EOF
run_one g4_combined --clear-memory configs/gates/config.g4_combined.toml

# === Full break (threshold=3 + bonus=3.0 + decay=0.999) ===
echo ""
echo "=== g4_fullbreak (low threshold + high dream + low decay) ==="
cat > configs/gates/config.g4_fullbreak.toml <<'EOF'
[model]
path = "data/google/gemma-3-4b-it-Q4_K_M.gguf"
variant = "gemma3"

[physics]
sigma = 7.59
field_wake_scale = 3.0
goal_force_scale = 0.5
splat_force_scale = 1.0
field_logit_bias_alpha = 0.15

[steering]
force_scale = 0.5
attractor_scale = 0.3
force_ramp_tokens = 6

[memory]
splat_memory_path = "data/splat_memory.safetensors"
tct_path = "data/splat_memory.tct"
consolidation_threshold = 3
dream_replay_bonus = 3.0
decay_rate = 0.999

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
log_dir = "logs/g4_attractor_break"
taco_db = "logs/g4_attractor_break/taco_fullbreak.db"
EOF
run_one g4_fullbreak --clear-memory configs/gates/config.g4_fullbreak.toml

# === Reload phase: test each config's memory persistence ===
# Descriptive slugs only (no GA/GB/… letter codes — avoid misreads like GF → "girlfriend").
echo ""
echo "=== RELOAD PHASE: Test memory persistence ==="

declare -A ARM_CONFIG=(
  [g4_lowconsol]="configs/gates/config.g4_lowconsol.toml"
  [g4_highdream]="configs/gates/config.g4_highdream.toml"
  [g4_combined]="configs/gates/config.g4_combined.toml"
  [g4_fullbreak]="configs/gates/config.g4_fullbreak.toml"
)

for run_id in g4_lowconsol g4_highdream g4_combined g4_fullbreak; do
  echo ""
  echo "=== RELOAD $run_id ==="
  config_file="${ARM_CONFIG[$run_id]:-config.toml}"
  [[ -f "$config_file" ]] || config_file="config.toml"
  run_one "${run_id}_reload" "$config_file"
done

echo ""
echo "=== ALL RUNS COMPLETE ==="
echo "Output dir: $OUT"
ls -la "$OUT/"
