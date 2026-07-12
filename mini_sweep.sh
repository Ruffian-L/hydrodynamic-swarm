#!/usr/bin/env bash
# =============================================================================
# Mini force-balance sweep — 40 tokens each, clear-memory, print force table
# =============================================================================
# Usage: ./mini_sweep.sh
# Results: logs/mini_sweep_summary.tsv + per-run JSONL
# =============================================================================
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="data/google/gemma-3-27b-it-Q8_0.gguf"
TOKENIZER="data/google/tokenizer.json"
PROMPT="Explain the Physics of Friendship in one paragraph."
TOKENS=40
BIN="$ROOT/target/release/hydrodynamic-swarm"
SUMMARY="$ROOT/logs/mini_sweep_summary.tsv"
CONFIG_BAK="$ROOT/config.toml.sweep_bak"

mkdir -p logs
[[ -x "$BIN" ]] || cargo build --release --bin hydrodynamic-swarm

# backup user config
cp -f config.toml "$CONFIG_BAK" 2>/dev/null || true

echo -e "run\tforce_cap\tsplat_scale\tgoal_scale\tmean_Fs\tmean_Fa\tmean_Fg\tmax_Fs\tmean_delta\tuniq\toutput_snip" > "$SUMMARY"

# name | force_cap | splat_force_scale | goal_force_scale | goal_force_max | splat_force_max
RUNS=(
  "A_baseline_old|5.0|0.08|1.0|9999|80"
  "B_goal_damped|4.0|0.08|0.15|60|80"
  "C_balanced|4.0|0.25|0.15|60|60"
  "D_memory_bias|4.0|0.40|0.10|40|80"
  "E_gentle|3.0|0.20|0.12|50|50"
)

run_id=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name fcap sscale gscale gmax smax <<<"$spec"
  run_id=$((run_id + 1))
  echo
  echo "======== SWEEP $run_id/${#RUNS[@]} : $name ========"

  cat > config.toml <<EOF
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = $fcap
splat_sigma = 25.0
splat_alpha = 1.2
min_splat_dist = 20.0
splat_delta_threshold = 70.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.25
splat_force_scale = $sscale
splat_force_max = $smax
goal_force_scale = $gscale
goal_force_max = $gmax
online_splat_interval = 6

[generation]
max_tokens = $TOKENS
temperature = 0.8
rep_penalty = 1.28
min_success_tokens = 10
pleasure_alpha = 1.2
pain_alpha = -0.6
default_prompt = "$PROMPT"

[memory]
max_splats = 80
consolidation_dist = 18.0
decay_rate = 0.96
prune_threshold = 0.02

[micro_dream]
entropy_threshold = 3.0
fixed_interval = 25
adaptive_interval = 8
blend_normal = 0.06
blend_high_entropy = 0.10
topocot_threshold = 12.0
EOF

  # rebuild not needed if only config changes
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --clear-memory \
    > "logs/mini_sweep_${name}.stdout" 2>&1 || true

  # find newest jsonl
  LATEST=$(ls -t logs/*gemma*.jsonl 2>/dev/null | head -1)
  python3 - "$name" "$fcap" "$sscale" "$gscale" "$LATEST" "$SUMMARY" <<'PY'
import json, sys, collections
name, fcap, sscale, gscale, path, summary = sys.argv[1:7]
steps=[]
out=""
with open(path) as f:
    for line in f:
        o=json.loads(line)
        if o.get("entry_type")=="step":
            steps.append(o["step"])
        elif o.get("entry_type")=="summary":
            s=o.get("summary") or o
            out=str(s.get("decoded_output",""))[:80].replace("\t"," ").replace("\n"," ")
if not steps:
    print(f"{name}\tFAIL\tno steps", file=sys.stderr)
    open(summary,"a").write(f"{name}\t{fcap}\t{sscale}\t{gscale}\tNA\tNA\tNA\tNA\tNA\tNA\tFAIL\n")
    sys.exit(0)
def m(k):
    return sum(float(s.get(k) or 0) for s in steps)/len(steps)
uniq=len(set(s.get("token_text") for s in steps))/len(steps)
row=(
  f"{name}\t{fcap}\t{sscale}\t{gscale}\t"
  f"{m('splat_force_mag'):.2f}\t{m('goal_force_mag'):.2f}\t{m('grad_force_mag'):.3f}\t"
  f"{max(s['splat_force_mag'] for s in steps):.2f}\t{m('steering_delta'):.1f}\t{uniq:.2f}\t{out}"
)
print(row)
open(summary,"a").write(row+"\n")
PY
done

# restore config
if [[ -f "$CONFIG_BAK" ]]; then
  mv -f "$CONFIG_BAK" config.toml
fi

echo
echo "======== SWEEP DONE ========"
echo "Table: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
