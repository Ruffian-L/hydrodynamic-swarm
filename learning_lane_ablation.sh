#!/usr/bin/env bash
# =============================================================================
# Niodoo Learning Lane — targeted ablations (Jason directive 2026-07-11)
# Small params, short runs, original ramp + selective splats.
# Model: data/google/gemma-3-27b-it-Q4_K_M.gguf
# =============================================================================
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="data/google/gemma-3-27b-it-Q4_K_M.gguf"
TOKENIZER="data/google/tokenizer.json"
PROMPT="Explain the Physics of Friendship in one paragraph."
TOKENS=50
BIN="$ROOT/target/release/hydrodynamic-swarm"
OUT="logs/learning_lane"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"

[[ -f "$MODEL" ]] || { echo "need $MODEL"; exit 1; }
[[ -x "$BIN" ]] || cargo build --release --bin hydrodynamic-swarm

echo -e "variant\tmean_d\tmean_Fg\tmean_Fs\tmean_Fa\tmax_Fs\tuniq\tpleasure\tpain\toutput_snip" > "$SUMMARY"

write_and_run() {
  local name="$1"
  shift
  # remaining args are KEY=VALUE overrides applied on top of BASE
  cat > config.toml <<'BASE'
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = 3.0
splat_sigma = 40.0
splat_alpha = 1.0
min_splat_dist = 30.0
splat_delta_threshold = 70.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.25
splat_force_scale = 0.25
splat_force_max = 28.0
goal_force_scale = 0.12
goal_force_max = 40.0
online_splat_interval = 6
field_wake_mode = "dist_weighted"
field_wake_k = 1
field_wake_scale = 0.18
field_wake_max = 25.0
field_grad_blend = 0.10
field_wake_dist_tau = 80.0
field_logit_alpha = 0.0
force_ramp_tokens = 0
force_ramp_start = 0.20
targeted_splat_only = false
prefill_micro_dream = false
pain_recovery_ocean = false

[generation]
max_tokens = 50
temperature = 0.8
rep_penalty = 1.28
min_success_tokens = 10
pleasure_alpha = 1.2
pain_alpha = -0.6
default_prompt = "Explain the Physics of Friendship in one paragraph."

[memory]
max_splats = 60
consolidation_dist = 25.0
decay_rate = 0.96
prune_threshold = 0.02

[micro_dream]
entropy_threshold = 3.0
fixed_interval = 25
adaptive_interval = 8
blend_normal = 0.06
blend_high_entropy = 0.10
topocot_threshold = 12.0
BASE
  # apply overrides via python for reliability
  python3 - "$name" "$@" <<'PY'
import sys, re
name = sys.argv[1]
overrides = dict(a.split("=",1) for a in sys.argv[2:])
path = "config.toml"
text = open(path).read()
for k,v in overrides.items():
    # bool/number as-is
    if v in ("true","false") or re.match(r'^-?\d+(\.\d+)?$', v):
        rep = f"{k} = {v}"
    else:
        rep = f'{k} = "{v}"'
    text2, n = re.subn(rf'^{re.escape(k)}\s*=.*$', rep, text, flags=re.M)
    if n == 0:
        # insert under [physics]
        text2 = text.replace("[physics]\n", f"[physics]\n{rep}\n")
    text = text2
open(path,"w").write(text)
print(f"  config overrides: {overrides}")
PY

  echo ""
  echo "======== VARIANT $name ========"
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --tokens "$TOKENS" --clear-memory \
    > "$OUT/${name}.stdout" 2>&1 || true

  LATEST=$(ls -t logs/*gemma*.jsonl 2>/dev/null | head -1)
  cp -f "$LATEST" "$OUT/${name}.jsonl" 2>/dev/null || true
  # count splat tags from stdout
  P=$(grep -c 'SPLAT Pleasure' "$OUT/${name}.stdout" 2>/dev/null || echo 0)
  A=$(grep -c 'SPLAT Pain' "$OUT/${name}.stdout" 2>/dev/null || echo 0)
  python3 - "$name" "$OUT/${name}.jsonl" "$SUMMARY" "$P" "$A" <<'PY'
import json,sys
name, path, summary, pleasure, pain = sys.argv[1:6]
steps=[]
try:
    with open(path) as f:
        for line in f:
            o=json.loads(line)
            if o.get("entry_type")=="step":
                steps.append(o["step"])
except Exception as e:
    open(summary,"a").write(f"{name}\tFAIL\t\t\t\t\t\t\t\t{e}\n")
    print("FAIL", name, e); sys.exit(0)
if not steps:
    open(summary,"a").write(f"{name}\tNA\n"); print("no steps", name); sys.exit(0)
def m(k):
    return sum(float(s.get(k) or 0) for s in steps)/len(steps)
out="".join(s.get("token_text","") for s in steps)[:90].replace("\t"," ").replace("\n"," ")
uniq=len(set(s.get("token_text") for s in steps))/len(steps)
row=(f"{name}\t{m('steering_delta'):.1f}\t{m('grad_force_mag'):.2f}\t{m('splat_force_mag'):.2f}\t"
     f"{m('goal_force_mag'):.2f}\t{max(s['splat_force_mag'] for s in steps):.1f}\t{uniq:.2f}\t"
     f"{pleasure}\t{pain}\t{out}")
print(row)
open(summary,"a").write(row+"\n")
PY
}

# A: baseline-ish (no ramp, not targeted-only, current caps, no logit)
write_and_run A_baseline \
  force_ramp_tokens=0 targeted_splat_only=false field_logit_alpha=0.0 \
  splat_force_max=28.0 force_cap=3.0

# B: ramp + targeted splats (original Niodoo spirit)
write_and_run B_ramp_targeted \
  force_ramp_tokens=12 force_ramp_start=0.20 targeted_splat_only=true \
  field_logit_alpha=0.0 splat_force_max=28.0

# C: lower overall governance
write_and_run C_low_gov \
  force_ramp_tokens=12 force_ramp_start=0.15 targeted_splat_only=true \
  splat_force_max=18.0 field_wake_max=15.0 goal_force_max=30.0 force_cap=2.5 \
  field_logit_alpha=0.0

# D: respect J-space (weaker early + prefill micro-dream)
write_and_run D_jspace \
  force_ramp_tokens=15 force_ramp_start=0.10 targeted_splat_only=true \
  prefill_micro_dream=true field_logit_alpha=0.0 goal_force_scale=0.08

# E: pain recovery ocean packets
write_and_run E_recovery \
  force_ramp_tokens=12 force_ramp_start=0.20 targeted_splat_only=true \
  pain_recovery_ocean=true field_logit_alpha=0.0

echo ""
echo "======== LEARNING LANE DONE ========"
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
