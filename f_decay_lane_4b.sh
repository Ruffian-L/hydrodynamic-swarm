#!/usr/bin/env bash
# F-decay + B4b retune lane on 4B Q4 (70 tok)
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one paragraph.}"
TOKENS="${TOKENS:-70}"
BIN="$ROOT/target/release/hydrodynamic-swarm"
OUT="logs/f_decay_lane_4b"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
echo -e "variant\tearly_Fs\tlate_Fs\tmax_Fs\tmean_Fa\tmean_Fg\tlate_d\tpleasure\tpain\tsnip" > "$SUMMARY"

[[ -f "$MODEL" ]] || { echo "need $MODEL"; exit 1; }
if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  cargo build --release --bin hydrodynamic-swarm
fi

write_and_run() {
  local name="$1"; shift
  cat > config.toml <<'BASE'
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = 3.1
splat_sigma = 30.0
splat_alpha = 1.0
min_splat_dist = 18.0
splat_delta_threshold = 90.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.25
splat_lambda_default = 0.02
pain_decay_factor = 0.7
splat_force_scale = 0.14
splat_force_max = 16.0
goal_force_scale = 0.125
goal_force_max = 40.0
online_splat_interval = 8
field_wake_mode = "dist_weighted"
field_wake_k = 1
field_wake_scale = 0.187
field_wake_max = 25.0
field_grad_blend = 0.10
field_wake_dist_tau = 80.0
field_logit_alpha = 0.0
force_ramp_tokens = 15
force_ramp_start = 0.15
targeted_splat_only = true
prefill_micro_dream = false
pain_recovery_ocean = false

[generation]
max_tokens = 70
temperature = 0.8
rep_penalty = 1.28
min_success_tokens = 10
pleasure_alpha = 1.2
pain_alpha = -0.6
default_prompt = "Explain the Physics of Friendship in one paragraph."

[memory]
max_splats = 60
consolidation_dist = 18.0
decay_rate = 0.96
prune_threshold = 0.02
online_decay_rate = 1.0

[micro_dream]
entropy_threshold = 3.0
fixed_interval = 25
adaptive_interval = 8
blend_normal = 0.06
blend_high_entropy = 0.10
topocot_threshold = 12.0
BASE
  python3 - "$@" <<'PY'
import sys, re
ov = dict(a.split("=",1) for a in sys.argv[1:])
t = open("config.toml").read()
for k,v in ov.items():
    rep = f"{k} = {v}" if v in ("true","false") or re.match(r'^-?\d+(\.\d+)?$', v) else f'{k} = "{v}"'
    t2,n = re.subn(rf'^{re.escape(k)}\s*=.*$', rep, t, flags=re.M)
    if n==0: t2 = t.replace("[physics]\n", f"[physics]\n{rep}\n")
    t = t2
open("config.toml","w").write(t)
print(" ", ov)
PY
  echo "======== $name ========"
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" --prompt "$PROMPT" --tokens "$TOKENS" --clear-memory \
    > "$OUT/${name}.stdout" 2>&1 || true
  L=$(ls -t logs/*.jsonl | head -1)
  cp -f "$L" "$OUT/${name}.jsonl"
  P=$(grep -c 'SPLAT Pleasure' "$OUT/${name}.stdout" || true); A=$(grep -c 'SPLAT Pain' "$OUT/${name}.stdout" || true)
  python3 - "$name" "$OUT/${name}.jsonl" "$SUMMARY" "${P:-0}" "${A:-0}" <<'PY'
import json,sys
name,path,summary,pleasure,pain=sys.argv[1:6]
steps=[]
for line in open(path):
    o=json.loads(line)
    if o.get("entry_type")=="step": steps.append(o["step"])
if not steps:
    open(summary,"a").write(f"{name}\tNA\n"); print("NA"); raise SystemExit
def m(ss,k): return sum(float(s.get(k) or 0) for s in ss)/len(ss)
early, late = steps[:20], steps[-20:]
snip="".join(s.get("token_text","") for s in steps)[:80].replace("\t"," ").replace("\n"," ")
row=f"{name}\t{m(early,'splat_force_mag'):.2f}\t{m(late,'splat_force_mag'):.2f}\t{max(s['splat_force_mag'] for s in steps):.1f}\t{m(steps,'goal_force_mag'):.1f}\t{m(steps,'grad_force_mag'):.1f}\t{m(late,'steering_delta'):.1f}\t{pleasure}\t{pain}\t{snip}"
print(row); open(summary,"a").write(row+"\n")
full=""
for line in open(path):
    o=json.loads(line)
    if o.get("entry_type")=="summary": full=o.get("summary",{}).get("decoded_output","")
print("  FULL:", full[:180])
PY
}

# B4a: money geometry, no online decay (control)
write_and_run B4a_no_online_decay online_decay_rate=1.0

# B4b: online decay + softer goal (retuned default)
write_and_run B4b_online_decay \
  online_decay_rate=0.975 goal_force_scale=0.10 goal_force_max=32.0 \
  manifold_pullback=0.28 pleasure_alpha=1.0 pain_alpha=-0.5 \
  rep_penalty=1.32 temperature=0.82 max_splats=48 prune_threshold=0.03 \
  splat_lambda_default=0.03

# B4c: stronger online decay
write_and_run B4c_fast_decay \
  online_decay_rate=0.95 goal_force_scale=0.10 goal_force_max=32.0 \
  pleasure_alpha=1.0 pain_alpha=-0.5

# restore B4b as default
cp config.toml /tmp/b4b_cfg.toml 2>/dev/null || true
cat > config.toml <<'EOF'
# B4b default (post f_decay_lane)
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = 3.1
splat_sigma = 30.0
splat_alpha = 1.0
min_splat_dist = 18.0
splat_delta_threshold = 90.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.28
splat_lambda_default = 0.03
pain_decay_factor = 0.7
splat_force_scale = 0.14
splat_force_max = 16.0
goal_force_scale = 0.10
goal_force_max = 32.0
online_splat_interval = 8
field_wake_mode = "dist_weighted"
field_wake_k = 1
field_wake_scale = 0.187
field_wake_max = 25.0
field_grad_blend = 0.10
field_wake_dist_tau = 80.0
field_logit_alpha = 0.0
force_ramp_tokens = 15
force_ramp_start = 0.15
targeted_splat_only = true
prefill_micro_dream = false
pain_recovery_ocean = false

[generation]
max_tokens = 90
temperature = 0.82
rep_penalty = 1.32
min_success_tokens = 10
pleasure_alpha = 1.0
pain_alpha = -0.5
default_prompt = "Explain the Physics of Friendship in one paragraph."

[memory]
max_splats = 48
consolidation_dist = 18.0
decay_rate = 0.96
prune_threshold = 0.03
online_decay_rate = 0.975

[micro_dream]
entropy_threshold = 3.0
fixed_interval = 25
adaptive_interval = 8
blend_normal = 0.06
blend_high_entropy = 0.10
topocot_threshold = 12.0
EOF

echo "======== DONE ========"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
