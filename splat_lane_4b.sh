#!/usr/bin/env bash
# =============================================================================
# Splat scaling lane 4B — S1–S4  (force was scaled; splat geometry was not)
# Hierarchy bands now relative to splat_delta_threshold (with_scale_ref).
# =============================================================================
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one paragraph.}"
TOKENS="${TOKENS:-55}"
BIN="$ROOT/target/release/hydrodynamic-swarm"
OUT="logs/splat_lane_4b"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"

[[ -f "$MODEL" ]] || { echo "need $MODEL"; exit 1; }
if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi

echo -e "variant\tearly_d\tlate_d\tearly_Fs\tlate_Fs\tmax_Fs\tFs_ceil%\tmean_Fg\tmean_Fa\tpleasure\tpain\tsnip" > "$SUMMARY"

write_and_run() {
  local name="$1"
  shift
  cat > config.toml <<'BASE'
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = 3.1
splat_sigma = 40.0
splat_alpha = 1.0
min_splat_dist = 30.0
splat_delta_threshold = 70.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.25
splat_force_scale = 0.25
splat_force_max = 28.0
goal_force_scale = 0.125
goal_force_max = 40.0
online_splat_interval = 6
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
max_tokens = 55
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

  python3 - "$@" <<'PY'
import sys, re
overrides = dict(a.split("=", 1) for a in sys.argv[1:])
text = open("config.toml").read()
for k, v in overrides.items():
    if v in ("true", "false") or re.match(r"^-?\d+(\.\d+)?$", v):
        rep = f"{k} = {v}"
    else:
        rep = f'{k} = "{v}"'
    text2, n = re.subn(rf"^{re.escape(k)}\s*=.*$", rep, text, flags=re.M)
    if n == 0:
        text2 = text.replace("[physics]\n", f"[physics]\n{rep}\n")
    text = text2
open("config.toml", "w").write(text)
print(f"  overrides: {overrides}")
PY

  echo ""
  echo "======== VARIANT $name (${TOKENS} tok) ========"
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --tokens "$TOKENS" --clear-memory \
    > "$OUT/${name}.stdout" 2>&1 || true

  LATEST=$(ls -t logs/*.jsonl 2>/dev/null | head -1)
  cp -f "$LATEST" "$OUT/${name}.jsonl" 2>/dev/null || true
  P=$(grep -c 'SPLAT Pleasure' "$OUT/${name}.stdout" 2>/dev/null || true); P=${P:-0}
  A=$(grep -c 'SPLAT Pain' "$OUT/${name}.stdout" 2>/dev/null || true); A=${A:-0}

  python3 - "$name" "$OUT/${name}.jsonl" "$SUMMARY" "$P" "$A" <<'PY'
import json, sys
name, path, summary, pleasure, pain = sys.argv[1:6]
steps = []
try:
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("entry_type") == "step":
                steps.append(o["step"])
except Exception as e:
    open(summary, "a").write(f"{name}\tFAIL\t{e}\n")
    print("FAIL", name, e)
    sys.exit(0)
if not steps:
    open(summary, "a").write(f"{name}\tNA\n")
    print("no steps", name)
    sys.exit(0)

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

n = len(steps)
early = steps[: min(20, n)]
late = steps[max(0, n - 20) :]
def col(ss, k):
    return [float(s.get(k) or 0) for s in ss]
fs = col(steps, "splat_force_mag")
ceil = 100.0 * sum(1 for x in fs if x >= 27.5) / max(1, len(fs))
out = "".join(s.get("token_text", "") for s in steps)[:90].replace("\t", " ").replace("\n", " ")
row = (
    f"{name}\t{mean(col(early,'steering_delta')):.1f}\t{mean(col(late,'steering_delta')):.1f}\t"
    f"{mean(col(early,'splat_force_mag')):.2f}\t{mean(col(late,'splat_force_mag')):.2f}\t"
    f"{max(fs):.1f}\t{ceil:.0f}\t"
    f"{mean(col(steps,'grad_force_mag')):.2f}\t{mean(col(steps,'goal_force_mag')):.2f}\t"
    f"{pleasure}\t{pain}\t{out}"
)
print(row)
open(summary, "a").write(row + "\n")
full = out
try:
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("entry_type") == "summary":
                full = o.get("summary", {}).get("decoded_output", full)
except Exception:
    pass
print(f"  FULL: {full[:200]}")
PY
}

# S1: current B4 force + unscaled 27B-era splat geometry (σ=40, min_dist=30, mass 0.25/28)
write_and_run S1_baseline_splat40

# S2: lower splat mass only (geometry still wide)
write_and_run S2_lower_mass \
  splat_force_scale=0.12 splat_force_max=14.0 online_splat_interval=8

# S3: tighter geometry toward 4B field (~σ7.6) / residual prior ~15
write_and_run S3_tight_sigma \
  splat_sigma=12.0 min_splat_dist=10.0 splat_delta_threshold=95.0

# S4: mass + geometry + slightly pickier deposits (recommended combined)
write_and_run S4_scaled_splats \
  splat_sigma=12.0 min_splat_dist=10.0 splat_delta_threshold=95.0 \
  splat_force_scale=0.12 splat_force_max=14.0 online_splat_interval=8

# Restore default = S4 (if it wins we keep; script always ends on S4 config — research log picks winner)
# Leave S4 knobs in config.toml as the post-lane candidate.
cat > config.toml <<'EOF'
# 4B B4 force + scaled splat geometry (post splat_lane_4b S4 candidate)
# Force: Algo √-law 4B instruct. Splats: width/mass for hidden=2560 / field σ~7.6
# Hierarchy: with_scale_ref vs splat_delta_threshold (not absolute 20/30)
[physics]
dt = 0.035
viscosity_scale = 0.25
force_cap = 3.1
splat_sigma = 12.0
splat_alpha = 1.0
min_splat_dist = 10.0
splat_delta_threshold = 95.0
gradient_topk = 1024
steer_hidden = true
manifold_pullback = 0.25
splat_force_scale = 0.12
splat_force_max = 14.0
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
max_tokens = 90
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

[micro_dream]
entropy_threshold = 3.0
fixed_interval = 25
adaptive_interval = 8
blend_normal = 0.06
blend_high_entropy = 0.10
topocot_threshold = 12.0
EOF

echo ""
echo "======== SPLAT LANE 4B DONE ========"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "Details: $OUT/"
