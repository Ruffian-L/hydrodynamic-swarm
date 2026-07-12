#!/usr/bin/env bash
# =============================================================================
# Learning Lane 4B — B4 / D4 / Ctrl4  (Jason + Grok 2026-07-11)
# Scaled √-law base (Algo_WIPjuly → 4B instruct) · 90 tokens · early vs late
# =============================================================================
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one paragraph.}"
TOKENS="${TOKENS:-90}"
BIN="$ROOT/target/release/hydrodynamic-swarm"
OUT="logs/learning_lane_4b"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

[[ -f "$MODEL" ]] || { echo "need $MODEL"; exit 1; }
if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi

echo -e "variant\tearly_d\tlate_d\tall_d\tearly_Fs\tlate_Fs\tmax_Fs\tmean_Fg\tmean_Fa\tuniq\tpleasure\tpain\tsnip" > "$SUMMARY"

write_and_run() {
  local name="$1"
  shift
  cat > config.toml <<'BASE'
# Auto-written by learning_lane_4b.sh — 4B scaled base
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
max_tokens = 90
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

  python3 - "$name" "$@" <<'PY'
import sys, re
name = sys.argv[1]
overrides = dict(a.split("=", 1) for a in sys.argv[2:])
path = "config.toml"
text = open(path).read()
for k, v in overrides.items():
    if v in ("true", "false") or re.match(r"^-?\d+(\.\d+)?$", v):
        rep = f"{k} = {v}"
    else:
        rep = f'{k} = "{v}"'
    text2, n = re.subn(rf"^{re.escape(k)}\s*=.*$", rep, text, flags=re.M)
    if n == 0:
        text2 = text.replace("[physics]\n", f"[physics]\n{rep}\n")
    text = text2
open(path, "w").write(text)
print(f"  config overrides: {overrides}")
PY

  echo ""
  echo "======== VARIANT $name (${TOKENS} tok) ========"
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" --tokens "$TOKENS" --clear-memory \
    > "$OUT/${name}.stdout" 2>&1 || true

  # Prefer newest jsonl that mentions 4b or gemma3
  LATEST=$(ls -t logs/*.jsonl 2>/dev/null | head -1)
  cp -f "$LATEST" "$OUT/${name}.jsonl" 2>/dev/null || true
  cp -f "$OUT/${name}.stdout" "$OUT/${name}.${STAMP}.stdout" 2>/dev/null || true

  P=$(grep -c 'SPLAT Pleasure' "$OUT/${name}.stdout" 2>/dev/null || true)
  A=$(grep -c 'SPLAT Pain' "$OUT/${name}.stdout" 2>/dev/null || true)
  P=${P:-0}; A=${A:-0}

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
    open(summary, "a").write(f"{name}\tFAIL\t\t\t\t\t\t\t\t\t\t\t{e}\n")
    print("FAIL", name, e)
    sys.exit(0)
if not steps:
    open(summary, "a").write(f"{name}\tNA\n")
    print("no steps", name)
    sys.exit(0)

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

n = len(steps)
# early = first 30; late = last third or steps >= 60
early = steps[: min(30, n)]
late = [s for s in steps if s.get("step", 0) >= 60] or steps[max(0, n * 2 // 3) :]

def col(ss, k):
    return [float(s.get(k) or 0) for s in ss]

out = "".join(s.get("token_text", "") for s in steps)[:100].replace("\t", " ").replace("\n", " ")
uniq = len(set(s.get("token_text") for s in steps)) / n
row = (
    f"{name}\t{mean(col(early,'steering_delta')):.1f}\t{mean(col(late,'steering_delta')):.1f}\t"
    f"{mean(col(steps,'steering_delta')):.1f}\t"
    f"{mean(col(early,'splat_force_mag')):.2f}\t{mean(col(late,'splat_force_mag')):.2f}\t"
    f"{max(col(steps,'splat_force_mag')):.1f}\t"
    f"{mean(col(steps,'grad_force_mag')):.2f}\t{mean(col(steps,'goal_force_mag')):.2f}\t"
    f"{uniq:.2f}\t{pleasure}\t{pain}\t{out}"
)
print(row)
open(summary, "a").write(row + "\n")

# full decoded from summary entry if present
full = out
try:
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("entry_type") == "summary":
                full = o.get("summary", {}).get("decoded_output", full)
except Exception:
    pass
print(f"  FULL[{name}]: {full[:220]}")
PY
}

# B4: ramp + targeted (scaled 4B — validated prime path)
write_and_run B4_ramp_targeted \
  force_ramp_tokens=15 force_ramp_start=0.15 targeted_splat_only=true \
  prefill_micro_dream=false goal_force_scale=0.125 field_logit_alpha=0.0

# D4: J-space respect — longer/weaker ramp + prefill micro-dream + softer goal
write_and_run D4_jspace \
  force_ramp_tokens=18 force_ramp_start=0.10 targeted_splat_only=true \
  prefill_micro_dream=true goal_force_scale=0.08 field_logit_alpha=0.0

# Ctrl4: same force mass, NO ramp / NO targeted (isolates formula vs process)
write_and_run Ctrl4_scaled_only \
  force_ramp_tokens=0 force_ramp_start=0.15 targeted_splat_only=false \
  prefill_micro_dream=false goal_force_scale=0.125 field_logit_alpha=0.0

# Restore working default = B4 (best prior short-smoke default)
cat > config.toml <<'EOF'
# =============================================================================
# Learning-lane defaults for Gemma-3-4B-it (√-law scaled) — B4 after ablation
# docs/MODEL_SIZE_PHYSICS_SCALING.md
# =============================================================================

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
max_tokens = 90
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
EOF

echo ""
echo "======== LEARNING LANE 4B DONE ========"
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
