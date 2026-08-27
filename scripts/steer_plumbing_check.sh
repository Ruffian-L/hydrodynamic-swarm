#!/usr/bin/env bash
# One-token intervention plumbing check on the exact chat path (generate_turn_ex).
# Asserts force_cap=0 vs force_cap>0 moves hidden and/or logits.
# Does not interpret loop wording. Does not run a J-direction sweep.
#
# Usage:
#   ./scripts/steer_plumbing_check.sh
#   HYDRO_MODEL=... HYDRO_TOKENIZER=... ./scripts/steer_plumbing_check.sh
#
# GPU. Do not launch if another hydro/niodoo job owns the card.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
MODEL="${HYDRO_MODEL:-$ROOT/data/google/gemma-4-12b-it-Q4_K_M.gguf}"
TOK="${HYDRO_TOKENIZER:-$ROOT/data/google/gemma4_assets/tokenizer.json}"
CFG_OFF="${CFG_OFF:-$ROOT/configs/experiments/config.plumbing_off.toml}"
CFG_ON="${CFG_ON:-$ROOT/configs/experiments/config.plumbing_on.toml}"
PROMPT="${PROMPT:-Say hi in three words.}"
TOKENS="${TOKENS:-2}"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN — cargo build --release" >&2
  exit 1
fi
if [[ ! -r "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi
if [[ ! -r "$TOK" ]]; then
  echo "missing tokenizer: $TOK" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="$ROOT/logs/steer_plumbing_${STAMP}"
mkdir -p "$OUT"

run_arm() {
  local name="$1" cfg="$2"
  local probe="$OUT/${name}.probe.jsonl"
  local ops="$OUT/${name}.ops.txt"
  local mouth="$OUT/${name}.txt"
  local prompts
  prompts="$(mktemp)"
  printf '%s\nquit\n' "$PROMPT" >"$prompts"
  export COLLAPSE_PROBE="$probe"
  set +e
  "$BIN" \
    --config "$cfg" \
    --model "$MODEL" \
    --tokenizer "$TOK" \
    --tokens "$TOKENS" \
    --chat \
    --clear-memory \
    --no-save-memory \
    --no-hud \
    <"$prompts" >"$mouth" 2>"$ops"
  local rc=$?
  set -e
  rm -f "$prompts"
  echo "arm=$name rc=$rc cfg=$cfg probe=$probe"
  return 0
}

echo "=== steer plumbing $STAMP ==="
echo "model=$MODEL"
echo "prompt=$PROMPT tokens=$TOKENS"
echo "off=$CFG_OFF"
echo "on=$CFG_ON"
echo

run_arm off "$CFG_OFF"
run_arm on "$CFG_ON"

python3 - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])

def first_delta(arm):
    ops = (out / f"{arm}.ops.txt").read_text(errors="replace")
    m = re.search(
        r"\[CHAT DELTA\] turn=\d+ residual_live=(\w+) delta_h_norm=([0-9.]+) hidden_delta=([0-9.]+) logit_delta=([0-9.]+)",
        ops,
    )
    probe = out / f"{arm}.probe.jsonl"
    tok = None
    if probe.exists():
        for line in probe.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "tok":
                tok = rec
                break
    banner = None
    if m:
        banner = {
            "residual_live": m.group(1) == "true",
            "delta_h_norm": float(m.group(2)),
            "hidden_delta": float(m.group(3)),
            "logit_delta": float(m.group(4)),
        }
    return banner, tok

off_b, off_t = first_delta("off")
on_b, on_t = first_delta("on")
print("off banner", off_b)
print("on  banner", on_b)
print("off tok hidden_delta/logit_delta", None if not off_t else (off_t.get("hidden_delta"), off_t.get("logit_delta"), off_t.get("force_on")))
print("on  tok hidden_delta/logit_delta", None if not on_t else (on_t.get("hidden_delta"), on_t.get("logit_delta"), on_t.get("force_on")))

def mag(banner, tok, key):
    if banner and key in banner:
        return banner[key]
    if tok and key in tok:
        return float(tok[key])
    return None

off_h = mag(off_b, off_t, "hidden_delta")
off_z = mag(off_b, off_t, "logit_delta")
on_h = mag(on_b, on_t, "hidden_delta")
on_z = mag(on_b, on_t, "logit_delta")

ok_off = off_h is not None and off_z is not None and off_h < 1e-4 and off_z < 1e-4
ok_on = on_h is not None and on_z is not None and (on_h > 0 or on_z > 0)
print("PASS_OFF" if ok_off else "FAIL_OFF", "hidden", off_h, "logit", off_z)
print("PASS_ON " if ok_on else "FAIL_ON ", "hidden", on_h, "logit", on_z)
if not (ok_off and ok_on):
    sys.exit(2)
print("PLUMBING_OK")
PY
