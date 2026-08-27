#!/usr/bin/env bash
# =============================================================================
# AI / team multi-turn SMOKE — same settings as talk.sh (see convo_defaults.sh)
# =============================================================================
# Multi-turn ONLY (stdin → --chat). Not a one-shot. See:
#   research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md
#
#   ./scripts/smoke_convo.sh
#   HYDRO_TOKENS=80 ./scripts/smoke_convo.sh
#
# Both look:  logs/smoke_convo_latest.txt
#             grep 'gemma4>' logs/smoke_convo_latest.txt
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
# shellcheck source=convo_defaults.sh
source "$ROOT/scripts/convo_defaults.sh"

BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
if [[ ! -x "$BIN" ]]; then
  echo "Missing binary: $BIN — run: cargo build --release" >&2
  exit 1
fi
if [[ ! -r "$HYDRO_MODEL" ]]; then
  echo "Model not readable: $HYDRO_MODEL" >&2
  exit 1
fi
if [[ ! -r "$HYDRO_TOKENIZER" ]]; then
  echo "Tokenizer not readable: $HYDRO_TOKENIZER" >&2
  exit 1
fi
if [[ ! -r "$HYDRO_CONFIG" ]]; then
  echo "Config not readable: $HYDRO_CONFIG" >&2
  exit 1
fi

mkdir -p logs
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/smoke_convo_${STAMP}.txt"
OPS="logs/smoke_convo_${STAMP}.ops.txt"
PROBE="logs/smoke_convo_${STAMP}.probe.jsonl"
SCALER_RECEIPT="logs/smoke_convo_${STAMP}.scaler.json"
LATEST="logs/smoke_convo_latest.txt"
LATEST_OPS="logs/smoke_convo_latest.ops.txt"
LATEST_PROBE="logs/smoke_convo_latest.probe.jsonl"
LATEST_SCALER="logs/smoke_convo_latest.scaler.json"

PROMPTS="$(mktemp)"
trap 'rm -f "$PROMPTS"' EXIT
PROMPTS_SRC="DEFAULT-9TURN"
# -f rejects process substitution (/dev/fd/N). Named evals use a real file.
if [[ -n "${PROMPTS_FILE:-}" && -r "$PROMPTS_FILE" && -s "$PROMPTS_FILE" ]]; then
  cat "$PROMPTS_FILE" >"$PROMPTS"
  PROMPTS_SRC="$PROMPTS_FILE"
else
  if [[ -n "${PROMPTS_FILE:-}" ]]; then
    echo "WARN: PROMPTS_FILE=$PROMPTS_FILE is not a readable non-empty file; using default 9-turn." >&2
  fi
  # Multi-turn script (history accumulates). Not one-shot.
  cat >"$PROMPTS" <<'EOF'
Say hi in three words.
What is 2+2?
Name one color.
Spell cat.
Count to three.
Write two short sentences about residual streams.
Reply with one word: ready
What is 17 times 23? Show arithmetic.
Repeat exactly: the quick brown fox jumps over the lazy dog
quit
EOF
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export COLLAPSE_PROBE="$PROBE"
export HYDRO_SCALER_RECEIPT="$SCALER_RECEIPT"

{
  echo "=== smoke_convo (MULTI-TURN) $STAMP ==="
  echo "parity: same as talk.sh via scripts/convo_defaults.sh"
  echo "model=$HYDRO_MODEL"
  echo "config=$HYDRO_CONFIG"
  echo "tokens=$HYDRO_TOKENS"
  echo "flags=${HYDRO_CHAT_FLAGS[*]}"
  echo "inject=${HYDRO_INJECT_TAG:-}"
  echo "keep_memory=${HYDRO_KEEP_MEMORY:-}"
  echo "prompts=$PROMPTS_SRC"
  echo "out=$OUT"
  echo "ops=$OPS"
  echo "probe=$PROBE"
  echo "scaler_receipt=$SCALER_RECEIPT"
  echo "size_rule=${HYDRO_SIZE_RULE:-config}"
  echo "scaler_gain=${HYDRO_SCALER_GAIN:-config}"
  echo "scaler_apply=${HYDRO_SCALER_APPLY:-config}"
  echo "sample_seed=${HYDRO_SAMPLE_SEED:-unset}"
  echo "tda_monitor=${HYDRO_TDA_MONITOR:-default-on}"
  echo "NOTE: one-shot smokes are not valid multi-turn evidence"
  echo "NOTE: mouth is $OUT (tags / Internal monitor / memory inject). Ops in $OPS."
  echo ""
} | tee "$OPS" >&2

: >"$OUT"
ln -sfn "$(basename "$OUT")" "$LATEST"
ln -sfn "$(basename "$OPS")" "$LATEST_OPS"
ln -sfn "$(basename "$PROBE")" "$LATEST_PROBE" 2>/dev/null || true
ln -sfn "$(basename "$SCALER_RECEIPT")" "$LATEST_SCALER" 2>/dev/null || true

"$BIN" \
  --config "$HYDRO_CONFIG" \
  --model "$HYDRO_MODEL" \
  --tokenizer "$HYDRO_TOKENIZER" \
  --tokens "$HYDRO_TOKENS" \
  "${HYDRO_CHAT_FLAGS[@]}" \
  <"$PROMPTS" 2>>"$OPS" | tee -a "$OUT"

ln -sfn "$(basename "$OUT")" "$LATEST"
ln -sfn "$(basename "$OPS")" "$LATEST_OPS"
ln -sfn "$(basename "$PROBE")" "$LATEST_PROBE" 2>/dev/null || true
ln -sfn "$(basename "$SCALER_RECEIPT")" "$LATEST_SCALER" 2>/dev/null || true

if [[ ! -s "$SCALER_RECEIPT" ]]; then
  echo "Missing immutable scaler receipt: $SCALER_RECEIPT" >&2
  exit 1
fi

echo ""
echo "DONE — mouth (follow this): $OUT"
echo "  $LATEST"
echo "ops (TUI-class telemetry, model does not see): $OPS"
echo "  grep 'gemma4>' $LATEST"
