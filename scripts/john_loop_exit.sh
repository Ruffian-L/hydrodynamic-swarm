#!/usr/bin/env bash
# John's minimum strong experiment on the live chat path (generate_turn_ex).
# A plumbing, C history controls, E c=0/target/reverse/random, F durable-exit score.
# D ordinary-logit ranks are on event=tok.diag; J-lens apply is not this binary.
#
# GPU. Do not launch if a training job owns the card unless MODEL is small enough.
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
A_TOKENS="${A_TOKENS:-2}"
C_TOKENS="${C_TOKENS:-48}"
E_TOKENS="${E_TOKENS:-64}"
DIR_C="${HYDRO_DIR_C:-0.35}"
DIR_TOKEN="${HYDRO_DIR_TOKEN:-repetitive}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="${OUT:-$ROOT/logs/john_loop_exit_${STAMP}}"
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — cargo build --release --bin hydrodynamic-swarm" >&2
  exit 1
fi
if [[ ! -r "$MODEL" || ! -r "$TOK" ]]; then
  echo "missing model/tokenizer: $MODEL $TOK" >&2
  exit 1
fi

COLLAPSE='ParisofparisofparisOf Paris Of Paris Of Paris Of Paris Of Paris Of Paris Of Paris Of Paris'

chat_run() {
  local name="$1" cfg="$2" tokens="$3" prompt="$4"
  shift 4 || true
  local probe="$OUT/${name}.probe.jsonl"
  local ops="$OUT/${name}.ops.txt"
  local mouth="$OUT/${name}.txt"
  local prompts
  prompts="$(mktemp)"
  printf '%s\nquit\n' "$prompt" >"$prompts"
  echo "arm=$name tokens=$tokens cfg=$(basename "$cfg")"
  set +e
  env COLLAPSE_PROBE="$probe" HYDRO_TAGS_ON=0 "$@" \
    "$BIN" \
      --config "$cfg" \
      --model "$MODEL" \
      --tokenizer "$TOK" \
      --tokens "$tokens" \
      --chat \
      --clear-memory \
      --no-save-memory \
      --no-hud \
      --no-endocrine \
      <"$prompts" >"$mouth" 2>"$ops"
  local rc=$?
  set -e
  rm -f "$prompts"
  echo "  rc=$rc mouth=$mouth"
  rg -n '\[CHAT DELTA\]|\[DIR STEER\]|\[SEED ASSISTANT\]' "$ops" | head -5 || true
}

echo "=== john loop-exit $STAMP ==="
echo "out=$OUT"
echo "model=$MODEL"
echo "bin=$BIN"
echo

echo "--- A plumbing ---"
chat_run A_off "$CFG_OFF" "$A_TOKENS" "Say hi in three words."
chat_run A_on  "$CFG_ON"  "$A_TOKENS" "Say hi in three words."

echo "--- C history controls (isolation, no dir) ---"
chat_run C_visible_plain "$CFG_OFF" "$C_TOKENS" \
  "Describe what happened in the previous assistant output in one sentence." \
  HYDRO_SEED_ASSISTANT="$COLLAPSE"
chat_run C_unprimed "$CFG_OFF" "$C_TOKENS" \
  "Describe what happened in the previous assistant output in one sentence."
chat_run C_visible_nodesc "$CFG_OFF" "$C_TOKENS" \
  "What did the previous assistant just produce? One sentence." \
  HYDRO_SEED_ASSISTANT="$COLLAPSE"
chat_run C_explicit "$CFG_OFF" "$C_TOKENS" \
  "Was that a hallucination loop or a failure? Diagnose it in one sentence." \
  HYDRO_SEED_ASSISTANT="$COLLAPSE"

echo "--- E dir-steer (isolation residual, direct unembed of $DIR_TOKEN) ---"
E_PROMPT="Write a poem where every word starts with P. Continue until you stop."
chat_run E_c0 "$CFG_OFF" "$E_TOKENS" "$E_PROMPT" \
  HYDRO_DIR_MODE=none HYDRO_DIR_C=0
chat_run E_target "$CFG_OFF" "$E_TOKENS" "$E_PROMPT" \
  HYDRO_DIR_MODE=target HYDRO_DIR_C="$DIR_C" HYDRO_DIR_TOKEN="$DIR_TOKEN"
chat_run E_reverse "$CFG_OFF" "$E_TOKENS" "$E_PROMPT" \
  HYDRO_DIR_MODE=reverse HYDRO_DIR_C="$DIR_C" HYDRO_DIR_TOKEN="$DIR_TOKEN"
chat_run E_random "$CFG_OFF" "$E_TOKENS" "$E_PROMPT" \
  HYDRO_DIR_MODE=random HYDRO_DIR_C="$DIR_C" HYDRO_DIR_SEED=1 HYDRO_DIR_TOKEN="$DIR_TOKEN"

python3 "$ROOT/scripts/score_durable_exit.py" "$OUT" | tee "$OUT/score.jsonl"
echo "DONE $OUT"
