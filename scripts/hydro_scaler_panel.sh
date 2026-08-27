#!/usr/bin/env bash
# Frozen matched scaler panel over the locked Niodoo Official 10 collaborator pack.
#
#   ./scripts/hydro_scaler_panel.sh check
#   ./scripts/hydro_scaler_panel.sh first
#   ./scripts/hydro_scaler_panel.sh arm piecewise 1.0
#   ./scripts/hydro_scaler_panel.sh full   # 12 long model runs; explicit only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CMD="${1:-check}"
SEED="${HYDRO_PANEL_SEED:-424242}"
PACK="/home/ruffianl/Hub/Projects/niodoo/NIODOO_OFFICIAL_PROMPT_PACK.md"
EVAL_NAME="official-10"

[[ -r "$PACK" ]] || { echo "missing locked prompt pack: $PACK" >&2; exit 2; }

panel_header() {
  local tokens
  tokens="$(awk -F= '$1 == "HYDRO_TOKENS" { print $2 }' "$ROOT/evals/$EVAL_NAME/eval.env")"
  echo "panel=hydro-scaler-official10-v1"
  echo "prompt_pack=$PACK"
  echo "prompt_pack_sha256=$(sha256sum "$PACK" | awk '{print $1}')"
  echo "eval=$EVAL_NAME"
  echo "seed=$SEED"
  echo "max_tokens=$tokens"
  echo "memory_start=empty (hydro_eval WIPE_STORE=1 before every arm)"
  echo "temperature/ramp/logit/governor=frozen by hydro-residual-profile-relative/v1"
  echo "tda_monitor=off (frozen; model-emitted control tags and lock remain on)"
}

run_arm() {
  local rule="$1"
  local gain="$2"
  case "$rule" in
    legacy|8b-sqrt|piecewise|off) ;;
    *) echo "invalid rule: $rule" >&2; exit 2 ;;
  esac
  case "$gain" in
    0.5|1.0|1.5) ;;
    *) echo "invalid gain: $gain (allowed: 0.5 1.0 1.5)" >&2; exit 2 ;;
  esac

  panel_header
  echo "arm=${rule}_k${gain}"
  export HYDRO_SIZE_RULE="$rule"
  export HYDRO_SCALER_GAIN="$gain"
  export HYDRO_SCALER_APPLY=1
  export HYDRO_SAMPLE_SEED="$SEED"
  ./scripts/hydro_eval.sh run "$EVAL_NAME"
}

case "$CMD" in
  check)
    panel_header
    ./scripts/hydro_eval.sh check "$EVAL_NAME"
    ;;
  first)
    run_arm piecewise 0.5
    ;;
  arm)
    [[ $# -eq 3 ]] || { echo "usage: $0 arm <legacy|8b-sqrt|piecewise|off> <0.5|1.0|1.5>" >&2; exit 2; }
    run_arm "$2" "$3"
    ;;
  full)
    for rule in legacy 8b-sqrt piecewise off; do
      for gain in 0.5 1.0 1.5; do
        run_arm "$rule" "$gain"
      done
    done
    ;;
  *)
    echo "usage: $0 check | first | arm <rule> <gain> | full" >&2
    exit 2
    ;;
esac
