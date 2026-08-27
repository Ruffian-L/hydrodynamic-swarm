#!/usr/bin/env bash
#
# talk.sh — HUMAN multi-turn chat (same settings as smoke_convo.sh)
#
#   ./scripts/talk.sh
#
# Defaults from scripts/convo_defaults.sh so Jason / Grok / Shep / Echo
# share one config + model + flags. Variance = real, not "wrong script".
#
# Research: one-shot is not multi-turn ready.
#   research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md
#
# Optional: pick another GGUF interactively (still same config/flags).
#   HYDRO_MODEL=... HYDRO_CONFIG=... HYDRO_TOKENS=80 ./scripts/talk.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
# shellcheck source=convo_defaults.sh
source "$ROOT/scripts/convo_defaults.sh"

BIN="$ROOT/target/release/hydrodynamic-swarm"

if [[ -t 1 ]]; then
  DIM=$'\033[2m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
else
  DIM=""; BOLD=""; OFF=""
fi
say() { printf '%s\n' "$*"; }
blank() { printf '\n'; }

# ---------------------------------------------------------------- model pick
# If HYDRO_MODEL already set and readable, use it (parity with smoke).
# Else offer a short menu of Gemma 4 GGUFs so humans don't path-hunt.

MODEL="$HYDRO_MODEL"
if [[ ! -r "$MODEL" ]]; then
  SEARCH_DIRS=(
    "${HYDRO_MODEL_DIRS:-}"
    "$HOME/models/gemma4"
    "$HOME/models"
    "$ROOT/data/google"
    "/media/ruffianl/ghost_team/models"
  )
  MODEL_PATHS=()
  seen=""
  for dir in "${SEARCH_DIRS[@]}"; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    while IFS= read -r found; do
      real="$(readlink -f "$found" 2>/dev/null || printf '%s' "$found")"
      case "$seen" in *"|$real|"*) continue ;; esac
      seen="$seen|$real|"
      # Prefer Gemma 4 in the menu for this lane
      case "$(basename "$found")" in
        *emma-4*|*emma4*|*EMMA-4*) MODEL_PATHS+=("$found") ;;
      esac
    done < <(find "$dir" -maxdepth 1 -name '*.gguf' -readable 2>/dev/null | sort)
  done
  if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
    say "No Gemma 4 .gguf found. Set HYDRO_MODEL=/path/to.gguf"
    exit 1
  fi
  blank
  say "  ${BOLD}Which Gemma 4? (same config as smoke_convo)${OFF}"
  blank
  i=1
  for path in "${MODEL_PATHS[@]}"; do
    printf '   %2d    %s\n' "$i" "$(basename "$path")"
    say "         ${DIM}$(dirname "$path")${OFF}"
    blank
    i=$((i + 1))
  done
  printf '  > '
  read -r choice
  choice="${choice:-1}"
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#MODEL_PATHS[@]} )); then
    say "Invalid choice."
    exit 1
  fi
  MODEL="${MODEL_PATHS[$((choice - 1))]}"
fi

TOKENIZER="$HYDRO_TOKENIZER"
if [[ ! -r "$TOKENIZER" ]]; then
  for candidate in \
    "$ROOT/data/qwen.tokenizer.json" \
    "$(dirname "$MODEL")/tokenizer.json" \
    "$ROOT/data/google/gemma4_assets/tokenizer.json"
  do
    [[ -r "$candidate" ]] && { TOKENIZER="$candidate"; break; }
  done
fi
if [[ ! -r "$TOKENIZER" ]]; then
  say "Tokenizer missing. HYDRO_TOKENIZER=/path/to/tokenizer.json"
  exit 1
fi

CONFIG="$HYDRO_CONFIG"
[[ -r "$CONFIG" ]] || CONFIG="$ROOT/config.toml"
TOKENS="$HYDRO_TOKENS"

# rebuild if src newer than binary
needs_build=0
if [[ ! -x "$BIN" ]]; then
  needs_build=1
elif [[ -n "$(find "$ROOT/src" -name '*.rs' -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
  needs_build=1
fi
if [[ $needs_build -eq 1 ]]; then
  blank
  say "  Rebuilding (source newer than binary)..."
  blank
  cargo build --release || exit 1
fi

blank
say "  ${BOLD}talk.sh — HUMAN multi-turn${OFF}"
blank
say "     Model     $(basename "$MODEL")"
say "     Config    $CONFIG"
say "     Tokens    $TOKENS"
say "     Flags     ${HYDRO_CHAT_FLAGS[*]}"
say "     Same as   ./scripts/smoke_convo.sh  (convo_defaults.sh)"
blank
say "  ${DIM}Type messages. reset = clear history. quit = leave.${OFF}"
blank
say "  ${DIM}One-shot is not the test. Stay multi-turn.${OFF}"
blank

exec "$BIN" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokens "$TOKENS" \
  "${HYDRO_CHAT_FLAGS[@]}"
