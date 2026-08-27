#!/usr/bin/env bash
# Multi-turn Gemma 4 target with the live three-surface slider panel.
# Usage:
#   ./scripts/chat_gemma4.sh
#   ./scripts/chat_gemma4.sh 48
#   GEMMA4_CONFIG=configs/gemma4/config.gemma4_stable.toml ./scripts/chat_gemma4.sh
#
# Commands in chat: /tui · /phys · /set name value · reset · quit/exit

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

TOKENS="${1:-80}"
# Prefer env → local data/google link → ghost_team models desk (real store).
MODELS_DESK="${GEMMA4_MODELS_DESK:-/media/ruffianl/ghost_team/models}"
if [[ -n "${GEMMA4_MODEL:-}" ]]; then
  MODEL="$GEMMA4_MODEL"
elif [[ -r "$ROOT/data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf" ]]; then
  MODEL="$ROOT/data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf"
elif [[ -r "$MODELS_DESK/bart_google_gemma-4-31B-it-Q4_K_M.gguf" ]]; then
  MODEL="$MODELS_DESK/bart_google_gemma-4-31B-it-Q4_K_M.gguf"
else
  MODEL="$ROOT/data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf"
fi
TOKENIZER="${GEMMA4_TOKENIZER:-$ROOT/data/google/gemma4_assets/tokenizer.json}"
CONFIG="${GEMMA4_CONFIG:-$ROOT/configs/gates/config.three_surface.toml}"

for REQUIRED in "$MODEL" "$TOKENIZER" "$CONFIG"; do
  if [[ ! -r "$REQUIRED" ]]; then
    printf 'Gemma 4 target file is not readable: %s\n' "$REQUIRED" >&2
    printf 'Models desk: %s\n' "$MODELS_DESK" >&2
    printf 'Override: GEMMA4_MODEL=/path/to.gguf %s\n' "$0" >&2
    exit 1
  fi
done

printf 'Gemma 4 model:     %s\n' "$MODEL"
printf 'Gemma 4 tokenizer: %s\n' "$TOKENIZER"
printf 'Physics config:    %s\n' "$CONFIG"
printf 'Open sliders with /tui after the chat prompt appears.\n'

exec "$ROOT/target/release/hydrodynamic-swarm" \
  --chat \
  --config "$CONFIG" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokens "$TOKENS"
