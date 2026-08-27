#!/usr/bin/env bash
# John A0 SWA integrity gate — static empty-key geometry + live finite prefill.
# Usage:
#   ./scripts/a0_swa_check.sh                 # default 31B from models desk
#   ./scripts/a0_swa_check.sh 12b
#   ./scripts/a0_swa_check.sh 31b
#   ./scripts/a0_swa_check.sh /path/to.gguf
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

MODELS_DESK="${GEMMA4_MODELS_DESK:-/media/ruffianl/ghost_team/models}"
TOKENIZER="${GEMMA4_TOKENIZER:-$ROOT/data/google/gemma4_assets/tokenizer.json}"
CHOICE="${1:-31b}"

case "$CHOICE" in
  31b|31B)
    MODEL="${GEMMA4_MODEL:-$MODELS_DESK/bart_google_gemma-4-31B-it-Q4_K_M.gguf}"
    ;;
  12b|12B)
    MODEL="${GEMMA4_MODEL:-$MODELS_DESK/gemma-4-12b-it-Q4_K_M.gguf}"
    ;;
  4b|3-4b|g3)
    MODEL="${GEMMA4_MODEL:-$ROOT/data/google/gemma-3-4b-it-Q4_K_M.gguf}"
    TOKENIZER="${GEMMA4_TOKENIZER:-$ROOT/data/google/tokenizer.json}"
    ;;
  *)
    MODEL="$CHOICE"
    ;;
esac

if [[ ! -r "$MODEL" ]]; then
  printf 'model not readable: %s\n' "$MODEL" >&2
  printf 'desk: %s\n' "$MODELS_DESK" >&2
  exit 1
fi
if [[ ! -r "$TOKENIZER" ]]; then
  printf 'tokenizer not readable: %s\n' "$TOKENIZER" >&2
  exit 1
fi

BIN="$ROOT/target/release/hydrodynamic-swarm"
if [[ ! -x "$BIN" ]]; then
  printf 'building release binary…\n' >&2
  cargo build --release
fi

printf 'A0 model:     %s\n' "$MODEL"
printf 'A0 tokenizer: %s\n' "$TOKENIZER"
exec "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" --a0-swa-check
