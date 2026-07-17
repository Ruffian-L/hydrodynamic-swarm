#!/usr/bin/env bash
# Interactive Hydrodynamic Swarm chat wrapper.
# Usage:
#   ./scripts/chat.sh [tokens] [model_path] [tokenizer_path]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh"

TOKENS="${1:-160}"
MODEL="${2:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${3:-$(dirname "$MODEL")/tokenizer.json}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Pass a path, or download into data/google/ (see SETUP.md / ./splat-lens check)." >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER" ]]; then
  echo "Tokenizer not found: $TOKENIZER" >&2
  exit 1
fi

exec cargo run --release --bin hydrodynamic-swarm -- \
    --chat \
    --tokens "$TOKENS" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER"
