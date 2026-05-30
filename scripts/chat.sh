#!/bin/bash
# Interactive Hydrodynamic Swarm chat wrapper.
# Usage:
#   ./scripts/chat.sh [tokens] [model_path] [tokenizer_path]

set -euo pipefail

TOKENS="${1:-160}"
MODEL="${2:-/home/ruff/projects/Homernd/team_build/niodoo/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf}"
TOKENIZER="${3:-$(dirname "$MODEL")/tokenizer.json}"

export RUSTFLAGS="${RUSTFLAGS:--C target-feature=+fp16}"

exec cargo run --bin hydrodynamic-swarm -- \
    --chat \
    --tokens "$TOKENS" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER"
