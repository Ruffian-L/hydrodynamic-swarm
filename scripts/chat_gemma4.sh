#!/usr/bin/env bash
# Multi-turn chat with local Gemma 4 (minimal wrap, greedy near-vanilla).
# Usage:
#   ./scripts/chat_gemma4.sh
#   ./scripts/chat_gemma4.sh 48   # max tokens per reply

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

TOKENS="${100:-300}"
MODEL="${GEMMA4_MODEL:-/home/ruffianl/Downloads/unsloth_gemma-4-31B-it-Q4_K_M.gguf}"
TOKENIZER="${GEMMA4_TOKENIZER:-/home/ruffianl/Downloads/tokenizer.json}"
CONFIG="${GEMMA4_CONFIG:-configs/gemma4/config.gemma4_greedy.toml}"

exec ./target/release/hydrodynamic-swarm \
  --chat \
  --config "$CONFIG" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokens "$TOKENS"
