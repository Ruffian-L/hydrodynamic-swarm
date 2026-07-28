#!/usr/bin/env bash
# Multi-turn chat with local Gemma 4 — **stable gen defaults** (not greedy soup).
# Usage:
#   ./scripts/chat_gemma4.sh
#   ./scripts/chat_gemma4.sh 48
#   GEMMA4_CONFIG=configs/gemma4/config.gemma4_greedy.toml ./scripts/chat_gemma4.sh  # old probe
#
# Commands in chat: quit/exit · reset (clear history)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

TOKENS="${1:-80}"
MODEL="${GEMMA4_MODEL:-/home/ruffianl/Downloads/unsloth_gemma-4-31B-it-Q4_K_M.gguf}"
TOKENIZER="${GEMMA4_TOKENIZER:-/home/ruffianl/Downloads/tokenizer.json}"
CONFIG="${GEMMA4_CONFIG:-configs/gemma4/config.gemma4_stable.toml}"

exec ./target/release/hydrodynamic-swarm \
  --chat \
  --config "$CONFIG" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokens "$TOKENS"
