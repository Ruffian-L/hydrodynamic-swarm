#!/usr/bin/env bash
# =============================================================================
# Hydrodynamic Swarm — easy launcher
# =============================================================================
# Usage:
#   ./run_swarm.sh
#   ./run_swarm.sh "your prompt"
#   ./run_swarm.sh "your prompt" 50
#
# Default: Gemma 3 4B Q4 + B4d-q (physics frozen; ~65 tok clean-paragraph zone)
# SplatLens museum (view-only, no GPU): ./splat-lens
# Generate + museum: ./demo_slice.sh
# Share doc: docs/MODEL_SIZE_PHYSICS_SCALING.md
# =============================================================================

set -euo pipefail

# ── CONFIG (edit these) ─────────────────────────────────────────────────────
PROMPT="${1:-Explain the Physics of Friendship in one short paragraph.}"
# 4B useful ceiling ~50–70; default hard budget in the good zone
TOKENS="${2:-65}"

# Small model prime path (gemma3 arch — loader works):
MODEL="data/google/gemma-3-4b-it-Q4_K_M.gguf"
TOKENIZER="data/google/tokenizer.json"

# 27B Q4 — use config.27b.toml (B4d lessons ported), not 4B B4d-q knobs:
#   cp config.27b.toml config.toml
# MODEL="data/google/gemma-3-27b-it-Q4_K_M.gguf"

# gemma3n E4B — NEEDS new loader (AltUp/Laurel/PLE), do not use yet:
# MODEL="data/google/google_gemma-3n-E4B-it-Q5_K_M.gguf"

# 27B Q8 (higher fidelity, slower):
# MODEL="data/google/gemma-3-27b-it-Q8_0.gguf"

CLEAR_MEMORY=1
EXTRA_FLAGS=""

export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BIN="$ROOT/target/release/hydrodynamic-swarm"

echo "=============================================="
echo "  Hydrodynamic Swarm launcher"
echo "=============================================="
echo "  model:      $MODEL"
echo "  tokens:     $TOKENS"
echo "  clear_mem:  $CLEAR_MEMORY"
echo "  config:     $ROOT/config.toml"
echo "  prompt:     $PROMPT"
echo "=============================================="
echo

[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER" ]] || { echo "ERROR: tokenizer not found: $TOKENIZER" >&2; exit 1; }

if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
  echo
fi

ARGS=(--model "$MODEL" --tokenizer "$TOKENIZER" --prompt "$PROMPT" --tokens "$TOKENS")
[[ "$CLEAR_MEMORY" == "1" ]] && ARGS+=(--clear-memory)
if [[ -n "$EXTRA_FLAGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( $EXTRA_FLAGS )
  ARGS+=("${EXTRA[@]}")
fi

echo "[*] Running: $BIN ${ARGS[*]}"
echo "--- live: tail -f logs/live.txt ---"
echo
exec "$BIN" "${ARGS[@]}"
