#!/usr/bin/env bash
# =============================================================================
# Hydrodynamic Swarm — easy launcher
# =============================================================================
# Usage:
#   ./run_swarm.sh
#   ./run_swarm.sh "your prompt"
#   ./run_swarm.sh "your prompt" 80
#
# Physics knobs live in config.toml (edit that for F_s damp, force_cap, etc.).
# Only edit the CONFIG block below for model/prompt/tokens.
# =============================================================================

set -euo pipefail

# ── CONFIG (edit these) ─────────────────────────────────────────────────────
PROMPT="${1:-Explain the Physics of Friendship in one paragraph.}"
TOKENS="${2:-80}"

MODEL="data/google/gemma-3-27b-it-Q8_0.gguf"
TOKENIZER="data/google/tokenizer.json"

# Llama fallback:
# MODEL="/home/ruffianl/projects/niodoo-live/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
# TOKENIZER="/home/ruffianl/projects/niodoo-live/model/tokenizer.json"

# 1 = wipe splat memory before run (recommended while tuning)
CLEAR_MEMORY=1

# Extra flags, e.g. "--viz"
EXTRA_FLAGS=""

# 1 = verify model/tokenizer against data/google/SHA256SUMS before run
# (first check hashes the file; later runs use a local cache unless the file changes)
VERIFY_SHAS=1

export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"
# =============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BIN="$ROOT/target/release/hydrodynamic-swarm"
SHA_MANIFEST="$ROOT/data/google/SHA256SUMS"

lookup_expected_sha() {
  local base="$1"
  awk -v name="$base" '$2 == name { print $1; exit }' "$SHA_MANIFEST"
}

verify_pinned_file() {
  local file="$1"
  local base expected actual size mtime stamp_hash stamp_size stamp_mtime

  if [[ ! -f "$SHA_MANIFEST" ]]; then
    echo "WARN: SHA manifest missing: $SHA_MANIFEST (skipping pin check)" >&2
    return 0
  fi

  base="$(basename "$file")"
  expected="$(lookup_expected_sha "$base" || true)"
  if [[ -z "$expected" ]]; then
    echo "  $base: not in SHA256SUMS (skipping)"
    return 0
  fi

  size=$(stat -c%s "$file")
  mtime=$(stat -c%Y "$file")
  stamp="${file}.sha256.verified"

  if [[ -f "$stamp" ]]; then
    read -r stamp_hash stamp_size stamp_mtime <"$stamp" || true
    if [[ "$stamp_hash" == "$expected" && "$stamp_size" == "$size" && "$stamp_mtime" == "$mtime" ]]; then
      echo "  $base: pinned OK"
      return 0
    fi
  fi

  echo "  $base: hashing (one-time unless the file changes)..."
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: SHA256 mismatch for $base" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi

  printf '%s %s %s\n' "$actual" "$size" "$mtime" >"$stamp"
  echo "  $base: pinned OK"
}

echo "=============================================="
echo "  Hydrodynamic Swarm launcher"
echo "=============================================="
echo "  dir:        $ROOT"
echo "  model:      $MODEL"
echo "  tokenizer:  $TOKENIZER"
echo "  tokens:     $TOKENS"
echo "  clear_mem:  $CLEAR_MEMORY"
echo "  config:     $ROOT/config.toml"
echo "  prompt:     $PROMPT"
echo "=============================================="
echo "  Tip: damp F_s via config.toml → splat_force_scale / splat_force_max"
echo "  Live: tail -f logs/live.txt"
echo "=============================================="
echo

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER" ]]; then
  echo "ERROR: tokenizer not found: $TOKENIZER" >&2
  exit 1
fi

if [[ "$VERIFY_SHAS" == "1" ]]; then
  echo "[*] Verifying pinned model files..."
  verify_pinned_file "$MODEL"
  verify_pinned_file "$TOKENIZER"
  echo
fi

if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]] \
   || [[ -f build.rs && build.rs -nt "$BIN" ]]; then
  echo "[*] Building release binary..."
  cargo build --release --bin hydrodynamic-swarm
  echo
fi

ARGS=(
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --prompt "$PROMPT"
  --tokens "$TOKENS"
)
if [[ "$CLEAR_MEMORY" == "1" ]]; then
  ARGS+=(--clear-memory)
fi
if [[ -n "$EXTRA_FLAGS" ]]; then
  # intentional word-split for user-supplied flags
  # shellcheck disable=SC2206
  EXTRA=( $EXTRA_FLAGS )
  ARGS+=("${EXTRA[@]}")
fi

echo "[*] Running..."
echo "    $BIN ${ARGS[*]}"
echo

exec "$BIN" "${ARGS[@]}"
