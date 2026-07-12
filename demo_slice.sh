#!/usr/bin/env bash
# =============================================================================
# Easy demo slice — B4d-q (4B, ~65 tok) + SplatLens
#
# Prefer for viewing only (no GPU):
#   ./splat-lens
#
# This script GENERATES a new recording (needs CUDA + model), then opens museum:
#   ./demo_slice.sh
#   ./demo_slice.sh "your short prompt"
#   ./demo_slice.sh "your short prompt" 65
# =============================================================================
set -euo pipefail
export PATH="/usr/local/cuda-13.1/bin:${PATH:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROMPT="${1:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${2:-65}"
MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="$ROOT/target/release/hydrodynamic-swarm"
PORT="${DEMO_PORT:-8765}"

echo "=============================================="
echo "  Hydrodynamic Swarm — SLICE DEMO"
echo "=============================================="
echo "  model:   $MODEL"
echo "  config:  config.toml (B4d-q)"
echo "  tokens:  $TOKENS"
echo "  viz:     ON → .viz.json + SplatLens"
echo "  prompt:  $PROMPT"
echo "=============================================="
echo

[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER" ]] || { echo "ERROR: tokenizer not found: $TOKENIZER" >&2; exit 1; }

if [[ ! -x "$BIN" ]] || [[ -n "$(find src -name '*.rs' -newer "$BIN" 2>/dev/null | head -1)" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
  echo
fi

echo "[*] Generating (clear memory, --viz)..."
"$BIN" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "$PROMPT" \
  --tokens "$TOKENS" \
  --clear-memory \
  --viz

# Newest viz session
VIZ=$(ls -t logs/*.viz.json 2>/dev/null | head -1 || true)
if [[ -z "$VIZ" || ! -f "$VIZ" ]]; then
  echo "ERROR: no .viz.json produced. Is --viz wired?" >&2
  exit 1
fi

# Pack museum-sized demo + latest pointer (subsample field for git-friendly size)
python3 - "$VIZ" "$ROOT/tools/museum/demos/custom-latest.viz.json" "$ROOT/tools/latest_demo.viz.json" <<'PY'
import json, sys
src, dst, latest = sys.argv[1:4]
with open(src) as f:
    d = json.load(f)
fp = d.get("field_points_3d") or []
if len(fp) > 800:
    step = max(1, len(fp) // 800)
    d["field_points_3d"] = fp[::step][:800]
for path in (dst, latest):
    with open(path, "w") as f:
        json.dump(d, f, separators=(",", ":"))
print("packed", dst)
PY

echo
echo "[*] Session: $VIZ"
echo "[*] Museum:  tools/museum/demos/custom-latest.viz.json"
echo "[*] Opening SplatLens museum (./splat-lens museum)"
echo

exec "$ROOT/splat-lens" museum
