#!/usr/bin/env bash
# Memory coupling smokes A–D for TEAM_GOAL_MEMORY_COUPLING.md
# Run from repo root (workbench or main). Needs GPU + 4B model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-data/google/tokenizer.json}"
BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
PROMPT="${PROMPT:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${TOKENS:-65}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/memory_coupling_${STAMP}"
mkdir -p "$OUT"

source "$ROOT/scripts/cuda_env.sh"

if [[ ! -x "$BIN" ]]; then
  echo "[*] Building release..."
  cargo build --release --bin hydrodynamic-swarm
fi
[[ -f "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

run_one() {
  local id="$1"; shift
  echo
  echo "========== RUN $id =========="
  echo "  args: $*"
  # shellcheck disable=SC2086
  set +e
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    "$@" 2>&1 | tee "$OUT/${id}.stdout"
  local ec=${PIPESTATUS[0]}
  set -e
  # newest session log
  local jl
  jl="$(ls -t logs/*.jsonl 2>/dev/null | head -1 || true)"
  if [[ -n "$jl" ]]; then
    cp -f "$jl" "$OUT/${id}.jsonl"
  fi
  if [[ -f data/splat_memory.tct.json ]]; then
    cp -f data/splat_memory.tct.json "$OUT/${id}.tct.json" 2>/dev/null || true
  fi
  echo "  exit=$ec → $OUT/${id}.*"
  return 0
}

echo "Output dir: $OUT"
echo "Prompt: $PROMPT ($TOKENS tok)"

# A — clear, mint store
run_one A --clear-memory
ls -la data/splat_memory.safetensors data/splat_memory.tct data/splat_memory.tct.json 2>&1 | tee "$OUT/A_files.txt" || true
cp -f data/splat_memory.tct "$OUT/A_splat_memory.tct" 2>/dev/null || true
cp -f data/splat_memory.safetensors "$OUT/A_splat_memory.safetensors" 2>/dev/null || true

# B — reload store (no clear)
run_one B

# C — clear then import TCT from A
if [[ -f data/splat_memory.tct ]]; then
  run_one C --clear-memory --import-tct data/splat_memory.tct
else
  echo "SKIP C: no data/splat_memory.tct" | tee "$OUT/C.skip"
fi

# D — optional: user must zero forces in a side config; we document the expectation
cat > "$OUT/D_NOTE.txt" <<'EOF'
Run D (force-off with store loaded):
  Copy config.toml → config.force_off.toml and set
    splat_force_scale=0, goal_force_scale=0, field_wake_scale=0 (or equivalent)
  Then:
    cp config.force_off.toml config.toml
    run without --clear-memory
    restore B4d-q config.toml after
Echo: compare early F_s in B vs D.
EOF
echo "D: see $OUT/D_NOTE.txt (manual force-off config — do not leave B4d-q broken)"

# stub receipt
cat > "$OUT/RECEIPT_STUB.md" <<EOF
# Memory coupling receipt — $STAMP

Fill PASS/FAIL after Echo extracts forces from * .jsonl / stdout.

| Run | early_Fs | late_Fs | n_scars | pass? | note |
|-----|----------|---------|---------|-------|------|
| A | | | | | mint + save |
| B | | | | | reload couple |
| C | | | | | TCT import |
| D | | | | | force-off |

Answers:
1. Save across death?
2. F_s moves early on load?
3. TCT couples?
4. Narrative vs geometric?
5. One next target:

Signed: Shep ____  Echo ____  Lumina ____
EOF

echo
echo "Smokes A–C launched. Fill $OUT/RECEIPT_STUB.md → RECEIPT.md"
echo "GOAL file: TEAM_GOAL_MEMORY_COUPLING.md"
