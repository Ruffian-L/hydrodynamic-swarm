#!/usr/bin/env bash
# Three-arm loop receipt (full stack ON). Quiet run; clean table at end.
#   A = cold (empty wills, force ON)
#   B = warm (real store, force ON)
#   C = force_off + real store
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

BIN="${BIN:-$ROOT/target/release/hydrodynamic-swarm}"
MODEL="${MODEL:-$ROOT/data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-$ROOT/data/google/tokenizer.json}"
CFG_ON="${CFG_ON:-$ROOT/configs/gates/config.three_surface.toml}"
CFG_OFF="${CFG_OFF:-$ROOT/configs/profiles/config.force_off.toml}"
PROMPT="${PROMPT:-Explain the residual stream in one sentence.}"
TOKENS="${TOKENS:-48}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/logs/loop_three_arm_${STAMP}"
mkdir -p "$OUT"

SPLAT="$ROOT/data/splat_memory.safetensors"
BAK="$OUT/splat_memory.safetensors.real_backup"

[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 1; }
[[ -r "$SPLAT" ]] || { echo "missing $SPLAT" >&2; exit 1; }

cp -a "$SPLAT" "$BAK"
for f in data/splat_memory.tct data/splat_memory.tct.json; do
  [[ -f "$f" ]] && cp -a "$f" "$OUT/$(basename "$f").real_backup" || true
done

echo "LOOP three-arm  out=$OUT"
echo "prompt: $PROMPT"
echo "tokens=$TOKENS  model=$(basename "$MODEL")"
echo

# Pull one metric line from an arm log (quiet helpers).
metric() {
  local log="$1" key="$2"
  case "$key" in
    loaded)
      rg -o 'loaded=(true|false)' "$log" 2>/dev/null | head -1 | cut -d= -f2
      ;;
    wills)
      rg -o 'wills_start=[0-9]+' "$log" 2>/dev/null | head -1 | cut -d= -f2 \
        || rg -o 'scars_start=[0-9]+' "$log" 2>/dev/null | head -1 | cut -d= -f2
      ;;
    nearest)
      rg -o 'nearest_L2=[0-9.]+' "$log" 2>/dev/null | head -1 | cut -d= -f2
      ;;
    pot)
      rg -o 'pot=[0-9.eE+-]+' "$log" 2>/dev/null | head -1 | cut -d= -f2
      ;;
    bridges)
      rg -o 'bridges=[0-9]+' "$log" 2>/dev/null | head -1 | cut -d= -f2
      ;;
    text)
      # text after Full Decoded block — first non-empty line of content
      awk '/Full Decoded/{p=1;next} p&&/Phase 5|===/{exit} p&&NF{print; exit}' "$log" \
        | tr '\n' ' ' | cut -c1-72
      ;;
  esac
}

run_arm() {
  local name="$1"
  shift
  local log="$OUT/${name}.stdout"
  printf '  arm %s … ' "$name"
  set +e
  "$BIN" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --tokens "$TOKENS" \
    --prompt "$PROMPT" \
    --no-save-memory \
    --no-hud \
    "$@" \
    >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "ok"
  else
    echo "rc=$rc (see $log)"
  fi
  {
    echo "===== $name  rc=$rc ====="
    rg -n "Memory session:|Learned-will geometry|Scar geometry|Loaded .* splats|Full Decoded|force_cap|bridges=" "$log" 2>/dev/null || true
    echo
  } >>"$OUT/RECEIPT.txt"
}

run_arm A --config "$CFG_ON" --clear-memory
cp -a "$BAK" "$SPLAT"
for f in splat_memory.tct splat_memory.tct.json; do
  [[ -f "$OUT/${f}.real_backup" ]] && cp -a "$OUT/${f}.real_backup" "data/$f" || true
done

run_arm B --config "$CFG_ON"
run_arm C --config "$CFG_OFF"
cp -a "$BAK" "$SPLAT"

# --- clean comparison (the whole point) ---
{
  echo
  echo "================ THREE-ARM COMPARE ================"
  echo "OUT=$OUT"
  echo "PROMPT=$PROMPT"
  echo
  printf '%-4s %-8s %-7s %-10s %-10s %-8s %s\n' \
    "ARM" "loaded" "wills" "nearest_L2" "pot" "bridges" "text (72c)"
  printf '%-4s %-8s %-7s %-10s %-10s %-8s %s\n' \
    "----" "--------" "-------" "----------" "----------" "--------" "--------"
  for arm in A B C; do
    log="$OUT/${arm}.stdout"
    printf '%-4s %-8s %-7s %-10s %-10s %-8s %s\n' \
      "$arm" \
      "$(metric "$log" loaded || echo '?')" \
      "$(metric "$log" wills || echo '?')" \
      "$(metric "$log" nearest || echo '?')" \
      "$(metric "$log" pot || echo '?')" \
      "$(metric "$log" bridges || echo '?')" \
      "$(metric "$log" text || echo '?')"
  done
  echo
  echo "A=cold (no wills)  B=warm (wills+force)  C=force_off+wills"
  echo "GO if B nearer/higher pot than A with loaded=true"
  echo "==================================================="
  echo "detail: $OUT/RECEIPT.txt  logs: A|B|C.stdout"
} | tee -a "$OUT/COMPARE.txt"

echo
echo "done. table above + $OUT/COMPARE.txt"
