#!/usr/bin/env bash
# Multi-pass Gemma smoke — do NOT judge from a single cold generation.
#
# Observed pattern: first tokens / first embed can dump garbage, then a few
# coherent tokens, then collapse again. Always run ≥2 cold passes + multi-turn.
#
# Usage (from repo root):
#   ./scripts/multipass_smoke.sh
#   TOKENS=80 PASSES=2 ./scripts/multipass_smoke.sh
#   GEMMA4_MODEL=... GEMMA4_CONFIG=... ./scripts/multipass_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=cuda_env.sh
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

BIN="${BIN:-./target/release/hydrodynamic-swarm}"
MODEL="${GEMMA4_MODEL:-data/google/gemma-4-12b-it-Q4_K_M.gguf}"
TOK="${GEMMA4_TOKENIZER:-data/google/gemma4_assets/tokenizer.json}"
CFG="${GEMMA4_CONFIG:-configs/ablation/config_isolation_baseline.toml}"
TOKENS="${TOKENS:-64}"
PASSES="${PASSES:-2}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT="logs/multipass_smoke_${STAMP}.txt"
SUM="logs/multipass_smoke_${STAMP}.summary.txt"

mkdir -p logs
{
  echo "=== multipass_smoke $STAMP ==="
  echo "model=$MODEL cfg=$CFG tokens=$TOKENS cold_passes=$PASSES"
  echo "rule: ≥${PASSES} cold oneshots per prompt + 1 multi-turn chat (same load)"
} | tee "$OUT"

PROMPTS=(
  "Repeat exactly: the quick brown fox jumps over the lazy dog"
  "What is 17 * 23? Show the arithmetic."
  "Explain residual stream in one short paragraph."
)

extract_decoded() {
  awk '
    /=== Full Decoded Output ===/ {grab=1; next}
    /--- Phase 5/ || /=== SplatRAG/ || /========== END/ {if(grab){grab=0}}
    grab {print}
  '
}

# Score a blob of text: printable-latin ratio + crude "englishy" token presence
score_text() {
  python3 - "$@" <<'PY'
import sys, re
text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
text = text.strip()
if not text:
    print("empty|latin=0|weird=1|words=0")
    sys.exit(0)
# latin letters / total non-space
chars = [c for c in text if not c.isspace()]
latin = sum(1 for c in chars if ("A" <= c <= "Z") or ("a" <= c <= "z") or c in ".,!?;:'\"-()[]")
ratio = latin / max(len(chars), 1)
# weird: CJK / heavy symbols / long same-token loops
cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\uac00" <= c <= "\ud7af")
loop = 1 if re.search(r"(.{4,40})\1{3,}", text) or "WikipediaLab" in text else 0
words = len(re.findall(r"[A-Za-z]{3,}", text))
print(f"latin={ratio:.2f}|cjk={cjk}|loop={loop}|words={words}|len={len(text)}")
# early / mid / late thirds for mid-run recovery detection
n = len(text)
if n >= 30:
    thirds = [text[: n//3], text[n//3 : 2*n//3], text[2*n//3 :]]
    for i, t in enumerate(["early", "mid", "late"]):
        ch = [c for c in t if not c.isspace()]
        lat = sum(1 for c in ch if c.isascii() and (c.isalpha() or c in ".,!?;:'\"-()")) / max(len(ch), 1)
        print(f"  slice_{t}: latin={lat:.2f} head={t!r}"[:80])
PY
}

echo "" | tee -a "$OUT"
echo "### A) COLD ONESHOTS (${PASSES}x each prompt) ###" | tee -a "$OUT"

declare -a SUMMARY_LINES=()
for p in "${PROMPTS[@]}"; do
  for ((pass=1; pass<=PASSES; pass++)); do
    echo "" | tee -a "$OUT"
    echo "========== COLD pass=${pass}/${PASSES} | $p ==========" | tee -a "$OUT"
    set +e
    full="$("$BIN" --config "$CFG" --model "$MODEL" --tokenizer "$TOK" \
      --prompt "$p" --tokens "$TOKENS" --clear-memory --no-save-memory \
      --no-endocrine --no-termsplat --no-hud 2>&1)"
    ec=$?
    set -e
    printf '%s\n' "$full" >>"$OUT"
    decoded="$(printf '%s\n' "$full" | extract_decoded | sed '/^$/d')"
    echo "--- decoded ---" | tee -a "$OUT"
    printf '%s\n' "$decoded" | tee -a "$OUT"
    sc="$(printf '%s\n' "$decoded" | score_text "$(printf '%s' "$decoded")")"
    echo "SCORE: $sc" | tee -a "$OUT"
    SUMMARY_LINES+=("cold p=${pass} exit=${ec} | ${p:0:40}… | $sc")
    echo "========== END exit=$ec ==========" | tee -a "$OUT"
  done
done

echo "" | tee -a "$OUT"
echo "### B) MULTI-TURN CHAT (one load, 3 turns) ###" | tee -a "$OUT"
# stdin turns then quit
CHAT_IN=$(mktemp)
cat >"$CHAT_IN" <<EOF
Repeat exactly: the quick brown fox jumps over the lazy dog
What is 17 * 23? Show the arithmetic.
Explain residual stream in one short paragraph.
quit
EOF

set +e
chat_out="$("$BIN" --chat --config "$CFG" --model "$MODEL" --tokenizer "$TOK" \
  --tokens "$TOKENS" --clear-memory --no-save-memory \
  --no-endocrine --no-termsplat --no-hud <"$CHAT_IN" 2>&1)"
chat_ec=$?
set -e
printf '%s\n' "$chat_out" >>"$OUT"
rm -f "$CHAT_IN"

# Pull assistant-ish blocks (lines after you> responses often marked by live stream)
echo "--- chat excerpt (assistant-ish lines) ---" | tee -a "$OUT"
printf '%s\n' "$chat_out" | grep -E '^(The |It |Okay|Hello|Friend|Gravity|17|23|residual|Wikipedia|correct|pleasure|Full Decoded|you>|===)' | head -80 | tee -a "$OUT" || true
echo "chat exit=$chat_ec" | tee -a "$OUT"

{
  echo "=== SUMMARY $STAMP ==="
  for line in "${SUMMARY_LINES[@]}"; do echo "$line"; done
  echo "chat_exit=$chat_ec"
  echo "full log: $OUT"
  echo ""
  echo "How to read: if pass1 garbage and pass2 clean → warm-up/embed issue."
  echo "If early slice bad, mid good, late bad → mid-run degeneration (not just cold start)."
  echo "If both passes identical nonsense → base forward/template, not stochastic warm-up."
} | tee "$SUM" | tee -a "$OUT"

echo "WROTE $OUT"
echo "WROTE $SUM"
