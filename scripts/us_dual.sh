#!/usr/bin/env bash
# =============================================================================
# Us dual run — speaker + weather (stateless)
# =============================================================================
# Terminal A runs this. Optional enzyme if ENDOCRINE_URL already set or
# START_ENZYME=1.
#
#   ./scripts/us_dual.sh
#   ./scripts/us_dual.sh "your prompt" 40
#   START_ENZYME=1 ./scripts/us_dual.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROMPT="${1:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${2:-40}"

TERMSPLAT_BIN="${TERMSPLAT_BIN:-$ROOT/../termsplat/target/release/termsplat}"
if [[ ! -x "$TERMSPLAT_BIN" ]]; then
  TERMSPLAT_BIN="$ROOT/../termsplat/target/debug/termsplat"
fi

echo "=============================================="
echo "  us dual — speaker + TermSplat weather"
echo "=============================================="
echo "  prompt:  $PROMPT"
echo "  tokens:  $TOKENS"
echo "=============================================="

if [[ "${START_ENZYME:-0}" == "1" ]]; then
  ./scripts/endocrine_enzyme.sh start || true
  export ENDOCRINE_URL="${ENDOCRINE_URL:-http://127.0.0.1:8210/v1}"
  export ENDOCRINE_MODEL="${ENDOCRINE_MODEL:-local}"
  echo "  enzyme:  ENDOCRINE_URL=$ENDOCRINE_URL"
fi

WEATHER_PID=""
if [[ -x "$TERMSPLAT_BIN" ]]; then
  # follow weather as soon as latest.termsplat.jsonl appears
  (
    for _ in $(seq 1 90); do
      [[ -e logs/latest.termsplat.jsonl ]] && break
      sleep 0.5
    done
    exec "$TERMSPLAT_BIN" pipe logs/latest.termsplat.jsonl --follow --ms 60
  ) &
  WEATHER_PID=$!
  echo "  weather: termsplat pipe pid=$WEATHER_PID"
else
  echo "  weather: termsplat binary missing — build termsplat or paint later:"
  echo "           termsplat pipe logs/latest.termsplat.jsonl"
fi

cleanup() {
  if [[ -n "${WEATHER_PID:-}" ]] && kill -0 "$WEATHER_PID" 2>/dev/null; then
    kill "$WEATHER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# speaker (blocks)
./run_swarm.sh "$PROMPT" "$TOKENS"
echo
echo "  weather file: logs/latest.termsplat.jsonl"
echo "  re-paint: termsplat pipe logs/latest.termsplat.jsonl"
