#!/usr/bin/env bash
# Mid-conversation collapse probe.
#
# Feeds multi-turn chat under COLLAPSE_PROBE logging, then scores:
#   - first turn/token where garbage appears
#   - residual_norm / entropy / margin trends before that
#   - short vs long previous assistant turns
#
# Usage:
#   ./scripts/collapse_probe.sh
#   TOKENS=80 ./scripts/collapse_probe.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

BIN="${BIN:-./target/release/hydrodynamic-swarm}"
MODEL="${GEMMA4_MODEL:-data/google/gemma-4-12b-it-Q4_K_M.gguf}"
TOK="${GEMMA4_TOKENIZER:-data/google/gemma4_assets/tokenizer.json}"
# Default: isolation baseline (physics OFF). Use PHYSICS_ON=1 for physics ON.
if [ "${PHYSICS_ON:-0}" = "1" ]; then
  CFG="${GEMMA4_CONFIG:-configs/ablation/config_collapse_physics_on.toml}"
else
  CFG="${GEMMA4_CONFIG:-configs/ablation/config_isolation_baseline.toml}"
fi
TOKENS="${TOKENS:-64}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
PROBE="logs/collapse_probe_${STAMP}.jsonl"
TRANS="logs/collapse_probe_${STAMP}.transcript.txt"
SUM="logs/collapse_probe_${STAMP}.summary.md"
export COLLAPSE_PROBE="$PROBE"

mkdir -p logs
: >"$PROBE"

echo "=== collapse_probe $STAMP ===" | tee "$TRANS"
echo "model=$MODEL cfg=$CFG tokens=$TOKENS probe=$PROBE" | tee -a "$TRANS"

run_scenario() {
  local name="$1"
  shift
  local infile
  infile=$(mktemp)
  printf '%s\n' "$@" >"$infile"
  echo "quit" >>"$infile"
  echo "" | tee -a "$TRANS"
  echo "### SCENARIO: $name ###" | tee -a "$TRANS"
  echo "{\"event\":\"scenario\",\"name\":\"$name\"}" >>"$PROBE"
  set +e
  COLLAPSE_PROBE="$PROBE" "$BIN" --chat --config "$CFG" --model "$MODEL" --tokenizer "$TOK" \
    --tokens "$TOKENS" --clear-memory --no-save-memory \
    --no-endocrine --no-termsplat --no-hud <"$infile" 2>&1 | tee -a "$TRANS"
  set -e
  rm -f "$infile"
}

# A) short user turns, force short assistant (low max_tokens already) — stack many
run_scenario "A_short_stack" \
  "Say hi in three words." \
  "What is 2+2?" \
  "Name one color." \
  "Spell cat." \
  "Yes or no: sky is blue?" \
  "One word for water." \
  "Repeat: hello" \
  "Count to three."

# B) long assistant bait — ask for paragraphs, then simple follow-ups
run_scenario "B_long_then_short" \
  "Write a long paragraph about residual streams in transformers, at least eight sentences." \
  "Now say just: ok" \
  "What is 3+5?" \
  "Repeat exactly: the quick brown fox jumps over the lazy dog"

# C) short then long then short
run_scenario "C_short_long_short" \
  "Hi" \
  "Explain the physics of friendship in two detailed paragraphs with examples." \
  "Reply with one word: ready" \
  "What is 17 times 23? Show arithmetic."

# Analyze probe JSONL
python3 - "$PROBE" "$SUM" <<'PY'
import json, sys, re
from pathlib import Path
probe_path, sum_path = Path(sys.argv[1]), Path(sys.argv[2])
lines = [json.loads(l) for l in probe_path.read_text().splitlines() if l.strip().startswith("{")]

def is_garbage_token(t: str) -> bool:
    if not t:
        return False
    # loops / control markers / pure punct soup
    if t in ("(c)", "(C)", "(D)", "**", "//", "…"):
        return True
    if re.fullmatch(r"[\W_]+", t) and len(t) > 2:
        return True
    # CJK heavy single token
    if any("\u4e00" <= c <= "\u9fff" for c in t):
        return True
    return False

def garbage_score(text: str) -> float:
    if not text:
        return 1.0
    bad = 0
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"-()[]"):
            continue
        bad += 1
    return bad / max(len(text), 1)

# group by scenario then turn
scenario = "unknown"
rows = []
for e in lines:
    if e.get("event") == "scenario":
        scenario = e.get("name", "unknown")
        continue
    if e.get("event") != "tok":
        continue
    e = dict(e)
    e["scenario"] = scenario
    rows.append(e)

# detect first garbage-ish token per scenario+turn via residual spike / entropy jump
report = []
report.append(f"# Collapse probe summary\n")
report.append(f"probe: `{probe_path}`\n")
report.append(f"total tok events: {len(rows)}\n")

from collections import defaultdict
by = defaultdict(list)
for r in rows:
    by[(r["scenario"], r["turn"])].append(r)

report.append("\n## Per-turn first anomaly (entropy jump or residual spike vs turn start)\n")
report.append("| scenario | turn | prev_asst_len | first_anomaly_step | reason | token | entropy | residual_norm | margin |\n")
report.append("|---|---:|---:|---:|---|---|---:|---:|---:|\n")

for key in sorted(by.keys()):
    scen, turn = key
    toks = sorted(by[key], key=lambda x: x["step"])
    if not toks:
        continue
    e0 = toks[0]["entropy"]
    r0 = toks[0]["residual_norm"]
    prev_len = toks[0].get("prev_asst_len", 0)
    first = None
    reason = ""
    for t in toks:
        reasons = []
        if t["entropy"] > e0 + 1.5 and t["step"] > 2:
            reasons.append("entropy+1.5")
        if r0 > 0 and t["residual_norm"] > r0 * 2.5 and t["step"] > 2:
            reasons.append("resid*2.5")
        if t["margin"] < 0.05 and t["step"] > 2 and t["entropy"] > 2.0:
            reasons.append("flat_margin")
        if is_garbage_token(t.get("token", "")):
            reasons.append("garbage_tok")
        if reasons:
            first = t
            reason = ",".join(reasons)
            break
    if first:
        report.append(
            f"| {scen} | {turn} | {prev_len} | {first['step']} | {reason} | {first.get('token','').replace('|','¦')!r} | {first['entropy']:.3f} | {first['residual_norm']:.1f} | {first['margin']:.3f} |\n"
        )
    else:
        report.append(f"| {scen} | {turn} | {prev_len} | — | clean |  |  |  |  |\n")

# correlate with prev_asst_len
report.append("\n## Collapse likelihood by prev assistant length bucket\n")
buckets = {"0 (first turn)": [], "1-40 short": [], "41-200 med": [], "201+ long": []}
for key, toks in by.items():
    prev = toks[0].get("prev_asst_len", 0)
    if prev == 0:
        b = "0 (first turn)"
    elif prev <= 40:
        b = "1-40 short"
    elif prev <= 200:
        b = "41-200 med"
    else:
        b = "201+ long"
    # collapsed if any garbage_tok or entropy spike late
    collapsed = False
    e0 = toks[0]["entropy"]
    for t in toks:
        if is_garbage_token(t.get("token", "")) or (t["step"] > 5 and t["entropy"] > e0 + 1.5):
            collapsed = True
            break
    buckets[b].append(collapsed)

for b, flags in buckets.items():
    if not flags:
        report.append(f"- **{b}**: n=0\n")
    else:
        rate = sum(flags) / len(flags)
        report.append(f"- **{b}**: n={len(flags)} collapse_rate={rate:.0%}\n")

# trend: does entropy rise before residual?
report.append("\n## Leading indicator (mean Δ over first 8 steps of turns that later collapse)\n")
for key, toks in by.items():
    if len(toks) < 10:
        continue
    e0 = toks[0]["entropy"]
    late_collapse = any(
        is_garbage_token(t.get("token", "")) or t["entropy"] > e0 + 1.5
        for t in toks[8:]
    )
    if not late_collapse:
        continue
    early = toks[:8]
    de = early[-1]["entropy"] - early[0]["entropy"]
    dr = early[-1]["residual_norm"] - early[0]["residual_norm"]
    report.append(
        f"- {key[0]} turn {key[1]}: early Δentropy={de:+.3f} early Δresidual={dr:+.1f} "
        f"(pre-garbage window steps 0-7)\n"
    )

text = "".join(report)
sum_path.write_text(text)
print(text)
PY

echo "WROTE $PROBE"
echo "WROTE $TRANS"
echo "WROTE $SUM"
