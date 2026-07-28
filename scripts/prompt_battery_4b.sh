#!/usr/bin/env bash
# Multi-prompt battery at 65 tok (B4d-q default) — option 4
set -euo pipefail
source "${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/scripts/cuda_env.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODEL="data/google/gemma-3-4b-it-Q4_K_M.gguf"
TOKENIZER="data/google/tokenizer.json"
TOKENS=65
BIN="$ROOT/target/release/hydrodynamic-swarm"
OUT="logs/prompt_battery_4b"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
echo -e "id\tlate_Fs\tlate_Fa\tmax_Fs\tchars\tsnip" > "$SUMMARY"

[[ -x "$BIN" ]] || cargo build --release --bin hydrodynamic-swarm

PROMPTS=(
  "friend|Explain the Physics of Friendship in one short paragraph."
  "noir|Explain the Physics of Friendship in a short 1980s noir detective monologue."
  "tech|In one short technical paragraph, describe residual physics steering for LLMs without jargon soup."
  "creative|Write one short lyrical paragraph about friendship as gravity and light."
)

for entry in "${PROMPTS[@]}"; do
  id="${entry%%|*}"
  prompt="${entry#*|}"
  echo "======== $id ========"
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" --prompt "$prompt" --tokens "$TOKENS" --clear-memory \
    > "$OUT/${id}.stdout" 2>&1 || true
  L=$(ls -t logs/*.jsonl | head -1)
  cp -f "$L" "$OUT/${id}.jsonl"
  python3 - "$id" "$OUT/${id}.jsonl" "$SUMMARY" <<'PY'
import json,sys
name,path,summary=sys.argv[1:4]
steps=[]; full=""
for line in open(path):
    o=json.loads(line)
    if o.get("entry_type")=="step": steps.append(o["step"])
    if o.get("entry_type")=="summary": full=o.get("summary",{}).get("decoded_output","")
def m(ss,k):
    return sum(float(s.get(k) or 0) for s in ss)/len(ss) if ss else 0
late=steps[-15:] if steps else []
snip=full[:90].replace("\t"," ").replace("\n"," ")
row=f"{name}\t{m(late,'splat_force_mag'):.2f}\t{m(late,'goal_force_mag'):.1f}\t{max((s['splat_force_mag'] for s in steps), default=0):.1f}\t{len(full)}\t{snip}"
print(row); open(summary,"a").write(row+"\n")
print("  FULL:", full[:220])
PY
done
echo "======== BATTERY DONE ========"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
