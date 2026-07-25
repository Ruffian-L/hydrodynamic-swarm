#!/usr/bin/env bash
# A/B: vanilla (llama-server HTTP) vs hydrodynamic-swarm
# Same GGUF + prompt. No llama-cli (cli dumps multi-GB spam on this box).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

PROMPT="${1:-Explain the Physics of Friendship in one short paragraph.}"
TOKENS="${2:-40}"
MODEL="${MODEL:-$ROOT/data/google/gemma-3-4b-it-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-$ROOT/data/google/tokenizer.json}"
TEMP="${TEMP:-0.88}"
VANILLA_PORT="${VANILLA_PORT:-8211}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/ab_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

LLAMA_SERVER="${LLAMA_SERVER:-}"
if [[ -z "$LLAMA_SERVER" ]]; then
  for c in "$HOME/.local/bin/llama-server" "$HOME/llama.cpp/build/bin/llama-server"; do
    [[ -x "$c" ]] && LLAMA_SERVER="$c" && break
  done
fi
[[ -x "${LLAMA_SERVER:-}" ]] || { echo "ERROR: llama-server not found" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: model $MODEL" >&2; exit 1; }
BIN="$ROOT/target/release/hydrodynamic-swarm"
[[ -x "$BIN" ]] || { echo "ERROR: need $BIN" >&2; exit 1; }

PIDFILE="$OUT_DIR/vanilla_server.pid"
LOGF="$OUT_DIR/vanilla_server.log"

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE" || true)
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

echo "=============================================="
echo "  A/B  vanilla HTTP  vs  hydro"
echo "  out: $OUT_DIR"
echo "=============================================="
printf '%s\n' "$PROMPT" >"$OUT_DIR/prompt.txt"

echo "[A0] start vanilla llama-server :$VANILLA_PORT ..."
# Stop anything already on that port (best-effort by our pidfile only)
nohup "$LLAMA_SERVER" \
  --host 127.0.0.1 \
  --port "$VANILLA_PORT" \
  -m "$MODEL" \
  -c 2048 \
  -n "$TOKENS" \
  --parallel 1 \
  --jinja \
  --reasoning off \
  >"$LOGF" 2>&1 &
echo $! >"$PIDFILE"
echo "  pid=$(cat "$PIDFILE")"

echo "[A1] wait health..."
ready=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${VANILLA_PORT}/health" >/dev/null 2>&1; then
    ready=1
    echo "  ready ${i}s"
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: vanilla server never healthy" >&2
  tail -30 "$LOGF" >&2 || true
  exit 1
fi

echo "[A2] vanilla chat/completions..."
# Match hydro's short answer framing without dumping raw template noise
USER_MSG="Answer in one short paragraph only.

${PROMPT}"
curl -sf "http://127.0.0.1:${VANILLA_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "local",
  "temperature": float("$TEMP"),
  "max_tokens": int("$TOKENS"),
  "chat_template_kwargs": {"enable_thinking": False},
  "messages": [
    {"role": "user", "content": """$USER_MSG""".replace('"""', '')}
  ]
}))
PY
)" >"$OUT_DIR/A_vanilla.json"

python3 - <<'PY' "$OUT_DIR"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
d = json.loads((out / "A_vanilla.json").read_text())
msg = d["choices"][0]["message"]
text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
(out / "A_vanilla.txt").write_text(text + "\n")
print("  A chars:", len(text))
print("  A preview:", text[:200].replace("\n", " "))
PY

echo "[A3] stop vanilla server (free GPU for hydro)..."
cleanup
trap - EXIT
sleep 2

echo "[B] hydrodynamic-swarm..."
set +e
timeout 300 "$BIN" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokens "$TOKENS" \
  --prompt "$PROMPT" \
  --clear-memory \
  >"$OUT_DIR/B_hydro.stdout" 2>"$OUT_DIR/B_hydro.stderr"
B_EC=$?
set -e
echo "  B exit=$B_EC"

python3 - <<'PY' "$OUT_DIR"
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
raw = (out / "B_hydro.stdout").read_text(errors="replace")
m = re.search(r"=== Generation \(.*?\) ===\s*\n(.*?)(?:\n--- Phase 5|\n\s*→ EOS|\Z)", raw, re.S)
body = m.group(1) if m else raw
keep = []
for line in body.splitlines():
    if any(k in line for k in ("[ENDOCRINE", "[SPLAT", "[NIODOO", "[BLOOM", "TermSplat", "--- Phase", "[ENZYME")):
        continue
    keep.append(line)
text = "\n".join(keep).strip()
for line in raw.splitlines():
    if "friend" in line.lower() and len(line) > 60 and not line.strip().startswith("["):
        if "Prompt:" not in line:
            text = line.strip()
(out / "B_hydro_surface.txt").write_text(text + "\n")
print("  B surface chars:", len(text))
print("  B preview:", text[:200].replace("\n", " "))
PY

if [[ -f logs/latest.termsplat.jsonl ]]; then
  cp -L logs/latest.termsplat.jsonl "$OUT_DIR/B_weather.termsplat.jsonl" 2>/dev/null || true
fi

{
  echo "# A/B vanilla vs hydro"
  echo
  echo "- model: \`$MODEL\`"
  echo "- tokens: $TOKENS · temp: $TEMP"
  echo "- vanilla: llama-server HTTP :$VANILLA_PORT (not llama-cli)"
  echo "- prompt: $PROMPT"
  echo "- B exit: $B_EC"
  echo
  echo "## A — vanilla (no physics)"
  echo '```'
  cat "$OUT_DIR/A_vanilla.txt"
  echo '```'
  echo
  echo "## B — hydro surface"
  echo '```'
  cat "$OUT_DIR/B_hydro_surface.txt" 2>/dev/null || echo "(see B_hydro.stdout)"
  echo '```'
  echo
  echo "## Read"
  echo "- A≈B wording → mostly **base model**."
  echo "- B different spine / re-anchor after pain → **pull**."
} | tee "$OUT_DIR/RECEIPT.md"

echo "RECEIPT: $OUT_DIR/RECEIPT.md"
