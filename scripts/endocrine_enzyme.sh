#!/usr/bin/env bash
# =============================================================================
# Endocrine text enzyme — second mouth (stateless)
# =============================================================================
# Geometry stays on the *speaking* model in hydrodynamic-swarm.
# This process only answers ENDOCRINE_URL chat/completions (cold, short).
#
# Usage:
#   ./scripts/endocrine_enzyme.sh          # start on :8210
#   ./scripts/endocrine_enzyme.sh stop
#   ./scripts/endocrine_enzyme.sh status
#
# Then in the swarm shell:
#   export ENDOCRINE_URL=http://127.0.0.1:8210/v1
#   export ENDOCRINE_MODEL=local
#   ./run_swarm.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${ENDOCRINE_PORT:-8210}"
HOST="${ENDOCRINE_HOST:-127.0.0.1}"
PIDFILE="${ENDOCRINE_PIDFILE:-$ROOT/logs/endocrine_enzyme.pid}"
LOGFILE="${ENDOCRINE_LOG:-$ROOT/logs/endocrine_enzyme.log}"

# Prefer a small spare model so the speaker (4B hydro) keeps the GPU headroom.
# Override: ENZYME_MODEL=/path/to.gguf
pick_model() {
  if [[ -n "${ENZYME_MODEL:-}" && -f "$ENZYME_MODEL" ]]; then
    echo "$ENZYME_MODEL"
    return
  fi
  local c
  for c in \
    /media/ruffianl/ghost_team/models/functiongemma-270m-it-BF16.gguf \
    /media/ruffianl/ghost_team/models/google_functiongemma-270m-it-bf16.gguf \
    /media/ruffianl/ghost_team/models/Qwen3.5-0.8B-BF16.gguf \
    "$HOME/models/Qwen3.5-0.8B-BF16.gguf" \
    "$ROOT/data/google/gemma-3-4b-it-Q4_K_M.gguf"
  do
    if [[ -f "$c" ]]; then
      echo "$c"
      return
    fi
  done
  echo "ERROR: no enzyme GGUF (set ENZYME_MODEL=…)" >&2
  exit 1
}

pick_llama() {
  if [[ -n "${LLAMA:-}" && -x "$LLAMA" ]]; then
    echo "$LLAMA"
    return
  fi
  local c
  for c in \
    "$HOME/.local/bin/llama-server" \
    "$HOME/llama.cpp/build/bin/llama-server" \
    /media/ruffianl/ghost_team/projects/llama-server
  do
    if [[ -x "$c" ]]; then
      echo "$c"
      return
    fi
  done
  echo "ERROR: llama-server not found (set LLAMA=…)" >&2
  exit 1
}

cmd="${1:-start}"

case "$cmd" in
  stop)
    if [[ -f "$PIDFILE" ]]; then
      pid="$(cat "$PIDFILE" || true)"
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" || true
        echo "[enzyme] stopped pid=$pid"
      fi
      rm -f "$PIDFILE"
    else
      echo "[enzyme] no pidfile"
    fi
    exit 0
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "[enzyme] UP pid=$(cat "$PIDFILE")  $HOST:$PORT"
      curl -sS "http://$HOST:$PORT/health" 2>/dev/null || curl -sS "http://$HOST:$PORT/v1/models" 2>/dev/null | head -c 200 || true
      echo
    else
      echo "[enzyme] DOWN"
    fi
    exit 0
    ;;
  start) ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac

mkdir -p "$ROOT/logs"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[enzyme] already running pid=$(cat "$PIDFILE")"
  exit 0
fi

LLAMA="$(pick_llama)"
MODEL="$(pick_model)"
echo "[enzyme] llama=$LLAMA"
echo "[enzyme] model=$MODEL"
echo "[enzyme] bind=$HOST:$PORT"
echo "[enzyme] log=$LOGFILE"
echo
echo "  export ENDOCRINE_URL=http://$HOST:$PORT/v1"
echo "  export ENDOCRINE_MODEL=local"
echo

# Small ctx / few slots — enzyme only, not team chat.
# --jinja + --reasoning off: Qwen3-class models otherwise fill reasoning_content
# and leave message.content empty (looks like "HTTP empty" in hydro).
nohup "$LLAMA" \
  --host "$HOST" \
  --port "$PORT" \
  -m "$MODEL" \
  -c 2048 \
  -n 128 \
  --parallel 1 \
  --jinja \
  --reasoning off \
  >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"
echo "[enzyme] started pid=$(cat "$PIDFILE") — wait a few seconds then: $0 status"
echo "[enzyme] jinja on, reasoning off (content must not be empty)"
