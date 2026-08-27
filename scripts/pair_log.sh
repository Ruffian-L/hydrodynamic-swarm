#!/usr/bin/env bash
# Pair a changelog why with a subject-sliced research log.
# Usage:
#   scripts/pair_log.sh --why "short why" --title "Subject title" [--body path] [--hyp "we think X"]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHY=""
TITLE=""
BODY=""
HYP=""
AGENT="${RESEARCH_AGENT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --why) WHY="${2:-}"; shift 2 ;;
    --title) TITLE="${2:-}"; shift 2 ;;
    --body) BODY="${2:-}"; shift 2 ;;
    --hyp|--hypothesis) HYP="${2:-}"; shift 2 ;;
    --agent) AGENT="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,5p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WHY" || -z "$TITLE" ]]; then
  echo "need --why and --title" >&2
  exit 2
fi

if [[ -d "$ROOT/research-log" && ! -d "$ROOT/research_logs" ]]; then
  LOGDIR="$ROOT/research-log"
  LOGREL="research-log"
else
  LOGDIR="$ROOT/research_logs"
  LOGREL="research_logs"
fi
mkdir -p "$LOGDIR" "$ROOT/scripts"

DATE="$(date -u +%Y-%m-%d)"
slug="$(printf '%s' "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g' | cut -c1-60)"
[[ -n "$slug" ]] || slug="entry"
LOGFILE="$LOGDIR/${DATE}_${slug}.md"
n=2
while [[ -e "$LOGFILE" ]]; do
  LOGFILE="$LOGDIR/${DATE}_${slug}-$n.md"
  n=$((n + 1))
done

CHANGELOG="$ROOT/CHANGELOG.md"
if [[ ! -f "$CHANGELOG" ]]; then
  cat > "$CHANGELOG" <<'EOF'
# Changelog

Research repo. Not production unless Jason says so.
Short whys here. Longer subject logs in the research log folder.
Agent contract: `AGENTS.md` (tracked).

EOF
fi

{
  echo
  echo "## ${DATE} — ${TITLE}"
  echo
  echo "- ${WHY}"
  if [[ -n "$HYP" ]]; then
    echo "- Hypothesis: ${HYP}"
  fi
  echo "- Research: \`${LOGREL}/$(basename "$LOGFILE")\`"
  if [[ -n "$AGENT" ]]; then
    echo "- Agent: ${AGENT}"
  fi
} >> "$CHANGELOG"

{
  echo "# ${TITLE}"
  echo
  echo "> Date: ${DATE}"
  if [[ -n "$AGENT" ]]; then
    echo "> Agent: ${AGENT}"
  fi
  echo "> Repo: $(basename "$ROOT")"
  echo
  echo "## Context"
  echo
  echo "$WHY"
  echo
  echo "## Hypothesis"
  echo
  if [[ -n "$HYP" ]]; then
    echo "$HYP"
  else
    echo "(not stated)"
  fi
  echo
  echo "## What changed"
  echo
  if [[ -n "$BODY" && -f "$BODY" ]]; then
    cat "$BODY"
  else
    echo "- See changelog for the short why. Expand this subject here."
  fi
  echo
  echo "## Findings"
  echo
  echo "(open)"
  echo
  echo "## Next"
  echo
  echo "(open)"
} > "$LOGFILE"

echo "changelog: $CHANGELOG"
echo "research:  $LOGFILE"
