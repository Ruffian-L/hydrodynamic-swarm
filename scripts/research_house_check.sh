#!/usr/bin/env bash
# Confirm research-house structure in this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
say() { printf '%s\n' "$1"; }

need_file() {
  if [[ -f "$1" ]]; then
    say "ok   $2"
  else
    say "MISS $2"
    fail=1
  fi
}

need_file "$ROOT/AGENTS.md" "AGENTS.md (house contract)"
need_file "$ROOT/CHANGELOG.md" "CHANGELOG.md"
need_file "$ROOT/scripts/pair_log.sh" "scripts/pair_log.sh"

if [[ -d "$ROOT/research_logs" ]]; then
  say "ok   research_logs/"
elif [[ -d "$ROOT/research-log" ]]; then
  say "ok   research-log/"
else
  say "MISS research_logs/ or research-log/"
  fail=1
fi

if [[ -d "$ROOT/.git" || -f "$ROOT/.git" ]]; then
  if git -C "$ROOT" check-ignore -q AGENTS.md 2>/dev/null; then
    say "MISS AGENTS.md is gitignored — drop it from .gitignore"
    fail=1
  elif git -C "$ROOT" ls-files --error-unmatch AGENTS.md >/dev/null 2>&1; then
    say "ok   AGENTS.md is tracked"
  else
    say "MISS AGENTS.md not ignored but untracked — git add AGENTS.md"
    fail=1
  fi
else
  say "note no git in this tree"
fi

exit "$fail"
