#!/usr/bin/env bash
set -uo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Jules Session Watcher
#
# Polls a Jules session until it completes, then:
#   1. Sends a desktop notification (notify-send)
#   2. Prints a terminal bell (^G) so your terminal tab/title flashes
#   3. Pulls the result and saves it to logs/reviews/
#   4. If Jules found issues, drops the quality-gate-failed marker
#
# Usage: jules-watcher.sh <session-id> <commit-sha>
# Typically launched by the post-commit hook in the background.
# ═══════════════════════════════════════════════════════════════════════════════

SESSION_ID="${1:?Usage: jules-watcher.sh <session-id> <commit-sha>}"
COMMIT_SHA="${2:?Usage: jules-watcher.sh <session-id> <commit-sha>}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(cd "$(dirname "$0")/.." && pwd)")"
REVIEW_LOG_DIR="$REPO_ROOT/logs/reviews"
GATE_FAIL_MARKER="$REPO_ROOT/.quality-gate-failed"
RESULT_FILE="$REVIEW_LOG_DIR/jules_result_${SESSION_ID}_${COMMIT_SHA}.txt"

POLL_INTERVAL=30   # seconds between polls
MAX_WAIT=1800       # 30 minutes max wait
ELAPSED=0

mkdir -p "$REVIEW_LOG_DIR"

echo "[jules-watcher] Watching session $SESSION_ID for commit $COMMIT_SHA"
echo "[jules-watcher] Polling every ${POLL_INTERVAL}s (max ${MAX_WAIT}s)"

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))

    # Check session status
    STATUS_OUTPUT="$(jules remote list --session 2>&1)"

    # Look for our session — check if it's completed
    if echo "$STATUS_OUTPUT" | grep -q "$SESSION_ID"; then
        # Check for completion signals (completed/done/finished/failed)
        if echo "$STATUS_OUTPUT" | grep "$SESSION_ID" | grep -qiE "(completed|done|finished|failed|ready)"; then
            echo "[jules-watcher] Session $SESSION_ID finished after ${ELAPSED}s"

            # Pull the result
            PULL_OUTPUT="$(jules remote pull --session "$SESSION_ID" 2>&1)"
            echo "$PULL_OUTPUT" > "$RESULT_FILE"

            # Determine if issues were found
            HAS_ISSUES=false
            if echo "$PULL_OUTPUT" | grep -qiE "(bug|error|issue|fix|regression|vulnerability)"; then
                HAS_ISSUES=true
            fi

            # ── Desktop notification ──────────────────────────────────────
            if command -v notify-send &>/dev/null; then
                if [[ "$HAS_ISSUES" == "true" ]]; then
                    notify-send -u critical "🤖 Jules Review Complete" \
                        "Session $SESSION_ID for commit $COMMIT_SHA found issues!\nSee: $RESULT_FILE" 2>/dev/null || true
                else
                    notify-send -u normal "🤖 Jules Review Complete" \
                        "Session $SESSION_ID for commit $COMMIT_SHA — looks clean.\nSee: $RESULT_FILE" 2>/dev/null || true
                fi
            fi

            # ── Terminal bell (makes tab flash in most terminals) ──────────
            printf '\a'

            # ── Print to any active terminal ──────────────────────────────
            # Write to all user's terminals so they see it
            for tty in /dev/pts/*; do
                if [[ -w "$tty" ]] && [[ "$tty" != "/dev/pts/ptmx" ]]; then
                    {
                        echo ""
                        echo "╔══════════════════════════════════════════════════════════════╗"
                        if [[ "$HAS_ISSUES" == "true" ]]; then
                            echo "║  🤖 JULES REVIEW COMPLETE — ISSUES FOUND                   ║"
                        else
                            echo "║  🤖 JULES REVIEW COMPLETE — CLEAN                           ║"
                        fi
                        echo "║  Session: $SESSION_ID  Commit: $COMMIT_SHA                    ║"
                        echo "║  Result:  $RESULT_FILE"
                        echo "╚══════════════════════════════════════════════════════════════╝"
                        echo ""
                    } > "$tty" 2>/dev/null || true
                fi
            done

            # ── Update gate marker if issues found ────────────────────────
            if [[ "$HAS_ISSUES" == "true" ]]; then
                if [[ -f "$GATE_FAIL_MARKER" ]]; then
                    echo ",jules" >> "$GATE_FAIL_MARKER"
                else
                    echo "$COMMIT_SHA:jules" > "$GATE_FAIL_MARKER"
                fi
            fi

            exit 0
        fi
    fi

    echo "[jules-watcher] Session $SESSION_ID still running (${ELAPSED}s elapsed)..."
done

echo "[jules-watcher] Timed out after ${MAX_WAIT}s — check manually: jules remote list --session"

# Timeout notification
if command -v notify-send &>/dev/null; then
    notify-send -u low "🤖 Jules Watcher Timeout" \
        "Session $SESSION_ID timed out after ${MAX_WAIT}s.\nCheck: jules remote list --session" 2>/dev/null || true
fi
