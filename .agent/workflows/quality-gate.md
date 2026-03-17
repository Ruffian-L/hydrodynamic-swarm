---
description: how the quality gate git hooks work
---

# Quality Gate — Git Hooks

## What Runs After Every Commit

The `post-commit` hook runs these 6 checks in order:

| # | Tool | What it checks | Skip with |
|---|------|---------------|-----------|
| 1 | `cargo fmt --check` | Code formatting | `SKIP_FMT=1` |
| 2 | `cargo clippy -D warnings` | Lint warnings (treat as errors) | `SKIP_CLIPPY=1` |
| 3 | `cargo machete` | Unused dependencies | `SKIP_MACHETE=1` |
| 4 | `cargo audit` | Security vulnerabilities | `SKIP_AUDIT=1` |
| 5 | `coderabbit review` | AI code review (sync) | `SKIP_CODERABBIT=1` |
| 6 | `jules new` | AI review session (async) | `SKIP_JULES=1` |

## How It Blocks You

- If **any check fails** → drops `.quality-gate-failed` marker
- The `pre-push` hook checks for this marker and **blocks pushes**
- Fix the issue → commit again → gate re-runs → marker cleared if pass

## Jules Notifications

Jules runs async. A background watcher (`scripts/jules-watcher.sh`) polls the session and:
1. **Desktop notification** via `notify-send` when done
2. **Terminal bell** (`^G`) so your terminal tab flashes
3. **Prints to all open terminal PTYs** so you see the result wherever you are

## Skip Everything

```bash
# Skip all post-commit checks
SKIP_QUALITY_GATE=1 git commit -m "wip"

# Skip only push blocker
SKIP_QUALITY_GATE=1 git push

# Skip specific tools
SKIP_CLIPPY=1 SKIP_AUDIT=1 git commit -m "quick fix"

# Git's built-in skip (skips ALL hooks)
git commit --no-verify -m "emergency"
```

## Review Logs

All review outputs are saved to `logs/reviews/`:
- `coderabbit_<sha>_<time>.txt`
- `jules_<sha>_<time>.txt`
- `jules_result_<session>_<sha>.txt`
- `cargo-fmt_<sha>_<time>.txt`
- `cargo-clippy_<sha>_<time>.txt`
- etc.
