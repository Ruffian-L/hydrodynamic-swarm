# Un-ignore AGENTS.md

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: ungit-ignore all the AGENTS.md files.

Yesterday's research-house stamp gitignored `AGENTS.md` on purpose so GitHub
would not get the agent contract. Grok skips gitignored project files, so the
law never loaded. Same mistake as nuking the files, quieter this time.

## What changed

- Removed `AGENTS.md`, `AGENTS*.md`, `AGENTS(jules).md`, `AGENT.md` from
  `.gitignore` in this tree.
- Left `CLAUDE.md` and local agent dirs (`.agent/`, `.jules/`, `.grok/`) ignored.
- Rewrote the contract so it no longer says "do not commit this file."
- Staged the AGENTS files. Did not commit. Jason taps the key.

## Hypothesis

We think repo `AGENTS.md` will now auto-load for Grok and other agents.

## Findings

- gitignore
- AGENTS.md
- CHANGELOG.md:appended
- scripts copied
- staged AGENTS.md, AGENTS(jules).md, .gitignore, CHANGELOG.md, scripts/pair_log.sh, scripts/research_house_check.sh
- tracked

## Next

Jason commits when he wants the contract on GitHub. Measure whether the next
session actually reads the repo file.
