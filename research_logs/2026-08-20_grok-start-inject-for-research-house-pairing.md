# Grok start inject for research-house pairing

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Hydro AGENTS.md is gitignored so Grok never injected it. Put the research-house + changelog pairing law in ~/.grok/AGENTS.md and ~/.grok/rules/ so it loads at session start. We think Jason will not have to remind pairing again.

## Hypothesis

Home AGENTS.md + rules load at begin; next session reads CHANGELOG first and pair_log after mutations without a reminder.

## What changed

Hydro `AGENTS.md` is gitignored, so Grok discovery never injected it — Jason had to keep reminding. Start-of-session inject now lives where Grok actually loads files:

- `~/.grok/AGENTS.md` (global, conversation start)
- `~/.grok/rules/01-research-house.md`
- `~/.grok/rules/02-hydro-smoke.md` (do not rewalk 070050 / 070557 / 073954)

Tree `AGENTS.md` now points at those paths. Pairing law is unchanged: read CHANGELOG first; `pair_log.sh` after every mutation; smoke is `smoke_convo.sh`.

We think the next session will pair without being asked.

Signed: Grok (xAI) · operator Jason

## Findings

`~/.grok/AGENTS.md` created. `~/.grok/rules/01-research-house.md` strengthened. `~/.grok/rules/02-hydro-smoke.md` names the three paired 9-turns as done. Tree AGENTS.md documents the gitignore skip. New Grok sessions from home load these without Jason pasting the contract.

## Next

Jason: new Grok session (or restart) so the inject is in context. No hydro smoke to re-run.

---

Signed: Grok (xAI) · operator Jason
