# Named hydro eval collaborator rate-reveal-regrade

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Named hydro evals: hydro_eval.sh run <name>, collaborator self-rate then reveal then regrade. smoke_convo prints prompts= and warns if PROMPTS_FILE is not a real file. Stops silent default-9turn when process substitution is used.

## Hypothesis

A named file on disk plus a self-rate/reveal loop will survive compaction and stop Jason getting a reconstructed smoke recipe that silently becomes Hello there friend.

## What changed

Named hydro evals live in `evals/<name>/` (prompts.txt is a real file). Runner `scripts/hydro_eval.sh run <name>` prints a collaborator opening, runs `smoke_convo.sh` with that file, then self-rate → reveal sealed `score.md` → regrade. `smoke_convo.sh` now records `prompts=` in the header and refuses silent process-substitution fallback without a warning. Skill `hydro-eval` + rule `04-hydro-eval.md`: never hand Jason `PROMPTS_FILE=<(...)`. Compaction uses the eval **name**, not reconstructed prompts.

Seeded: `climb-after-fail`, `isolation-9turn`.

## Findings

`./scripts/hydro_eval.sh list` shows `climb-after-fail` and `isolation-9turn`. Process substitution is why the 20260821_023317 run printed `Hello there, friend.` — default 9-turn, not lumina. Header will now say `prompts=DEFAULT-9TURN` or the real path. Skill + rule load at Grok start.

## Next

Jason runs `./scripts/hydro_eval.sh run climb-after-fail`. Agents self-rate then reveal, never invent a substitute command.

---

Signed: Grok (xAI) · operator Jason

