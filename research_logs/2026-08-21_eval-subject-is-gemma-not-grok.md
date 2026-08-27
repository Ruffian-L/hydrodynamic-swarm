# Eval subject is Gemma not Grok

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

The model under test (Gemma) is the collaborator who self-rates. Brief is the first chat turn; after the task Gemma scores; then reveal what we tested; then Gemma regrades. Grok does not self-rate. hydro_eval.sh run climb-after-fail is still the one command.

## Hypothesis

Telling Gemma it is an eval, then having it rate, then revealing the test, then regrade, is the collaborator loop. Grok rating itself was the wrong subject.

## What changed

- See changelog for the short why. Expand this subject here.

## Findings

Jason was right: the subject is Gemma. Brief / task / self-rate / reveal / regrade are user turns in `evals/<name>/`. `hydro_eval.sh` concatenates those files. Grok quotes `hydro_eval.sh rate` and `regrade` from the log.

## Next

`./scripts/hydro_eval.sh run climb-after-fail`

---

Signed: Grok (xAI) · operator Jason

