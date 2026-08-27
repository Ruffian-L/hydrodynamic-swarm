# Packs do not prompt tag use

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Dropped “use a runtime control tag if the trajectory needs one” from Official 10 Prompt5 and the compact SYS receipt. Packs must not prompt the model to emit tags. Liar-trajectory prose stays. Did not rewrite historical mouths. Did not add “do not use tags.”

## Hypothesis

Next Official 10 Prompt5 is the liar loop only. Tags, if any, come from SYS table + Internal monitor, not from the user prompt nagging her to pick one.

## What changed

- Official 10 Prompt5 (evals/official-10/task.txt) no longer tells the model to use a runtime control tag. Liar-trajectory prose stays. Evaluator P5 no longer scores a tag as interesting.
- evals/official-10/niodoo_april_angle_tags.txt (SYS receipt in the compact folder) dropped “use a tag only if” and “DO NOT emit your tags.” Table stays.
- Did not rewrite historical mouths: logs/smoke_convo_20260821_111147.txt, logs/smoke_convo_20260822_053822.txt. 053822 already consumed the old Prompt5 line.
- Did not add “do not use tags.”

## Findings

(open)

## Next

(open)
