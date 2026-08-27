# Official 10 mouth is Niodoo pack layout

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: look at the Niodoo pack; hydro is ass; make it the same layout. 105737 ran the 14 turns as unlabeled `you> gemma4>` blobs. Niodoo `official10.txt` is `Prompt>` / `Prompt1>` … `Expected answer — EVALUATOR ONLY>` / `Model>`.

## Hypothesis

Printing the pack markers around hydro `--chat` (expected from `evals/official-10/expected.txt`, never sent to Gemma) makes `tail` / `follow_mouth.sh` match Niodoo.

## What changed

- `HYDRO_OFFICIAL_LAYOUT=1` + `HYDRO_EXPECTED_FILE` on `hydro_eval.sh run official-10`.
- Chat mouth: 72 `=`, `PromptN>`, user text, expected block, `Model>`, tokens, `[PromptN done  Xs]`.
- `scripts/follow_mouth.sh` polls (uutils `tail -f` dies on inotify).

## Next

Restart official-10 so Jason sees Prompt1> snail as the second block.

Signed: Grok (xAI) · operator Jason
