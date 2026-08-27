# Do control tags perturb token generation?

**Date:** 2026-08-19  
**Seat:** Gemma 4 12B IT Q4 · isolation residual off · god-tier system wrap  
**Harness:** `--tag-ablation` · same prefill · inject tag at step 0  
**Out:** `logs/tag_ablation.jsonl`

Prompt: `Count from 1 to 20 using digits and spaces only.` · 40 tokens.

None baseline (both seats): `1 2 3 10 11 12 13 14 15 16 17 18 19 20`

## Isolation (T=0, top_k=1, rep=1.0)

| Arm | Scan | Applied knobs | vs none |
|---|---|---|---|
| spike | Spike | T 0→0.05, rep 1.0 | IDENTICAL |
| explore | Explore | T 0→0.05, rep 1.0 | IDENTICAL |
| focus | Focus | T 0→0.05, **rep 1.15** | **CHANGED** @ char 7 → `1 2 3 14 5 6…` |
| reset | Reset | T 0, rep 1.0 | IDENTICAL |
| remember | **empty** | none | IDENTICAL |
| lock | **Focus** | T 0→0.05, **rep 1.15** | **CHANGED** — same string as focus |

`top_k=1` keeps softmax on a single token, so T cannot move greedy. Only **rep>1** can.

## Open (T=0.8, top_k=40, seed=1)

| Arm | Applied knobs | vs none |
|---|---|---|
| spike | T 0.8→1.08, rep 1.0 | IDENTICAL |
| explore | T 0.8→0.96, rep 1.0 | IDENTICAL |
| focus | T 0.8→0.56, **rep 1.15** | **CHANGED** @ char 6 → `1 2 3 4 5 6 4 8…` |
| reset | T 0.8, rep 1.0 | IDENTICAL |
| remember | no hook | IDENTICAL |
| lock | **Focus** again | **CHANGED** — same as focus |

Temperature multipliers are wired (logged) but did **not** change this sequence. Rep-penalty 1.15 did.

## Verdict

- **Focus** perturbs generation (via rep_penalty 1.15). Proven on both seats.
- **Spike / explore** change T only; no token delta on this prompt.
- **Reset** is identity, as designed.
- **Remember** is dead (scan empty).
- **Lock is a bug:** `normalize_name` matches `LOCK` inside `<lock>` → applied as **Focus**, not a commit/stop.

Tags do not move residual. Sample knobs only.
