# Vanilla Llama and Gemma 4 on official 11-compact

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydro-3surface

## Context

Copied Niodoo april_angle_tags onto the 11-compact as a receipt. Did not rewalk official-10 080720. Vanilla Llama 3.1 and vanilla Gemma 4 on that compact ran in niodoo-live.

## Hypothesis

Same compact + Niodoo tag SYS on stock llama.cpp will not actuate Path B; exact scars can still return from user context; hydro 080720 remains the physics Gemma arm.

## What changed

- Copied Niodoo `april_angle_tags.txt` into `evals/official-10/niodoo_april_angle_tags.txt` (receipt only). The live hydro eval already used god-tier SYS; that stamp is paired and is not this run.
- Vanilla Llama 3.1 + vanilla Gemma 4 on the same 11-compact live in niodoo-live `runs/2026-08-21_vanilla_official11_llama31` and `..._gemma4`.

## Findings

Hydro official-10 `20260821_080720` is already paired. Do not rewalk it.

Vanilla comparison (niodoo-live, stock llama.cpp, SYS `april_angle_tags.txt`):

- Llama 3.1: `runs/2026-08-21_vanilla_official11_llama31` — P3/P6/P7 exact; P2 mint missed (fake Internal monitor); P5 narrated `<focus>` `<explore>` (no physics); P11 `[4,3,2,1,5]`; SCORE **8/10** inflated.
- Gemma 4: `runs/2026-08-21_vanilla_official11_gemma4` — after `--reasoning off`. P2/P3/P6/P7 exact; P11 used reverse/append and truncated; no SCORE (512-cap). Empty-content abort kept as `..._gemma4_empty_content`.
- Full table: niodoo-live `research_logs/2026-08-21_vanilla-official-11-compact-llama-3-1-and-gemma-4.md`

Copied SYS file on this tree: `evals/official-10/niodoo_april_angle_tags.txt`. The 080720 hydro run used god-tier SYS, not this file.

## Next

Do not rewalk 080720. Trail-own vs three-line and cycle-settle-before-SCORE stay the hydro cost.

Signed: Grok (xAI) · operator Jason

