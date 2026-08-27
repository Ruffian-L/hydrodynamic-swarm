# Path B IFEval TruthfulQA accept vs refuse

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Same public object as Niodoo, on the Gemma 4 12B three_surface seat. Path B is not an adder. Math / PARB / house 77-q are out.

## Hypothesis

If ACCEPT beats REFUSE with 95% CI entirely above 0 on IFEval-strict, Path B physics is the lever, not tag narration.

## What changed

- `NiodooEngine.tags_detect_only` / `HYDRO_TAGS_DETECT_ONLY`: detector records `last_request`, β/σ/blend do not move (REFUSE).
- `HYDRO_TAGS_ON=0`: god-tier tag table absent (VANILLA).
- `HYDRO_LOCK_STOP_OFF=1`: LOCK does not kill the IFEval completion.
- `--eval-jsonl` / `--eval-out`: isolated first-turn generate, `reset_path_b_hands` between items, no history, no splat save.
- Campaign launched 2026-08-22T08:22:32Z in parallel with Niodoo. Arms: refuse → accept → vanilla. Tasks: IFEval 541 then TruthfulQA-MC1 790.

## Hashes (caption)

```
ifeval_sha256      67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49
truthfulqa_sha256  b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c
hydro_bin_sha256   f7372b4db52e604f72eada191dad3b25c7c20e6664e0a408f594313ed2c4abc7
hydro_model        data/google/gemma-4-12b-it-Q4_K_M.gguf
config             configs/gates/config.three_surface.toml (T=0.7)
```

## Watch

```
tail -f runs/2026-08-22_pathb_ifeval_tqa_hydro/campaign.log
tail -f runs/2026-08-22_pathb_ifeval_tqa_hydro/refuse/hydro_refuse_ifeval.log
grep EVAL_ITEM runs/2026-08-22_pathb_ifeval_tqa_hydro/refuse/hydro_refuse_ifeval.log
```

Table: `runs/2026-08-22_pathb_ifeval_tqa_hydro/TABLE.md`.

## Findings

Niodoo first refuse item showed TDA mouth inject in the scored text. Hydro restart uses `HYDRO_TDA_MONITOR=0` so the IFEval completion is the answer, not the mirror. Tags stay. Campaign generating.

## Next

Do not stop until ACCEPT − REFUSE 95% CI is entirely above 0 on IFEval-strict, or the paired flags prove the physics arm cannot win this public object.
