# Nuke tag refusal always accept

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: no tag refusals. Hydro already had no accept/refuse gate. I had added a detect-only skip for a refuse arm. That skip is gone. Redo tags vs vanilla.

## Hypothesis

Always-apply Path B plus IFEval-strict tags vs vanilla is the public object. No refuse.

## What changed

- `apply_request_effects` / `fire_tag` always write physics. Detect-only no longer skips β/σ.
- `--eval-jsonl` still isolates items. `HYDRO_TAGS_ON=1` for tags, `0` for vanilla.
- Campaign: `runs/2026-08-22_pathb_ifeval_tags_hydro` arms `tags` then `vanilla`.

## Findings

(open — generating)

## Next

Do not stop until tags − vanilla 95% CI is entirely above 0 on IFEval-strict.
