# IFEval 10-item tags vs vanilla hydro launch

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Niodoo 10 is paid. Launch Hydro matching 10-item IFEval-strict tags vs vanilla on always-apply binary 62c018a8, limit 10, no refuse, do not resume the paused 541 jsonl.

## Hypothesis

Gemma GOD_TIER already fires FOCUS/LOCK. Ten items will show whether tags help or wreck Hydro mouths the same way before paying 541.

## What changed

Jason: 10 first then see, then keep going. Niodoo 10 is paid (tags 5/10 vs vanilla 3/10, CI includes 0). Hydro 541 tags2 is paused at 4/541 for the scaler and must not be resumed (eval-jsonl truncates). Launch a fresh Hydro 10-item tags vs vanilla on the always-apply disk binary `62c018a8…`, `--limit 10 --tasks ifeval`, no refuse, HYDRO_TAGS_DETECT_ONLY=0, HYDRO_LOCK_STOP_OFF=1, TDA monitor off.

Stamp: `runs/2026-08-22_pathb_ifeval_tags10_hydro/`

Mouth: `tail -f …/mouth.txt`
Campaign: `tail -f …/campaign.log`

## Findings

Launch in progress. Niodoo 10 already paid (5/10 vs 3/10, CI includes 0). Hydro 541 tags2 stays paused at 4 items; this stamp is a fresh `--limit 10`.

## Next

Read Hydro TABLE.md + mouths when 10+10 land. No 541.
