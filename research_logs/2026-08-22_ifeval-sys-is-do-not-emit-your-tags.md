# IFEval SYS is DO NOT emit your tags

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Killed Hydro 10 GOD_TIER emit-lecture. Eval tags arm packs tags_do_not_emit.txt via HYDRO_SYSTEM_PROMPT_FILE. Chat GOD_TIER unchanged. Rebuild then 10-item.

## Hypothesis

Gemma writing tasks will not be told to emit spike. Same house DNE line as Niodoo.

## What changed

Killed the Hydro 10 that packed GOD_TIER ("If you doubt the path, emit <spike>."). Chat GOD_TIER stays. Eval tags arm now reads HYDRO_SYSTEM_PROMPT_FILE = niodoo `prompts/tags_do_not_emit.txt` so writing tasks get **DO NOT emit your tags.** Vanilla still tags_on=0. Rebuild required. Launch after niodoo DNE 10 or as soon as the binary is up.

## Findings

GOD_TIER 10 killed (no TABLE). Rebuild `d7a9c86c…` has `HYDRO_SYSTEM_PROMPT_FILE`. Chat packing tests still forbid DNE on default GOD_TIER. Eval-only file override.

## Next

Hydro DNE 10 chained to start when Niodoo DNE-10 TABLE lands. No 541.
