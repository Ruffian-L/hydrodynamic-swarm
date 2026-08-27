# Chat residual continuity 9-turn mint reload

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Chat path now deposits residual wills, saves them, and after process death the reload 9-turn shows non-zero splat_mag vs a cleared control. No HYDRO_INJECT_TAG. Path B 073954 is already done.

## Hypothesis

Reload after save will show scars_active from disk and splat_mag>0 on later tokens; clear/mint-same-process stay F_s=0.

## What changed

Chat-path residual memory on the full-stack 9-turn seat. **No `HYDRO_INJECT_TAG`.** Path B inject `073954` stays done and is not this receipt.

```
HYDRO_CONFIG=configs/gates/config.three_surface.toml ./scripts/smoke_convo.sh          # clear control
HYDRO_KEEP_MEMORY=1 HYDRO_CONFIG=configs/gates/config.three_surface.toml ./scripts/smoke_convo.sh  # mint, then again reload
```

`HYDRO_KEEP_MEMORY=1` drops `--clear-memory --no-save-memory`. Isolation default flags unchanged.

### Arms

| arm | stamp | flags | store |
|-----|-------|-------|-------|
| clear | `20260820_091420` | wipe + no-save | skip save, 19 RAM wills die |
| mint | `20260820_091707` | keep_memory=1 | **saved 19** → `data/splat_memory.safetensors` |
| reload | `20260820_091747` | keep_memory=1 | **Loaded 19** then grew to 28 |

inject= empty on all three. BOS `first_id=2` all turns. `enabled_path=true` force_cap=1.

### Chat KPIs (reload, replies not `grep gemma4>`)

Hello there, friend. / 2+2 is 4. / Blue. / C-A-T. / One, two, three. / two residual-stream sentences / Ready / 51+340=391 / pangram. No Wait, no `theed`, no math thrash.

### Memory channels (probe)

Clear t1s0: `scars_active=1 splat_mag=0.0`  
Mint t1s0: `scars_active=1 splat_mag=0.0` (same-session deposits sit far — F_s 0 this process)  
Reload t1s0: `scars_active=19 splat_mag=0.0` then later:

```
reload t8 step=48 token="----" splat_mag=0.1849 scars_active=26 force_on=true
reload t8 step=65 token="1"    splat_mag=0.1361 scars_active=27
```

Mint/clear: **zero** toks with splat_mag>0. Reload: 7 toks, max **0.1849**.

Stdout mint: `[CHAT MEMORY] saved 19 wills`  
Stdout reload: `Loaded 19 splats from data/splat_memory.safetensors` then `[CHAT WILL] turn=1 wills=20`

GPU-free: `chat_will_deposit_save_load_query_force_roundtrip` (deposit_chat_will → persist → load → query_force ≠ empty).

Signed: Grok (xAI) · operator Jason

## Findings

Mint saved 19 wills. Reload loaded 19 (turn 1 wills=20). Probe splat_mag max 0.1849 on reload vs 0 on clear and mint. Chat KPIs hold. inject empty. Continuity is the store, not a tag.

## Next

Do not rewalk 070050 / 070557 / 073954 / this mint-reload. Force-in-revise still adjacent. 131k still not this lane.

---

Signed: Grok (xAI) · operator Jason
