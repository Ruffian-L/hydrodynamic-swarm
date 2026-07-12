# Telemetry eyeball + mini force-balance sweep

**Date:** 2026-07-11

## What "memory" was attracting toward (honest)

| Force | Source | Role after damp fix |
|-------|--------|---------------------|
| **F_a goal** | Frozen **prefill last hidden state** (prompt residual) | Was monopoly (~450). Not "good text memory." |
| **F_s splat** | Online deposits of **current residual** at high-δ steps + end pleasure | After anti-runaway: **~0** mid-run — kernel dead or scale too low |
| **F_g field** | Token embedding cloud ridge | **Always 0** — residual ≠ emb manifold |
| **F_ocean** | Host residual packets every 4 steps | Real but secondary |

So garbling with "coherent enough" = **mostly goal-pull toward prompt embedding + model prior**, not a curated memory museum.

## Mini sweep (40 tok, 5 configs)

See `logs/mini_sweep_summary.tsv`.

| run | mean_Fs | mean_Fa | mean_δ | snip |
|-----|---------|---------|--------|------|
| A baseline goal×1 | 0.25 | **462** | 93 | a friendships… |
| B goal damped | 0.12 | **58** | 66 | a friendships… |
| C balanced | 0.24 | 58 | 67 | a friendships… |
| D memory bias | 0.69 | 39 | 71 | explain physics concepts… |
| E gentle | 0.08 | 49 | 67 | …emergent phenome… |

**Takeaways**
1. Goal damp works (Fa 462→~50).
2. Splat still barely fires even at scale 0.4 — geometric (σ vs residual distance), not just scale.
3. F_g remains dead; field is not in the game yet.
4. No second ocean depositor needed until F_a/F_s balance is real.

## Default after sweep

`config.toml` set toward E/C hybrid + wider `splat_sigma=40` so scars can engage.

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---

