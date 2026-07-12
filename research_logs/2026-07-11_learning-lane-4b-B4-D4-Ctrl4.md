# Learning Lane 4B — B4 / D4 / Ctrl4 (90 tok)

**Date:** 2026-07-11  
**Authors:** Jason (directive / co-engineer) · Grok (xAI) (run + analysis)  
**Model:** `data/google/gemma-3-4b-it-Q4_K_M.gguf` (gemma3, hidden=2560)  
**Prompt:** Physics of Friendship  
**Harness:** `./learning_lane_4b.sh`  
**Artifacts:** `logs/learning_lane_4b/{B4_ramp_targeted,D4_jspace,Ctrl4_scaled_only}.{jsonl,stdout}` · `summary.tsv`

---

## Locked design

| ID | Focus | Diff vs scaled base |
|----|-------|---------------------|
| **B4** | Ramp + targeted | ramp 15 @ 0.15, `targeted_splat_only=true`, goal_scale=0.125 |
| **D4** | J-space respect | ramp 18 @ 0.10, targeted, `prefill_micro_dream=true`, goal_scale=0.08 |
| **Ctrl4** | Formula mass only | **no ramp**, **targeted off**, same force caps |

Shared scaled mass (Algo √-law 4B instruct): `force_cap=3.1`, `splat_force_max=28`, wake/goal ceilings from scale script.  
**No geometry retune this pass** (field auto-σ=7.59 noted; deferred).

---

## Headline results

| variant | early δ (0–29) | late δ (60–89) | early F_s | late F_s | F_s @ ceiling | mean F_g | mean F_a |
|---------|----------------:|---------------:|----------:|---------:|--------------:|---------:|---------:|
| **B4** | 71.5 | 114.8 | **16.6** | **16.1** | **14%** (13/90) | 8.1 | 37.4 |
| **D4** | 71.0 | 111.1 | 14.3 | 21.0 | **1%** (1/90) | 8.3 | **24.2** |
| **Ctrl4** | 74.3 | 114.7 | 21.3 | **28.0** | **88%** (79/90) | 7.7 | 38.3 |

### Mid window (30–59) — where Ctrl4 dies

| variant | δ | F_s | note |
|---------|--:|----:|------|
| B4 | 110.7 | 24.4 | peaks mid, **then eases** late |
| D4 | 106.7 | 16.3 | gentlest mid |
| Ctrl4 | 111.2 | **28.0** | **glued to splat ceiling** from ~30 onward |

---

## Interpretation (data-driven)

1. **√-law mass alone is not enough process.** Ctrl4 has the same caps as B4 but spends **88% of tokens at F_s=28**. That is the late-run thrash path we feared — even on 4B, without ramp + targeted.
2. **B4 wins the F_s story.** Early gentle → mid climb → **late F_s drops back to ~16** (max late only 17.7). Ramp + selective splats actually *unload* scar force late instead of latching the ceiling.
3. **D4 softens F_a (24 vs ~38)** as designed (weaker goal + longer ramp). Almost never hits F_s ceiling, but **late F_s still climbs** (14 → 21) and prose is the mushiest of the three.
4. **δ late is similar (~111–115) for all three.** Process knobs mainly change **how hard scars push**, not the residual walk magnitude. δ alone would miss the Ctrl4 failure mode — **always report F_s + ceiling hit rate**.
5. **Text:** all stay on-topic early; all fray after ~50–60 tok (complex / gravity / network bleed). **Not gibberish / not Fason.** B4 open is cleanest; none produce a tight one-paragraph finish at 90 tok. Length stress is real; chaos is not.

### Full outputs (trimmed)

**B4:**  
> Friendship, surprisingly exhibits a fascinating interplay between physics and social dynamics—like complex interactions within human relationships can be understood through principles like physical science as intricate network-based networks involving gravity dynamic interaction… *(late: arraying systems / complexinceptions)*

**D4:**  
> Friendship, surprisingly has a fascinating physics-like many aspects to friendship is not simply an complex phenomenon… *(grammar breaks earlier; soft goal shows)*

**Ctrl4:**  
> Friendship is surprisingly complex, yet a fascinating phenomenon rooted within physics and social dynamics… *(solid open; mid/late pinned at max F_s with network/energy bleed)*

---

## Decision

| Keep as 4B default | **B4** — ramp 15 @ 0.15 + targeted + scaled mass |
|--------------------|---------------------------------------------------|
| Do not default | **Ctrl4** — formula without process (ceiling latch) |
| Optional experiment | **D4** — when testing softer goal / prefill dream; not best prose |

`config.toml` restored to **B4** after the harness run.

---

## What this does *not* claim

- No geometry / wake retune (auto-σ=7.59 still open).  
- No multi-seed stats (n=1 per variant).  
- No “physics improved reasoning” claim — control is stability + F_s process.  
- 90 tok is enough to see late F_s latch; not enough for polished long-form.

---

## Next lanes (Jason steers)

1. **Geometry smoke:** lower `splat_sigma` / wake τ toward emb-scale (σ~7.6) under **B4** only.  
2. **B4 × 2–3 seeds** at 90 tok for variance.  
3. **Stop-early / quality exit** if δ stays high but F_s controlled (accept soft landing).  
4. Do **not** re-open Ctrl4-style “full splat every step” on small models.

---

**Authorship:** Jason · Grok (xAI)  
**Note:** Failures logged — Ctrl4 ceiling latch is the re-walk trap when “just scale force” without ramp/targeted.
