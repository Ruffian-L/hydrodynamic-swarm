# Splat scaling gap — 4B S1–S4

**Date:** 2026-07-11  
**Authors:** Jason (diagnosis) · Grok (xAI) (fix + lane)  
**Model:** `gemma-3-4b-it-Q4_K_M.gguf` (hidden=2560, field auto-σ≈7.59)  
**Harness:** `./splat_lane_4b.sh` · `logs/splat_lane_4b/`

---

## Diagnosis (confirmed)

We scaled **steering force caps** (Algo √-law) but **not the splat system**:

| Knob | 27B-era default left on 4B | Problem |
|------|----------------------------|---------|
| `splat_sigma` | **40** | Scars too wide for residual walk coupling on 4B (and/or wrong band) |
| hierarchical `with_scale` | absolute **δ>20/30 → Coarse×4** | 4B δ~80–120 ⇒ **every scar Coarse×4** (σ_eff up to 160) |
| `splat_force_scale/max` | 0.25 / 28 | Same mass as large-model lane |
| `min_splat_dist` / deposit thr | 30 / 70 | Not retuned to 4B residual geometry |

Field emb auto-σ (~7.6) ≠ residual scar width, but both must be **re-thought on size change**. Force √-law alone is not enough.

**Code fix shipped:** `Splat::with_scale_ref(δ, splat_delta_threshold)` — bands at 0.85× / 1.25× threshold. `main.rs` uses it. Absolute 20/30 only remains on legacy `with_scale`.

---

## S1–S4 @ 55 tok (ramp 15@0.15, targeted ON)

| ID | Change | early F_s | late F_s | max F_s | pain | note |
|----|--------|----------:|---------:|--------:|-----:|------|
| **S1** | σ=40, mass 0.25/28 (old geometry) | 1.1 | **22.2** | 27.1 | 4 | late climb returns |
| **S2** | mass only 0.12/14, interval 8 | 4.2 | **14.0** | 14.0 | 1 | latched at **new** ceiling |
| **S3** | σ=12, min_dist=10, thr=95 | 0.0 | **0.09** | 0.9 | 2 | **residual-cold** — scars don't couple |
| **S4** | S2+S3 combined | 0.0 | **0.03** | 0.6 | 1 | same cold; F_s dead |

δ late ~108–111 all variants — again **δ alone misses the splat story**.

### Prose (short)

- **S1:** solid open, classic physics/social; late path still scar-heavy.  
- **S2:** on-topic; mass limited.  
- **S3/S4:** still readable at 55 tok (goal/wake carry) but **splats effectively off** — not a “scaled splat” win, a mute.

---

## Interpretation

1. **Hierarchy bug was real** — fixed for all subsequent runs.  
2. **σ=12 ≈ field emb scale is too narrow for residual forces** (step δ~100). Copying field auto-σ straight into `splat_sigma` **under-couples**.  
3. **σ=40 late F_s climbs** even with relative hierarchy — width/mass still hot.  
4. **S2 mass caps work** but pin to max — need mid width so force is present *without* needing the ceiling.  
5. Long 1000-tok noir run chaos matches S1-class: F_s→28, Pain flood, trajectory yank.

### Post-lane default (candidate, not multi-seed proven)

Mid geometry + soft mass (between S1 and S3):

```toml
splat_sigma = 22.0
min_splat_dist = 16.0
splat_delta_threshold = 90.0
splat_force_scale = 0.14
splat_force_max = 16.0
online_splat_interval = 8
# + B4 ramp/targeted/force_cap 3.1
```

Smoke this before another 1000-tok novel.

---

## Checklist when changing model size (updated)

1. Algo √-law → force caps / ramp / type_mult  
2. **Splat geometry** → `splat_sigma`, `min_splat_dist`, deposit thr (residual, not only field σ)  
3. **Splat mass** → `splat_force_scale`, `splat_force_max`, interval  
4. Hierarchy **relative** to deposit threshold  
5. Short smoke 50–60 tok → watch **late F_s + ceiling%**, not only δ  
6. Only then long runs

Script: `scripts/scale_physics_for_model.py` now emits splat geometry priors for small models.

---

## Reproduce

```bash
cd /home/ruffianl/projects/hydrodynamic-swarm
./splat_lane_4b.sh
# money mid-fix smoke:
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 90
```

---

**Authorship:** Jason · Grok (xAI)  
**Failure logged:** size-change without splat retune → F_s latch + Pain spam / long-run garble.
