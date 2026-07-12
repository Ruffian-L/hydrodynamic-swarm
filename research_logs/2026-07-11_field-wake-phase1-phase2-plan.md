# Field Wake Plan + Ablation Tables (Phase 1 & 2)

**Date:** 2026-07-11  
**Context:** Diderot audit proved emb cloud is unit shell (pairwise ~1.43, σ~11, r*~570). Residual ||h||~450 has ρ=0, ∇ρ=0. Pure F_g is geometrically dead.

**Implementation:** `niodoo.rs` — `FieldWakeMode` + `nearest_emb_wake`  
**Config:** `field_wake_*` in `config.toml`

---

## Geometry → GMM reading

Treat token embeddings as a **Gaussian mixture** on a thin shell:

\[
\rho(x)=\sum_i \pi_i\,\mathcal{N}(x;\mu_i,\sigma^2 I)
\quad\text{(we use unnormalized }G_i=\mathrm{e}^{-r_i^2/\sigma^2}\text{)}
\]

| GMM view | In our code |
|----------|-------------|
| Full mixture density | `probe` / `probe_gradient` |
| **Local responsibility** (hard/soft nearest component) | **nearest-emb wake** |
| Sink basins (div F < 0) | on-manifold only |
| Off-support residual | need **local component pull** μ* − h |

Phase 1 = **local GMM component force**. Phase 2 = **bandwidth / normalization of the mixture**.

---

## Phase 1 — Nearest-embedding wake (architectural)

**Goal:** Make F_g fire from residual space.

| Step | Change | Config / variant | Measure | Expected |
|------|--------|------------------|---------|----------|
| **1.0** | Baseline pure ∇ρ | `field_wake_mode = "off"` | mean/max F_g, F_s, F_a, δ, pain splats, first 150 tok | F_g ≈ 0, goal dominates |
| **1.1** | Nearest-emb wake k=1 | `mode=wake`, `k=1` | same | F_g rises (~cap 40) from residual |
| **1.2** | Wake + small ∇ρ blend | `mode=blend`, `field_grad_blend=0.15` | same | hybrid; ∇ρ dead → ≈ wake |
| **1.3** | Distance-weighted wake | `mode=dist_weighted`, `τ=80` | same | less over-pull when closer to cloud |

**Protocol (each step):**
- Prompt: `Explain the Physics of Friendship in one paragraph.`
- Tokens: 200 (or 80 for smoke)
- `--clear-memory`
- Log: `logs/*gemma*.jsonl` + note in this file

### Phase 1 results log

| Step | variant | mean F_g | max F_g | mean F_s | mean F_a | mean δ | pain# | coherence (1–5) | notes |
|------|---------|----------|---------|----------|----------|--------|-------|-----------------|-------|
| 1.0 | off | ~0 | ~0 | (prior) | ~50 | | | 2 | pure ∇ρ dead (field_audit) |
| **1.1** | **wake_k1 max40** | **35.5** | **40.0** | **31.4** | **49.8** | **79.7** | **14 pain / 5 pleasure** | **3 early / 1 late** | F_g LIVE. REFLEX@100 |
| **1.1b** | **wake_k1 max30** | **29.9** | **30.0** | **36.7** | **49.7** | **79.3** | pain high late | **2–3 early / 1 late** | Cap works; late collapse not fixed |
| 1.2 | blend | | | | | | | | |
| **1.3** | **dist_weighted (fixed)** | **~9.6** | **~10** | **~40** | **50** | **~80** | pain late | **3 early / 1 late** | F_g no longer glued to max. F_s still →50 late |
| 1.3a | dist_weighted (bugged k=1) | ~30 | 30 | ~same as 1.1b | 50 | | | | **BUG:** /wsum cancelled falloff; discarded |

**1.1 run detail** (`logs/2026-07-11_14-23-26_...jsonl`, 200 tok):

| window | mean F_g | mean F_s | mean F_a | mean δ |
|--------|----------|----------|----------|--------|
| 0–50 | **39.7** | 8.5 | 49.0 | 68.1 |
| 50–100 | 35.4 | **48.7** | 50.0 | 82.8 |
| 100–150 | 35.1 | 35.9 | 50.0 | 84.8 |
| 150–200 | 31.7 | 32.6 | 50.0 | 83.0 |

- Banner: `Field wake: mode=wake k=1 scale=0.2 max=40`
- Milestone `[50/200] F_g=38.3 F_s=37.3 F_a=50.0` — **three-way force balance**
- Pleasure early (`interactions` p=0.97, `concept` p=0.95); Pain flood after ~100 + 1 REFLEX
- First ~80 tokens: on-topic physics/friendship metaphor; then instruction-bleed / garble
- **Verdict:** Phase 1.1 **succeeds on F_g liveness** (was 0 → mean 35.5). Quality still late-run fragile — try 1.3 or lower `field_wake_max` before Phase 2 σ sweeps.

### Option A: wake_k1_max30 (2026-07-11)

Config: same as 1.1, only `field_wake_max = 30`. Log: `logs/2026-07-11_14-35-51_...jsonl`.

| window | F_g | F_s | F_a | δ | ocean |
|--------|-----|-----|-----|---|-------|
| [50] | **30.0** | 29.5 | 50 | 75.4 | n=12 noise=0.43 F_o=34 |
| [100] | **30.0** | **50** | 50 | 78.6 | n=25 noise=0.33 F_o=26 |
| [150] | **30.0** | **50** | 50 | 83.8 | n=37 noise=0.25 F_o=30 |

- F_g **hard-capped at 30** (no longer 40 thrash).
- Late collapse **persists** (Pain flood, REFLEX@100, garble after ~80–100).
- **F_s climbs to 50** (its own cap) mid-run — memory scars may dominate late thrash more than wake max.
- **Conclusion:** Option A alone insufficient. Go **1.3 dist_weighted** and/or quality-gate ocean deposits; investigate REFLEX false-positive (below).

---

## Double-check: REFLEX trigger

**Where:** `main.rs` + `ridge::check_vr_h1_reflex`

| gate | value |
|------|--------|
| when checked | `step > 50 && step % 100 == 0 && (step - last_reflex) >= 100` |
| so fires at | **100, 200, 300…** only (scheduled, not continuous) |
| window | last **8** raw hidden states |
| criterion | any triple with `d_max / d_mid < threshold` |
| threshold in call | **`2.0`** |

**Honesty check:** Zero-persistence H1 would need `d_max/d_mid ≈ 1.0` (e.g. threshold **1.05** as the comment suggests). Threshold **2.0** is extremely loose — most residual triples with comparable norms satisfy it. So REFLEX@100 is nearly **always-on every 100 steps**, not a rare topological emergency.

**On fire:** `steered = 0.7*steered + 0.3*baseline` (mild pullback).

**Shared with ocean?** **No.** REFLEX does not read ocean noise, deposits, or F_ocean. Independent path.

**Action ideas (later):** raise bar to `threshold=1.05–1.15`, or require ≥N bad triples, or disable until F_s/pain rate spikes.

---

## Double-check: Shared ocean decay

**Where:** `ocean.rs` — **not** splat `memory.decay_step`

| knob | default | 1.1 run behavior |
|------|---------|------------------|
| deposit | every **4** tokens | host residual, weight=1, initial_noise=**0.65** |
| refine | every **2** deposits | μ ← blend toward consensus; noise *= **0.88** |
| consensus_blend | 0.12 | packets pull together |
| force | `scale * w_i * (1-noise) * (μ_i - pos)` | stronger as noise drops |
| max_packets | 64 | truncate high-noise first |

**Observed noise trajectory (milestones):** 0.43 @50 → 0.33 @100 → 0.25 @150 → **0.199** end  
→ ocean **is crystallizing** (decay working as designed).

**Risk:** deposits are **quality-agnostic** (every 4 steps). Late garbage residual gets refined into consensus → F_ocean keeps pulling (~26–34) toward a contaminated mean while F_s also maxes. That can **amplify** late thrash even as mean_noise falls.

**Shared with splat decay?** **No.** Splat uses `memory.decay_rate` / evaporate; ocean uses `residual_noise *= noise_decay` only on refine. Two clocks.

**Action ideas:** deposit only on Pleasure quality; raise `noise_decay` slower (0.95); or cut `force_scale` after step 80.

---

## Phase 1.3 dist_weighted (fixed formula) — 2026-07-11

### Bug found then fixed

For **k=1**, old formula `F ∝ w·(μ−x) / w` cancelled distance weight → **identical to wake**.  
**Fix:** unitize direction, then `strength = scale·200 · falloff(d)`, with  
`falloff = 1/(1+d/τ)` only in `DistWeighted`. Soft-cap uses `max_mag * falloff`.

At residual d≈450, τ=80 → falloff≈0.15 → F_g ≈ **4–10** (not 30).

### Run (fixed)

Log: `logs/2026-07-11_14-47-42_...jsonl`  
Banner: `mode=dist_weighted k=1 scale=0.2 max=30 τ=80`

| window | F_g | F_s | F_a | δ |
|--------|-----|-----|-----|---|
| [50] | **9.6** | 34.6 | 50 | 78.3 |
| [100] | **10.0** | **50** | 50 | 77.0 + REFLEX |
| [150] | **9.4** | **50** | 50 | 83.9 |

| metric | 1.1 max40 | 1.1b max30 | **1.3 dist (fixed)** |
|--------|-----------|------------|----------------------|
| mean F_g | 35.5 | 29.9 | **~9–10** |
| mean F_s | 31.4 | 36.7 | high late (→50) |
| late quality | bad | bad | still bad |

### First ~100 tokens (1.3 fixed)

> a friendships can be understood through several physical principles, though not as an emergent phenomenon arising from complex interactions and social physics'sically understanded… mutual attraction… then degrades by ~80–100 into meta/instruction bleed.

### Verdict

| Question | Answer |
|----------|--------|
| Does dist_weighted change F_g? | **Yes** (after formula fix): 30 → ~10 |
| Does it fix late collapse? | **No** — F_s→50 + ocean + REFLEX still dominate |
| Next | Quality-gate **ocean + splat** late deposits; tighten REFLEX threshold; optional lower `splat_force_max` |

---

## Phase 2 — Sigma / field normalization (after Phase 1)

**Goal:** Once wake is live, tune how far field influence reaches **without** near-cloud instability.  
**Rule:** change **one** variable per step.

| Step | Change | Config / variant | Measure | Expected / risk |
|------|--------|------------------|---------|-----------------|
| **2.0** | Phase 1 winner frozen | best of 1.x | baseline table | reference |
| **2.1** | Wider emb σ | `field` rebuild σ=15 | F_g pure on near-cloud probes; residual still needs wake | wider tails; may blur sinks |
| **2.2** | σ=20 | same | same | more overlap of emb Gaussians |
| **2.3** | σ=25 | same | same | risk: flatter landscape on shell |
| **2.4** | Soft normalize force | `F ← F / (1 + d²/τ²)` on wake | max F_g, loop rate | tame long-range snap |
| **2.5** | Global field multiplier | `viscosity_scale` × {0.5,1,1.5,2} | force balance F_g vs F_a | simple gain sweep |
| **2.6** | Local+global mixture | blend mode + σ from 2.x | all forces + coherence | full GMM-style hybrid |
| **2.7** | Sink-aware cap | cap F_g by r*/d | overshoot / garble | prevent residual→emb slam |

### Phase 2 results log

| Step | variant | mean F_g | max F_g | mean F_s | mean F_a | mean δ | pain# | coherence (1–5) | notes |
|------|---------|----------|---------|----------|----------|--------|-------|-----------------|-------|
| 2.0 | phase1_winner | | | | | | | | |
| 2.1 | sigma_15 | | | | | | | | |
| 2.2 | sigma_20 | | | | | | | | |
| 2.3 | sigma_25 | | | | | | | | |
| 2.4 | soft_norm | | | | | | | | |
| 2.5 | visc_gain | | | | | | | | |
| 2.6 | local_global | | | | | | | | |
| 2.7 | sink_cap | | | | | | | | |

---

## How to switch variants

```toml
# config.toml
field_wake_mode = "off"           # 1.0
field_wake_mode = "wake"          # 1.1
field_wake_mode = "blend"         # 1.2
field_wake_mode = "dist_weighted" # 1.3
field_wake_k = 1
field_wake_scale = 0.20
field_wake_max = 40.0
field_grad_blend = 0.15
field_wake_dist_tau = 80.0
```

```bash
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 200
```

---

## Decision rule

1. Prefer lowest garble in first 150 tokens **with** mean F_g > 1.  
2. F_g should not exceed F_a by >2× (avoid emb-snap thrashing).  
3. Only then enter Phase 2 sigma sweeps.

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---

