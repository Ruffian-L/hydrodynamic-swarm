# F decay · quant effects · B4b retune

**Date:** 2026-07-11  
**Authors:** Jason · Grok (xAI)  
**Model under test:** `gemma-3-4b-it-Q4_K_M.gguf` only (no 4B Q8 in `data/google/`)  
**Harness:** `./f_decay_lane_4b.sh` · artifacts `logs/f_decay_lane_4b/`

---

## 1. F decay — what exists vs what was broken

| Mechanism | When it ran | Role |
|-----------|-------------|------|
| `query_force` **1/√n** | every steer | sublinear multi-scar sum |
| `splat_force_scale` / `_max` | every steer | mass + L2 latch cap |
| **`decay_per_token`** (new) | each gen step if `online_decay_rate < 1` | mid-run scar **α** fade → F_s |
| `decay_step` (wall-clock) | Phase 5 / dream only | inter-session / end-of-run |
| `memory.decay_rate` | fallback inside `decay_step` when Δt=0 | legacy |
| ocean `noise_decay` | refine packets | crystallization (can lock junk) |
| force **ramp** | first N tokens | total force schedule, not scar mass |

### Bugs found

1. **No mid-generation scar decay** — alphas only evaporated after the full loop, so long runs kept full-strength scars.  
2. **Wall-clock `decay_step` double-count** — old code did `α *= exp(−λ · age_from_create)` every call; calling every token would over-decay. Fixed to use **Δt since last `decay_step`**.  
3. **`splat_lambda_default` was dead config** — deposits always used hardcoded `λ=0.02`. Now `with_scale_ref_lambda(...)`.

### Correct mid-run control

```text
memory.online_decay_rate = 0.975   # each token: pleasure α *= 0.975
physics.pain_decay_factor = 0.7    # pain fades slower (lasts longer)
# every 25 steps: cull |α| < prune_threshold
```

Do **not** call wall-clock `decay_step` every token.

---

## 2. Quantization effects (investigation)

### What we have on disk

| GGUF | Arch | Notes |
|------|------|--------|
| **gemma-3-4b-it-Q4_K_M** | gemma3 | **only 4B** — all B4 lanes |
| gemma-3-27b-it-Q4_K_M | gemma3 | large |
| gemma-3-27b-it-Q8_0 | gemma3 | large, higher fidelity |
| gemma-3n / gemma4 | other | **not loadable** |

### Expected physics impact (theory + practice)

| Layer | Q4 vs Q8 | Action |
|-------|----------|--------|
| **Hidden dim / arch** | same | no √-law change |
| **Force mass (Algo)** | same | do **not** re-scale force for quant alone |
| **Splat geometry** | residual walk similar | same σ/mass family; retune if δ distribution shifts |
| **Sampling noise** | Q4 noisier logits | slight ↑ temperature or ↑ rep_penalty often helps |
| **Field emb auto-σ** | may shift slightly | rebuild field per model file; don't copy σ across quants blind |
| **Long-run stability** | Q4 more brittle under hard F_s | prefer softer scars + online decay on Q4 |

**Rule:** quant is **fidelity**, not **mass**. Size/type √-law stays; Q4 may need gentler **scar α** and online decay, not higher `force_cap`.

No A/B Q4 vs Q8 on 4B (no Q8 file). When a 4B Q8 lands, smoke same B4b knobs 70 tok and compare late F_s / entropy / prose only.

---

## 3. B4a / B4b / B4c @ 70 tok (4B Q4)

Shared base: σ=30, mass 0.14/16, ramp 15@0.15, targeted, mid-run prune.

| variant | online_decay | early F_s | late F_s | max F_s | mean F_a | note |
|---------|--------------|----------:|---------:|--------:|---------:|------|
| **B4a** | off (1.0) | 0.98 | 2.85 | 5.7 | **37.7** | money geometry, F_a still hot |
| **B4b** | **0.975** + soft goal | 0.09 | **1.05** | **1.6** | **30.5** | **default** |
| **B4c** | 0.95 + soft goal | 0.35 | 1.14 | 3.1 | 29.3 | scars almost mute; not better prose |

**B4b wins** on F_s control without going residual-cold. Late δ still ~108–110 (δ ≠ F_s story).

### B4b knobs (now `config.toml`)

```toml
# process
force_ramp_tokens = 15
force_ramp_start = 0.15
targeted_splat_only = true
# residual-mid splats
splat_sigma = 30.0
splat_force_scale = 0.14
splat_force_max = 16.0
# F decay
online_decay_rate = 0.975
splat_lambda_default = 0.03
# softer goal (was ~38 Fa)
goal_force_scale = 0.10
goal_force_max = 32.0
manifold_pullback = 0.28
# Q4-friendly sample
temperature = 0.82
rep_penalty = 1.32
pleasure_alpha = 1.0
pain_alpha = -0.5
max_splats = 48
```

---

## 4. Code shipped

- `SplatMemory::decay_per_token(rate, pain_factor)`  
- `decay_step` uses Δt since last call  
- `memory.online_decay_rate` in config + validation  
- deposits via `with_scale_ref_lambda(..., splat_lambda_default)`  
- gen loop: per-token decay + cull every 25 steps  

---

## Reproduce

```bash
cd /home/ruffianl/projects/hydrodynamic-swarm
./f_decay_lane_4b.sh
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 90
```

---

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** F decay fix, quant notes, B4b retune lane
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---
