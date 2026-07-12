# Model Size ↔ Physics Steering

**Share this when anyone changes model size** (Gemma / Llama / Qwen under Niodoo-style residual physics).

| | |
|--|--|
| **Source algo** | `Documents/Algo_WIPjuly.md` — golden 1B/3B/4B runs + √-law |
| **Swarm mapping** | This file + `scripts/scale_physics_for_model.py` |
| **Authors** | Jason (algo experiments / golden configs) · Grok (xAI) (swarm mapping, share doc) |
| **Date** | 2026-07-11 |

---

## The one sentence

**Smaller models need gentler force.**  
They have less “mass” to absorb a shove, so knobs that barely stabilize a 27B will **shatter or over-govern** a 1B–4B.

$$
\text{Steering force} \propto \sqrt{\frac{\text{params (B)}}{3}} \times \text{type\_mult}
$$

**Golden reference:** 3B **standard** → comfortable mid-zone (σ≈0.15, θ≈2.0 in Algo vocabulary).

---

## Do you need the scaling algo?

| Question | Answer |
|----------|--------|
| Changing model size? | **Yes — run the script, don’t copy 27B `config.toml` blind.** |
| Same size, different quant (Q4 vs Q8)? | Usually **same force knobs**; quant is fidelity/speed, not mass. Smoke once. |
| Same size, different *type* (instruct → thinking)? | **Yes — type_mult often matters more than size.** |
| Only changing prompt / token budget? | No size re-scale; keep ramp + short smokes. |

```bash
# From hydrodynamic-swarm root:
python3 scripts/scale_physics_for_model.py --params 4 --type instruct
python3 scripts/scale_physics_for_model.py --params 4 --type instruct --toml   # paste fragment
python3 scripts/scale_physics_for_model.py --params 1 --type standard --algo-only  # classic σ/θ/β
```

---

## Original symbols (Algo_WIP) → swarm knobs today

Algo_WIP used an older Niodoo process vocabulary. Hydrodynamic-swarm uses different names; **same idea**.

| Algo_WIP | Meaning (plain) | Swarm / `config.toml` today | Notes |
|----------|-----------------|------------------------------|--------|
| **σ (sigma)** | “jiggle” / process noise | `force_cap`, `field_wake_max`, `splat_force_max` | **Not** `splat_sigma` (scar width) |
| **θ (theta)** | drift correction / mean reversion | `goal_force_scale` + `goal_force_max`, `manifold_pullback` | Goal = J-space prefill attractor |
| **β (beta)** | inverse temperature | `generation.temperature` (set T directly; high β ≈ colder) | Type mult does **not** scale β in Algo |
| **loop_repulsion** | push off stuck loops | pain splats + quality gate + REFLEX (sparingly) | Don’t always-on thr=2.0 |
| **dt** | fixed 0.1 in Algo | `dt` (swarm often ~0.035) | Keep unless re-tuning whole engine |
| hierarchical splat | fine/medium/coarse scars | `Splat::with_scale(delta_norm)` | **Keep for all sizes** |
| force ramp | gentler early tokens | `force_ramp_tokens`, `force_ramp_start` | Stronger on small models |
| targeted tokens | don’t physics every step | `targeted_splat_only` + quality gate | Keep **true** on all sizes |

### Two different “sigmas” (common footgun)

| Name | What it is | Scale with √(params/3)? |
|------|------------|-------------------------|
| Algo **σ** | Force / noise **intensity** of steering | **Yes** |
| Config **`splat_sigma`** | Gaussian **scar width** in residual space | **No** — retune with residual geometry / emb norms |

Force caps and wake/goal scales follow the √ size rule first.  
Field geometry (emb shell, residual ‖h‖) is about **dimension and norms**, not only param count.

---

## The algorithm (from Algo_WIPjuly)

### 1. Core scaling law

Force magnitude scales with model “mass.” Smaller models have fewer parameters absorbing the shove → destabilize faster.

```text
GOLDEN_PARAMS = 3.0          # 3B reference
scale         = sqrt(params_B / 3.0)
type_mult     = { standard:1.0, instruct:0.9, chat:1.1, thinking:0.4, coding:0.27 }
intensity     = scale * type_mult

# Algo process (clamped to traversable zone)
σ   = clamp(0.15 * intensity, 0.04, 0.20)
θ   = clamp(2.0  * intensity, 0.5,  3.0)
β   = clamp(100  * scale,     40,   150)    # no type_mult
rep = clamp(2.0  * intensity, 0.3,  3.0)
```

### 2. Experimental table (Algo golden runs)

| Model | Size | σ | θ | β | loop_rep | Quality |
|-------|------|---|---|---|----------|---------|
| Llama-3.2-1B | 1B | 0.15 (too hot) | 2.0 | 100 | 2.0 | **UNSTABLE** — gibberish at high params |
| Llama-3.2-3B | 3B | 0.15 | 2.0 | 100 | 2.0 | **GOLDEN** — stable, coherent |
| Qwen2.5-3B | 3B | 0.10 | 1.5 | 80 | 1.5 | GOOD — slightly reduced |
| DASD-4B-Think | 4B | 0.06 | 0.8 | 50 | 0.8 | GOOD — thinking needs gentle |

**Predicted after √-law (standard / with type):**

| Model | params | scale | type | σ | θ | β | rep |
|-------|--------|-------|------|---|---|---|-----|
| Llama-1B | 1 | 0.577 | standard | 0.087 | 1.15 | 57.7 | 1.15 |
| Llama-3B | 3 | 1.000 | standard | 0.150 | 2.00 | 100 | 2.00 |
| Qwen-3B | 3 | 1.000 | instruct | 0.135 | 1.80 | 100 | 1.80 |
| DASD-4B | 4 | 1.155 | thinking | 0.069 | 0.92 | 115 | 0.92 |
| Gemma-3-4B | 4 | 1.155 | instruct | ~0.156 | ~2.08 | 115 | ~2.08 |
| Llama-7B | 7 | 1.528 | standard | 0.200† | 2.50 | 150† | 2.50 |
| 27B instruct | 27 | 3.0 | instruct | 0.200† | 3.00† | 150† | 3.00† |

† Hits Algo hard clamps (same spirit as swarm CEILING on force caps).

### 3. Traversable stability zone

```
                    CHAOS ZONE
              (σ > 0.20 → garbled / Fason)
                         │
    ┌────────────────────┼────────────────────┐
    │   TRAVERSABLE      │   TOO HOT          │
    │   steers + coherent│   oscillates       │
    │   σ ∈ [0.10, 0.18] │   σ > 0.18         │
    │   θ ∈ [1.5, 2.5]   │   θ > 2.5          │
    ├────────────────────┼────────────────────┤
    │   TOO COLD         │   DEAD / FROZEN    │
    │   no steering      │   Buridan / argmax │
    │   σ < 0.08         │   σ < 0.04         │
    │   θ < 1.0          │   θ < 0.5          │
    └────────────────────┴────────────────────┘
```

| Failure (Algo name) | Meaning | Swarm symptom |
|---------------------|---------|----------------|
| Buridan’s Ass / frozen | σ too low | δ flat, text ignores physics |
| Fason singularity / chaos | σ too high | garble, meta loops, Pain flood |
| Oscillation | θ too high | thrash, REFLEX spam, overshoot |
| Over-governing | caps always on / too loud | “balanced” forces, dead prose (MountainCar governor lesson) |

---

## Model type often matters **more** than size

$$
\text{Total force} = \underbrace{\sqrt{\text{params}/3B}}_{\text{inertia (size)}} \times \underbrace{\text{type\_mult}}_{\text{fragility (topology)}}
$$

| Type | Mult | Why |
|------|------|-----|
| **standard** | 1.0 | Sleepwalker / System-1 — needs a real kick out of average attractors |
| **instruct** | 0.9 | Follows ghost “instructions” more easily; slightly reduced |
| **chat** | 1.1 | RLHF persona well can need extra force to leave refusal/generic rails |
| **thinking** | **0.4** | CoT is a house of cards / tightrope — same shove shatters logic |
| **coding** | **0.27** | Syntax wall — noise kicks you out of valid programs |

A **4B thinking** model needs ~half the force of a **3B standard** even though it is larger.

---

## Map intensity → hydrodynamic-swarm knobs

**Anchor = 3B standard mid-zone** (not “copy 27B and hope”):

| Swarm knob | Role | Scale with intensity? |
|------------|------|------------------------|
| `force_cap` | per-dim clamp | **yes** |
| `splat_force_max` | scar ceiling | **yes** |
| `field_wake_max` | emb wake ceiling | **yes** |
| `goal_force_max` | J-space pull ceiling | **yes** |
| `goal_force_scale` | goal strength | **yes** |
| `field_wake_scale` | wake strength | **yes** |
| `force_ramp_start` | early gentleness | **lower** on small (0.10–0.15) |
| `force_ramp_tokens` | ramp length | **longer** on small (15–18) |
| `field_logit_alpha` | surface tip | **0** on small; A/B later |
| `temperature` | sampling | slight lever only; not main physics |
| `splat_sigma` | scar width | **geometry retune**, not √params — see splat lane |
| `splat_force_scale` / `splat_force_max` | scar **mass** | scale down on small; separate from force_cap |
| `min_splat_dist` / `splat_delta_threshold` | deposit spacing / high-δ | retune with residual δ scale |
| hierarchical width | fine/med/coarse | **`with_scale_ref(δ, threshold)`** — never absolute 20/30 on 4B |
| `targeted_splat_only` | high-signal only | **true** always |

### Splat gap (2026-07-11, 4B)

Force √-law alone left `splat_sigma=40` + absolute Coarse×4 on every high-δ token → late F_s climb / long-run Pain spam.  
**Field auto-σ (~7.6) ≠ residual scar width** — copying emb σ into `splat_sigma` made F_s≈0 (S3/S4 cold).  
Need **mid residual width** (~28–32 on 4B) + lower mass + relative hierarchy. Details: `research_logs/2026-07-11_splat-lane-4b.md`.

### F decay (mid-run)

| Knob | Role |
|------|------|
| `memory.online_decay_rate` | Per-token `α *= rate` (e.g. **0.975**). `1.0` = off. **Primary mid-run F_s fade.** |
| `physics.pain_decay_factor` | Pain lasts longer (slower fade). |
| `physics.splat_lambda_default` | Wall-clock / end-of-run evaporation λ on new scars. |
| `memory.decay_rate` | Fallback inside end-of-run `decay_step` only. |

Do **not** call wall-clock `decay_step` every token (was double-counting age). Use `decay_per_token`.  
B4b defaults: see `research_logs/2026-07-11_f-decay-quant-B4b.md`.

### Quantization (Q4 vs Q8)

- Same arch/size → **same** Algo force √-law (quant ≠ mass).  
- Q4 is noisier at the surface → slightly higher `temperature` / `rep_penalty`, softer scar α, online decay on — not higher `force_cap`.  
- Rebuild Diderot field per GGUF; field auto-σ can shift slightly.

### Worked swarm starting guesses (smoke-verify)

| Model | intensity | force_cap | splat_max | wake_max | goal_max | ramp | logit α |
|-------|-----------|-----------|-----------|----------|----------|------|---------|
| 1B standard | 0.58 | ~1.7 | ~16 | ~14 | ~23 | 18 / 0.10 | 0 |
| **3B golden std** | **1.00** | **3.0** | **28** | **25** | **40** | 15 / 0.15 | 0 |
| 3B instruct | 0.90 | ~2.7 | ~25 | ~22 | ~36 | 15 / 0.15 | 0 |
| **4B instruct (Gemma3)** | **~1.04** | **~3.1** | **~28** | **~25** | **~40** | **15 / 0.15** | **0** |
| 4B thinking | 0.46 | ~1.4 | ~13 | ~12 | ~18 | 15 / 0.12 | 0 |
| 12B instruct | 1.80 | ~3.5† | 28† | 25† | 40† | 12 / 0.18 | 0–0.05 |
| 27B instruct | 2.70 | 3.5† | 28† | 25† | 40† | 12 / 0.20 | 0–0.15 A/B |

† Swarm CEILING (Algo clamp spirit). **Never put 27B ceilings on 1B–4B without scaling.**

---

## Extra rules learned **after** Algo_WIP (still apply when scaling)

1. **Geometry ≠ size** — residual can sit off the emb shell (F_g≈0) on *any* size → nearest-emb wake, not only √params.  
2. **Targeted tokens** — don’t full-physics every token; high-δ / high-signal only.  
3. **Ramp early** — respect prefill J-space for first ~10–15 tokens.  
4. **Don’t “fix” with more force** — Niodoo latch + MountainCar: louder often **worse**.  
5. **Model type > size** — thinking/coding need whisper force.  
6. **Prompt surface ≠ residual** — Qwen35 addendum: control-surface **prefill** can dominate tag reliability; force may not help (and can hurt exact syntax).  
7. **Physics of Friendship / unison** — first proven on MountainCar; governor helps short runs, can **cap ceiling** long-run.  
8. **Loader arch** — `gemma3` / `llama` load today; **`gemma3n` / `gemma4` need new loaders** (file present ≠ runnable).

### Qwen35 caution (condensed from Algo addendum)

- √-law at 27B **saturates** Algo clamps (σ→0.20, θ→3).  
- That profile was telemetry-stable but did **not** prove “force improves reasoning.”  
- For control tags: **prefill at the control surface** often beats hot force.  
- Prefer a profile **below** hard caps when both work (headroom before chaos).

---

## Checklist when you drop a new model in `data/google/`

1. Confirm **arch** loads (`gemma3` / `llama` today; not `gemma3n`/`gemma4` without work).  
2. Note **params B** and **type** (instruct / thinking / …).  
3. Run  
   `python3 scripts/scale_physics_for_model.py --params X --type instruct`  
4. Paste suggested knobs into `config.toml` (or a size profile).  
5. Smoke **40–50 tokens**, one prompt, `--clear-memory`.  
6. Read: first 15 tokens (J-space), mean F_s / F_g / F_a, Pain count — **not** 200-token novels first.  
7. Only then re-run learning-lane A–E or longer.  
8. If residual dead / F_g≈0: fix **wake geometry**, don’t crank force.

---

## What we ship in-repo

| Need | Status |
|------|--------|
| Human share doc for size changes | **This file** |
| Script: params → suggested config | `scripts/scale_physics_for_model.py` |
| Hierarchical splat by δ | `Splat::with_scale` |
| Force ramp + targeted splat | `force_ramp_*`, `targeted_splat_only` |
| Auto-scale at GGUF load | Optional later (metadata → print knobs) |

Copies for sharing outside the repo:  
`Documents/MODEL_SIZE_PHYSICS_SCALING_swarm.md` (sync from this file).

---

## Provenance

- Golden size law + type mults + stability zone: Jason, Algo_WIPjuly experiments.  
- Swarm knob map + residual geometry + ramp/target + learning-lane B: hydrodynamic-swarm 2026-07.  
- Qwen35 control-surface notes: Algo_WIP 2026-04 addendum (caution, not swarm default).  
- Failures logged so size-changes don’t re-walk **1B gibberish** / **27B over-govern** / **MountainCar governor ceiling**.

**Authorship:** Jason (co-engineer / algo) · **Grok (xAI)** (swarm mapping, this document)
