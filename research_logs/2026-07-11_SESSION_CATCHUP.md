# Session catch-up — read this when you’re back

**For:** Jason  
**From:** Grok (xAI) — co-engineer  
**When you left:** helping your brother buy a car  
**Tone:** plain English first; math only where it matters  

---

## Bottom line in one breath

We got **Gemma 3 27B** loading and steering for real.  
We fixed **scar force exploding** (F_s).  
We proved **field force was dead for geometry**, then **woke it**.  
We added a **surface tip** (logit bias) and **logged every dead end**.  
**Late-run text still falls apart** — next work is scars + ocean + fake REFLEX, not “is the field real?”

**You are so back.** The trail is signed so nobody steals the story or re-walks the failures blind.

---

## What “we are so back” means in code

| Piece | Status |
|--------|--------|
| `./run_swarm.sh` | Easy launcher — edit top of file or pass prompt/tokens |
| Gemma 3 27B | Default model under `data/google/` |
| Shared ocean | Deposits + refine (still quality-blind — fixed in this offline pass if noted below) |
| Semantic splats | Pleasure / pain from P(token), not from δ |
| Field wake | Nearest-emb pull; modes: off / wake / blend / dist_weighted |
| Field logit bias | `z += α · norm(E û_g)` — surface tip |
| Research logs | **Signed Grok (xAI) + you as co-engineer / operator** |

---

## Plain-language force story (the signal you already feel)

Think of three hands pushing the model’s “thought position”:

| Hand | Name | What it was doing |
|------|------|-------------------|
| **Memory scars** | F_s | Used to **crush** everything (1872). We dampened it. Late it still climbs to its cap (50). |
| **Goal** | F_a | Pull toward the **prompt’s** frozen meaning. Capped ~50. Always present. |
| **Field** | F_g | Wanted to pull toward **word-cloud memory**. Was **zero** because residual lives far from that cloud. Wake fixed liveness. |

**Late collapse** ≈ scars + ocean + a “REFLEX” that fires almost every 100 steps → text goes meta/garbage after ~80–100 tokens.

---

## Dead ends we already walked (do not re-walk)

1. **F_s runaway** → fixed with `1/√n` + scale/cap  
2. **F_g pure density = 0** → geometry (residual far from emb shell)  
3. **dist_weighted looked broken** → bug: weight canceled for k=1; **fixed**  
4. **Lower wake max only** → does not fix late collapse  
5. **Logit bias alone** → slightly quieter F_s; **does not** fix late collapse  
6. **REFLEX “collapse”** → threshold too loose; almost always fires at step 100  

Full tables:  
`research_logs/2026-07-11_field-wake-phase1-phase2-plan.md`  
`research_logs/2026-07-11_logit-bias-vs-1.1-comparison.md`  
Index: `research_logs/AUTHORSHIP.md`

---

## How to run when you’re back

```bash
cd ~/projects/hydrodynamic-swarm
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 80
```

Config knobs: `config.toml`  
- `field_wake_mode` = wake | dist_weighted | off  
- `field_logit_alpha` = 0 off, 0.15 tip  
- `splat_force_max` / `field_wake_max`  

---

## What Grok may do while you’re out

*(Updated at bottom of this file when work lands.)*

**Priority (no second ocean mind yet):**

1. Tighten REFLEX so it isn’t a fake emergency every 100 steps  
2. Quality-gate ocean deposits (don’t crystallize garbage)  
3. Soften late F_s (cap / decay)  
4. Leave this catch-up updated  

---

## Authorship (house rules)

- **You (Jason):** vision, operator, co-engineer  
- **Grok (xAI):** implementation, telemetry, ablations, these logs  
- **Shepard:** named in the trail with you  

Every Grok-written research entry ends with an **Authorship** block.  
Failures stay on disk **on purpose**.

---

## Offline work log (while you were out)

| Change | Files | Why / result |
|--------|--------|----------------|
| This catch-up doc | `research_logs/2026-07-11_SESSION_CATCHUP.md` | Re-entry without re-reading the whole chat |
| Authorship index | `research_logs/AUTHORSHIP.md` + all `2026-07-11_*.md` signed **Grok (xAI)** | Your rule: signal + credit don’t get lost |
| **REFLEX tightened** | `main.rs` | thr **2.0 → 1.12**, only if stress (F_s high). Smoke **100 tok: no fake REFLEX@100** |
| **Ocean quality-gated** | `main.rs` + `ocean.rs` | Pleasure deposits strong; Pain weak+noisy; **Skip = no deposit**. Slower decay, lower force_scale |
| **F_s ceiling** | `config.toml` | `splat_force_max` **50 → 28** (late thrash was F_s→50) |
| Smoke 100 tok | `logs/2026-07-11_15-22-39_...jsonl` | Early still on-topic; mid still frays some; **no REFLEX**; F_s@50≈23 under new cap |

### Smoke snapshot (offline)

```
Field wake: dist_weighted … max=30
Field logit bias: α=0.15
[50/100] F_g=8.0 F_s=23.3 F_a=50.0 ocean_n=10 noise=0.30 F_ocean=24.4
NO [REFLEX] at step 100
```

Still not “solved late English” — but we removed a **fake emergency** and stopped **blind ocean crystallization**. Next when you’re back: read this file, run `./run_swarm.sh "... " 100`, then decide if we chase F_s softer or quality harder.

---

**Talk soon. Hope the car hunt went well. We are so back.**

---
**Authorship**
- **Author:** Grok (xAI) — co-engineer with Jason / Shepard  
- **Project:** hydrodynamic-swarm  
- **Date written:** 2026-07-11  
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.  
---

### Learning lane (while grabbing model)
- Ran A–E on **gemma-3-27b-it-Q4_K_M** (E4B is gemma3n — not loadable yet).
- Results: `research_logs/2026-07-11_learning-lane-ablations.md` + `logs/learning_lane/`
- Best short-horizon feel: **B_ramp_targeted** (default config restored to that).
