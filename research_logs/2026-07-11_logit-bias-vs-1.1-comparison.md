# Field logit bias vs Phase 1.1 — side-by-side

> **Author:** Grok (xAI) — co-engineering session with Jason / Shepard  
> **Date:** 2026-07-11  
> **Project:** hydrodynamic-swarm  
> **Purpose:** Keep the signal. Keep authorship. Log failures so the next person does not re-traverse them.

---

## Why this log exists

Every failed ablation is still a **map**.  
F_s=1872, F_g=0, dist_weighted /wsum bug, REFLEX threshold=2.0 false positive, late F_s→50 — those are not shame; they are **trail markers**.

Jason's rule: *every failure is another failure somebody else doesn't have to traverse.*

---

## Configs compared (same prompt, 200 tokens)

**Prompt:** `Explain the Physics of Friendship in one paragraph.`

| ID | Label | Wake | field_wake_max | field_logit_alpha | Log file |
|----|-------|------|----------------|-------------------|----------|
| A | Phase **1.1** | `wake` k=1 | 40 | **0** | `logs/2026-07-11_14-23-26_gemma3-27b_v3-forcecap3_T0_8_s40_a1_d30.jsonl` |
| B | 1.3 dist only | `dist_weighted` | 30 | **0** | `logs/2026-07-11_14-47-42_...jsonl` |
| C | **NOW** (mixed) | `dist_weighted` | 30 | **0.15** | `logs/2026-07-11_15-03-27_...jsonl` |
| D | Pure A/B wake+logit | `wake` k=1 | 40 | **0.15** | `logs/ab_D_wake40_logit015.jsonl` (= `15-13-43_...jsonl`) |
| E | Pure A/B wake only | `wake` k=1 | 40 | **0** | `logs/ab_E_wake40_logit0.jsonl` (= `15-10-01_...jsonl`) |

---

## Force balance (A vs B vs C)

| metric | 1.1 wake max40 (A) | 1.3 dist no logit (B) | dist + logit α0.15 (C) |
|--------|-------------------:|----------------------:|-----------------------:|
| mean F_g | **35.48** | 8.79 | 8.55 |
| max F_g | **40.00** | 10.63 | 10.84 |
| mean F_s | 31.40 | 39.09 | 35.49 |
| max F_s | 50.00 | 50.00 | 50.00 |
| mean F_a | 49.75 | 49.73 | 49.75 |
| mean δ | 79.67 | 78.93 | 81.29 |
| uniq ratio | 0.95 | 0.96 | 0.94 |
| F_s 50–100 | 48.70 | 47.82 | **37.55** |
| F_s 100–150 | 35.87 | 50.00 | 50.00 |
| late meta markers | 6 | 3 | 6 |
| REFLEX @100 | yes | yes | yes |

### What the numbers actually say (plain language)

1. **Logit bias barely moves residual forces** (F_g/F_s/F_a). That is correct — it tips **vocab scores**, not the residual field.
2. **F_g drop from ~35 → ~9** is from **dist_weighted**, not from logit α.
3. **Late collapse (F_s→50, REFLEX, meta text)** survived all three. Surface tip did **not** fix late thrash.
4. Early text on (C) was **slightly cleaner** on topic line than (A); still dies after ~80–100 tokens.

---

## Text samples (first ~120 tokens)

### A — 1.1 wake max40
> a friendships can be understood through several physical principles, though not as an emergent phenomenon arising from complex interactions – specifically related to physics's relationship between people's laws concerning attraction and repulsion—the concept of social physics terms: mutual energy exchange dynamic systems with fundamental way or more than just like gravity is about understanding by observation for these concepts are you describe how we understand...

Then: *please provide only… Please see often others…*

### C — dist_w + logit α=0.15
> a friendships can be understood through several physical principles, though not as a complex interplay of forces and energy exchange—specifically, friendship is fundamentally about mutual attraction – both physics's terms regarding interactions: emotional bonding based on reciprocal interaction's concept to understand with each other people'...

Then still frays into chemistry/meta/apology loops.

**Coherence (1–5):** early A≈3, C≈3+; late A≈1, C≈1. **Late: shared failure.**

---

## Pure A/B: same wake max40, only α_logit (E vs D)

Isolates surface tip. Same prompt, 200 tokens, `field_wake_mode=wake`, max=40.

| metric | **E α=0** | **D α=0.15** | note |
|--------|----------:|-------------:|------|
| mean F_g | **40.0** | **40.0** | both glued to wake max |
| mean F_s | 37.3 | **26.8** | logit run: lower mean F_s |
| max F_s | 50.0 | 50.0 | still hits ceiling late |
| F_s 0–50 | 11.9 | **5.9** | quieter early scars |
| F_s 50–100 | 43.2 | **21.2** | |
| F_s 100–150 | 45.2 | **34.3** | |
| mean δ | 81.5 | **78.8** | slightly calmer |
| mean F_a | 49.7 | 49.7 | unchanged |
| uniq | 0.94 | 0.93 | ~same |

### Early text (~100 tok)

**E (α=0):** mutual attraction/repulsion… “friendship is more than simply… people’s laws… energy” — on topic, some garble (`it'sically`).

**D (α=0.15):** mutual attraction… “emergent properties like gravity, quantum entanglement… social forces” — also on topic; then early meta (“you want do they”).

### Late (100–150)

Both fray. E: “I would consider… explaining her character…”. D: fragmented quotes / “objects”.

### Pure A/B verdict

| Question | Answer |
|----------|--------|
| Does α=0.15 change residual F_g? | **No** (both maxed at 40) |
| Does it change F_s? | **Yes — mean F_s lower** (37→27); scars quieter mid-run |
| Does it fix late collapse? | **No** |
| Worth logging? | **Yes** — isolates surface tip; failure is still a trail marker |

---

## Dead ends already walked (do not re-walk blind)

| Dead end | Signal | Fix / status |
|----------|--------|--------------|
| F_s ≈ 1872 | scar O(N) runaway | `1/√n` + scale/cap |
| F_g = 0 pure ∇ρ | residual off emb shell | nearest-emb wake |
| dist_weighted k=1 no-op | /wsum cancelled falloff | unit dir × falloff strength |
| REFLEX "collapse" | threshold=2.0 every 100 steps | almost always-on; not rare H1 |
| Ocean crystallizing junk | quality-blind deposits | still open |
| Logit bias alone | late still dies | expected; residual F_s still king late |

---

## Authorship

| | |
|--|--|
| **Author** | **Grok (xAI)** |
| **Operator / vision** | Jason (with Shepard) |
| **Date** | 2026-07-11 |
| **Repo** | hydrodynamic-swarm |

*If you quote this log, keep the author line. The math can be re-derived; the **sequence of failures** cannot without this trail.*

---

**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---
