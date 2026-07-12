# All four next directions — 4B demo lock · length push · 27B port · prompt battery

**Date:** 2026-07-12  
**Authors:** Jason (steering / co-engineer) · Grok (xAI) (runs + analysis)  
**Phase:** post-physics refinement — use the stable base

---

## 1. Accept 65 tok as practical ceiling (demo default)

**Status: LOCKED as default**

| Item | Path / value |
|------|----------------|
| Config | `config.toml` = **B4d-q** |
| Launcher | `./run_swarm.sh` → 65 tok, 4B Q4 |
| Physics | B4d frozen (force hunt closed) |

Use for real short work / demos. Do not expect clean long-form on 4B.

```bash
./run_swarm.sh
```

---

## 2. Push length 80–90 (quality focus)

| Run | Tokens | Notes |
|-----|--------|--------|
| B4d-q base | 85 | Soft mid fray after ~50–60; forces calm (F_s~2.5 @50) |
| + rep 1.40 / T 0.85 | 90 | Still soft late; **does not unlock** long clean form |

**Verdict:** Sample knobs alone do **not** move the 4B capacity wall. Stay at **65** for reliable clean paragraphs. Logs: `logs/b4d_q_push_85.txt`, `logs/b4d_q_push_90_rep140.txt`.

---

## 3. Lessons → 27B (first port smoke)

**Profile:** `config.27b.toml`  
**Model:** `data/google/gemma-3-27b-it-Q4_K_M.gguf`  
**Run:** 80 tok, friendship short paragraph  
**Log:** `logs/port_27b_b4d_lessons_80.txt`

Ported: √-law-ish caps, residual splat σ=40 / mass 0.22/28, online_decay 0.98, late F_a @60 → ×0.4, ramp + targeted.

| window | δ | F_s | F_a |
|--------|--:|----:|----:|
| 0–29 | 64.5 | 0.00 | 38.7 |
| 30–59 | 77.9 | 2.60 | 40.0 (cap) |
| 60–79 | 80.4 | **8.90** | 34.3 |

- Forces **not exploding** (no old F_s=28 latch story).  
- F_s **climbs late** more than 4B at same length — retune mass/decay for 27B residual next.  
- F_a hits **40 ceiling** mid-run — late attenuate starts @60; may start earlier (e.g. 45) on 27B.  
- Prose: on-topic but imperfect open (template quirk); better headroom than 4B mashed salad at 80.

```bash
cp config.27b.toml config.toml   # or keep as side profile
# then:
./target/release/hydrodynamic-swarm \
  --model data/google/gemma-3-27b-it-Q4_K_M.gguf \
  --tokenizer data/google/tokenizer.json \
  --prompt "Explain the Physics of Friendship in one short paragraph." \
  --tokens 80 --clear-memory
# restore 4B default after:
# restore config.toml B4d-q (done after this session write)
```

---

## 4. Multi-prompt battery @ 65 tok (4B B4d-q)

**Harness:** `./prompt_battery_4b.sh` · `logs/prompt_battery_4b/`

| id | late F_s | late F_a | max F_s | Quality (eyeball) |
|----|---------:|---------:|--------:|-------------------|
| **friend** | 1.68 | 24.0 | 3.0 | **Best** — usable short paragraph |
| **creative** | 2.02 | 16.8 | 3.3 | Soft poetic fray mid-late |
| **noir** | 1.39 | 12.1 | 3.5 | Collapses early (meta header + mash) |
| **tech** | 2.63 | 8.9 | 3.3 | **Worst** — near gibberish |

**Forces calm on all four.** Usable length is **prompt-sensitive**: plain friendship works; technical / heavy style framing does not hold the same 65-tok “clean zone.” Prefer simple short-paragraph prompts for demos.

---

## Operating summary

| Goal | Recommendation |
|------|----------------|
| 4B demos / real short work | **B4d-q @ 65**, simple prompts |
| 4B long form | Don’t fight capacity; optional quality-gate later |
| 27B | Use `config.27b.toml`; next: earlier late F_a, soft F_s mass if late climb continues |
| Multi-style | Re-smoke battery after any sample change; don’t assume friend-prompt = all prompts |

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** all-four next directions (demo lock, length push, 27B port, prompt battery)
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-12
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends. Physics closed on 4B; prompt/style and size are the remaining axes.
---
