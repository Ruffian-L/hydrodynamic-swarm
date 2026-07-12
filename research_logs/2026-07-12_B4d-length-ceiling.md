# B4d length ceiling (option 1) — 120 / 150 tok

**Date:** 2026-07-12  
**Authors:** Jason (steering / co-engineer) · Grok (xAI) (runs + analysis)  
**Config:** B4d (online decay 0.975, late F_a @48 → ×0.35, residual-mid splats)  
**Model:** gemma-3-4b-it-Q4_K_M  
**Prompt:** Physics of Friendship  
**Artifacts:** `logs/b4d_len_120.txt`, `logs/b4d_len_150.txt`

---

## Force health (still good at length)

### 120 tok

| window | δ | F_s | F_a |
|--------|--:|----:|----:|
| 0–29 | 70.7 | 0.40 | 28.8 |
| 30–47 | 107.4 | 1.89 | 29.2 |
| 48–79 | 108.1 | 2.96 | **20.4** |
| 80–119 | 108.0 | 1.32 | **11.2** |

### 150 tok

| window | δ | F_s | F_a | max F_s |
|--------|--:|----:|----:|--------:|
| 0–29 | 70.9 | 0.25 | 28.5 | 1.2 |
| 30–59 | 107.2 | 3.08 | 28.2 | 4.5 |
| 60–99 | 106.7 | 1.75 | **14.0** | 3.9 |
| 100–149 | 108.8 | 1.53 | **11.2** | 2.2 |

No F_s latch. F_a attenuates as designed. δ plateaus ~107–109 after early ramp.

**Conclusion:** length does **not** re-open force runaway. Ceiling is surface/capacity.

---

## Coherence map (4B + B4d)

| Tokens | Rough quality | Notes |
|--------|---------------|--------|
| **~0–40** | **Good** | On-topic, readable English |
| **~40–70** | Soft fray | Grammar slips, concept stacking; still on theme |
| **~70–100** | Clear fray | Compound words, broken phrasing (`physicsical`, `levelsofkines`) |
| **100–150** | Hard fray | Near salad (`wayingsideatanceorwithful`); force still calm |

**Practical ceiling for “useful paragraph” on this stack:** ~**50–70 tokens**.  
**Hard ceiling before gibberish:** ~**80–100**.  
Pushing 120–150 only extends the fray; physics stays behaved.

---

## Implication

Next is **quality tuning** (option 2) or **accept short budgets** for 4B demos — not more force knobs.  
Candidates: stop-early / quality exit, prompt framing for one tight paragraph, sample knobs (T / rep), softer late ocean deposits.

---

## Reproduce

```bash
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 120
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 150
```

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** length-ceiling ablation (120/150 tok), force health vs surface quality
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-12
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends. Force layer stable; 4B long-form ceiling is capacity/surface, not F_s latch.
---
