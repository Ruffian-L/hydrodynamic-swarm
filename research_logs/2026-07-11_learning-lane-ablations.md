# Learning Lane Ablations — ramp + targeted splats (Q4)

**Date:** 2026-07-11  
**Author:** Grok (xAI) — co-engineer with Jason / Shepard  
**Directive:** smaller/faster model, short smokes, original Niodoo ramp + high-signal only, focused A–E  

---

## Model note

| File | Used? | Why |
|------|-------|-----|
| `data/google/gemma-3-27b-it-Q4_K_M.gguf` | **Yes** | `gemma3` arch, ~16G, works with loader |
| `data/google/google_gemma-3n-E4B-it-Q5_K_M.gguf` | **Not yet** | `gemma3n` (AltUp/Laurel/PLE) — needs new backend |
| Q8 27B | Skip for this round | slower |

---

## Code landed for this lane

| Feature | Config / code |
|---------|----------------|
| Force ramp | `force_ramp_tokens`, `force_ramp_start` → scale total force early |
| Targeted splats | `targeted_splat_only` → high-δ **or** pain **or** strong pleasure |
| Prefill J-space dream | `prefill_micro_dream` |
| Pain recovery ocean | `pain_recovery_ocean` |
| Default model path | prefers Q4 |
| Script | `./learning_lane_ablation.sh` |

---

## Variants (50 tokens, same prompt)

**Prompt:** Explain the Physics of Friendship in one paragraph.

| ID | Idea |
|----|------|
| **A** | Baseline: no ramp, not targeted-only |
| **B** | Ramp 12 tok + targeted splats |
| **C** | Lower caps (F_s 18, wake 15, goal 30, force_cap 2.5) |
| **D** | Stronger J-space: ramp start 0.10, 15 tok, prefill micro-dream, weaker goal scale |
| **E** | Pain recovery ocean packets |

---

## Results (clean)

| variant | mean δ | mean F_g | mean F_s | mean F_a | max F_s | uniq | P / pain |
|---------|-------:|---------:|---------:|---------:|--------:|-----:|----------|
| A_baseline | 67.8 | 4.72 | 5.45 | 39.2 | 17.9 | 0.98 | 6 / 0 |
| **B_ramp_targeted** | **63.9** | 5.14 | 6.64 | 39.2 | 28.0 | **1.00** | 5 / 1 |
| C_low_gov | 64.6 | **3.03** | 7.58 | **29.4** | 18.0 | 0.98 | 5 / 0 |
| D_jspace | 66.2 | 5.12 | 6.03 | 35.7 | 26.7 | 0.98 | 6 / 0 |
| E_recovery | 64.5 | 4.98 | **4.60** | 39.2 | 24.8 | 0.98 | 6 / 0 |

Artifacts: `logs/learning_lane/{A..E}_*.jsonl`, `summary_clean.tsv`

### Early text (first ~100 chars)

| variant | snip |
|---------|------|
| A | …lens of physics as **emergent phenomena arising** |
| **B** | **aren't just about emotional connections, but also have physical basis** |
| C | aren't just about emotional connections… physical basis roo… |
| D | lens of physics as **arising from shared energy** |
| E | aren't just about emotional connections… physical basis |

Q4 **20-token smoke** (pre-ablation) was already clean:  
> "friendships aren't just about emotional connections, but also have a physical basis too! While"

---

## Read (honest)

1. **At 50 tokens, nothing collapses hard** — shorter horizon is the right lab scale (matches your “smoke first” call).
2. **B (ramp + targeted)** has lowest mean δ, uniq=1.00, natural opening — **best default candidate**.
3. **C (low gov)** softest F_g/F_a — good for “less over-governing” feel; text still OK early.
4. **D (J-space)** nice semantic opening (“shared energy”); forces similar to B.
5. **E** quietest mean F_s — recovery path didn’t hurt; need longer runs to see Pain recovery fire.
6. **F_g stays ~3–5** under dist_weighted (expected); F_a still the biggest hand (~30–40).

**Not proven yet:** late collapse at 100–200 (that was the old failure). Next session: take **B or D** defaults → 100 tok smoke → only then longer.

---

## When smaller `gemma3` lands

Point `MODEL=` in `run_swarm.sh` / ablation script at it. Re-run `./learning_lane_ablation.sh`.  
**Do not** use E4B until `gemma3n` loader exists (clear bail message if tried).

---

## Authorship

- **Author:** Grok (xAI) — co-engineer with Jason / Shepard  
- **Role:** implement ramp/target, run A–E, analyze  
- **Note:** Failures and short-horizon wins both logged so the next person doesn’t re-walk blind.

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---
