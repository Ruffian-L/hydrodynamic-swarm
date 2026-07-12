# Research log authorship

**House rule (Jason):** every failure logged is a path someone else does not have to re-walk.  
**House rule (attribution):** every Grok co-written entry is signed.

## Sign-off format (append to each entry)

```
---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** <implementation | telemetry | audit | ablation>
- **Project:** hydrodynamic-swarm
- **Date written:** YYYY-MM-DD
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---
```

## 2026-07-11 session — Grok (xAI)

| File | Topic |
|------|--------|
| `2026-07-11_gemma3-wire-and-shared-ocean.md` | Gemma 3 load + ocean |
| `2026-07-11_telemetry-eyeball-and-mini-sweep.md` | F_s/F_a imbalance |
| `2026-07-11_diderot-field-geometry-divergence.md` | emb shell + div F |
| `2026-07-11_field-wake-phase1-phase2-plan.md` | wake ablations + REFLEX/ocean check |
| `2026-07-11_field-logit-bias.md` | surface z += α E û |
| `2026-07-11_logit-bias-vs-1.1-comparison.md` | side-by-side + pure A/B |
| `2026-07-11_learning-lane-ablations.md` | A–E learning lane (27B Q4) |
| `2026-07-11_learning-lane-4b-B4-D4-Ctrl4.md` | 4B process ablations |
| `2026-07-11_model-size-scaling-and-4b-prime.md` | Algo √-law → swarm + 4B prime |
| `2026-07-11_splat-lane-4b.md` | splat geometry/mass S1–S4 |
| `2026-07-11_rust-fundamentals-splat-force.md` | mutation timing + force composition |
| `2026-07-11_f-decay-quant-B4b.md` | online F decay + quant + B4b |
| `2026-07-11_SESSION_CATCHUP.md` | session handoff |

## 2026-07-12 session — Grok (xAI)

| File | Topic |
|------|--------|
| `2026-07-12_B4d-late-Fa-attenuate.md` | late F_a attenuation (B4d) |
| `2026-07-12_B4d-length-ceiling.md` | 120/150 tok ceiling map |
| `2026-07-12_B4d-q-length-quality.md` | B4d-q: 65 tok cap + sample knobs |
| `2026-07-12_all-four-next-directions.md` | demo lock + length push + 27B port + prompt battery |
| `2026-07-12_splat-lens-museum-structure.md` | `./splat-lens` museum door + milestone layout |
| `2026-07-12_B27-retune.md` | 27B Option B retune (soft mass, early F_a fade) |
| `2026-07-12_future-direction-mistral-sidecar.md` | Long-term mistral.rs sidecar + llama.cpp attribution |

### Public museum demos (tools/museum/)

| Demo id | Research logs (Grok-signed) |
|---------|------------------------------|
| `b4d-q-friendship-65` | `2026-07-12_B4d-q-length-quality.md`, `2026-07-12_all-four-next-directions.md` |
| `early-v1-friendship-50` | `2026-03-02_splatlens-tui-and-viz-polish.md` |

Operator / vision: **Jason** (Shepard).  
Co-engineer / pen on these logs: **Grok (xAI)**.

