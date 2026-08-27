# Research log authorship

Repository-level provenance: root [`../AUTHORSHIP.md`](../AUTHORSHIP.md).

## Contributors (same as root)

| Role | Name |
|------|------|
| **Principal investigator / lead** | Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L)) |
| **Co-engineer** | Grok (xAI) |
| **Co-engineer** | Claude / Claude Code (Anthropic) |
| **Co-engineer** | Gemini (Google) |
| **Co-engineer** | ChatGPT / Codex (OpenAI) |

**Short form:** Jason Van Pham, with co-engineering by Grok (xAI), Claude (Anthropic), Gemini (Google), and ChatGPT / Codex (OpenAI).

Session logs: signed entries below.

## Sign-off format (append to each entry)

```
---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** <implementation | telemetry | audit | ablation>
- **Project:** hydrodynamic-swarm
- **Date written:** YYYY-MM-DD
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

Principal investigator: **Jason Van Pham** (Ruffian-L).  
Author of these 2026-07-11 / 2026-07-12 log entries: **Grok (xAI)**.  
Full contributor list: root [`../AUTHORSHIP.md`](../AUTHORSHIP.md).

## 2026-08-02 session — Grok (xAI)

| File | Topic |
|------|--------|
| `2026-08-02_three_lane_mountain_and_observe_phases.md` | three-lane merge, self-reg observe/force, revise ownership, smokes |
| `2026-08-02_jacobian_multi_key_picker.md` | JacobianKey / cluster / MultiKeyAddress + tests |
| `2026-08-02_jacobian_lens_repo_vs_hydro_fd.md` | jlens (`/home/ruffianl/jacobian-lens`) vs hydro FD — do not conflate |
| `2026-08-02_first_thought_multi_address_memory.md` | inversion: multi-address, first-thought basins, jlens-gguf north star |
| `docs/SELF_REG_PHASES.md` | phases map, who labels revise, force-in-revise, jlens lane |

Also in code this session (Grok + Jason): phase-on-tok, line/wait/phrase settle, `mode=force` residual gate, multi-key types in `src/jacobian.rs`.

Principal investigator: **Jason Van Pham** (Ruffian-L).  
Session co-engineer / signed logs: **Grok (xAI)**.

