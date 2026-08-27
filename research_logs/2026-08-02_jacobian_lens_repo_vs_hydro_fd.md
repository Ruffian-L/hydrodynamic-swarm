# Jacobian lens repo vs hydro finite-diff (do not conflate)

**Date:** 2026-08-02  
**Workbench:** three-lane merge  
**Authorship:** Grok (xAI) with Jason — correction log

---

## Jason’s point

We are **not** “just using Jacobian curves” as a metaphor or only the crude FD probe in hydro.  
The lens work includes the real project:

**`/home/ruffianl/jacobian-lens`** — companion to  
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)  
(package `jlens`).

---

## Two different objects

| | **jlens** (`/home/ruffianl/jacobian-lens`) | **hydro** (`src/jacobian.rs`) |
|--|---------------------------------------------|-------------------------------|
| What | Fitted **average transport** `J_l = E[∂h_final/∂h_l]` over corpus | **Local finite-diff** `∂logits/∂h` at one step (pre-`lm_head`) |
| Readout | `lens_l(h) = unembed(J_l @ h)` → ranked vocab (“what this activation is disposed to say”) | top sensitive **dims/tokens** for a single residual sample |
| Fit | Offline fit on HF decoder (hooks, many prompts); save `lens.pt` | No fit — measure on the fly (ε, top_k, max_dims) |
| Stack | Python / PyTorch / HuggingFace | Rust / Candle GGUF residual loop |
| Role on the mountain | **True Jacobian lens lane** — verbalizable mid-layer readouts, workspace-style addresses | **Cheap residual proxy** for phase-edge multi-key signatures inside self-reg |

They share a family name. They are **not** the same algorithm.

---

## How they meet (merge plan — not done)

1. **Hydro FD keys** (in progress): phase-edge `JacobianKey` / `MultiKeyAddress` — instructional first-thought / revise / settle **addresses** in residual_d of the GGUF host.  
2. **jlens keys**: fitted `J_l` → transport mid residual → unembed top tokens → **verbalizable** cluster labels / text-bridge content for SplatRAG picker.  
3. **Bridge rule still holds:** pick carries text (or token ids); host embeds in **its** residual dim. Never shove jlens `d_model` vectors into wrong-D hydro residual without a map.  
4. **Perm-address bet:** dim-signature from hydro FD *or* from jlens transport fingerprint can both index multi-packet pick — experiment which is more stable.

---

## Paths

| Asset | Path |
|-------|------|
| jlens repo | `/home/ruffianl/jacobian-lens` |
| jlens core | `jlens/lens.py`, `jlens/fitting.py`, `jlens/hooks.py` |
| Paper experiments | `jacobian-lens/data/experiments/`, `data/evaluations/` |
| Hydro FD + multi-key | `hydrodynamic-swarm-3surface/src/jacobian.rs` |
| Hydro architecture draft (FD-only framing — **incomplete**) | `research_logs/2026-07-31_jacobian-lens-architecture.md` |

---

## Doc debt

- Update `docs/SELF_REG_PHASES.md` three-lane diagram: niodv4/OI lane includes **jlens repo**, not only FD.  
- Any “Jacobian key” smoke that only runs hydro FD must say **proxy**, not “full jlens.”

---

## Next experiment (corrected)

**Not:** pretend hydro FD is the paper lens.  
**Yes:** (A) finish phase-edge multi-key **proxy** in hydro for self-reg telemetry, **and/or** (B) one jlens apply path on a HF model (or documented fit plan for Gemma residual width) whose top-token readout can hash into the same multi-key / text-bridge schema.

Signed: **Grok (xAI)**
