# Future direction — physics layer + mature inference backend

**Date:** 2026-07-12  
**Authors:** Jason (project lead / co-engineer) · Grok (xAI) (co-engineer, drafting)  
**Status:** Direction recorded on purpose **before** any port work starts

---

## Decision

**Official long-term intent:**

- Validate and stabilize Niodoo-style residual physics steering on the **current** hydrodynamic-swarm harness (4B B4d-q, 27B B27, SplatLens museum).
- Then host that physics layer on a **mature Rust inference stack**, starting with **mistral.rs** (or similar), as **sidecar / library** rather than forever maintaining a full custom GGUF+forward path.

Saying this **now** avoids looking like we jumped on another project after the fact. Accessibility and “others can actually run it” are first-class goals — Niodoo’s lost/unrunnable history is the anti-pattern.

Canonical short writeup for README later: [`docs/FUTURE_DIRECTION.md`](../docs/FUTURE_DIRECTION.md).

---

## Why not stop 27B tuning

Near-term work stays on the custom stack:

1. Finish force-healthy 27B (B27) and surface quality (template/prompt/ocean).  
2. Keep 4B as short-demo / museum path.  
3. Only then spend serious cycles on integration architecture.

Physics R&D and “future backend” are sequential for **shipping a port**, parallel for **planning**.

---

## Integration sketch (non-binding)

Builder / future sessions should look at:

- Where mistral.rs (or peer) exposes **last-layer / residual hidden** during decode.  
- Cleanest **per-token hook** to inject F_g + F_s + F_a + ocean without forking the entire sampler forever.  
- How GGUF load / chat templates become **their** problem so ours stays physics + memory + viz.  
- Keep **museum / `.viz.json`** contracts so `./splat-lens` still works for watchers without GPU.

---

## llama.cpp attribution (related honesty)

Custom Rust GGUF path; structure and early metadata handling **referenced** llama.cpp’s open implementation. Not a bulk code paste.

Safe line (also in `docs/FUTURE_DIRECTION.md` + `NOTICE`):

> GGUF loading and model metadata handling were developed with reference to llama.cpp’s open-source implementation.

---

## Framing choices (locked for this note)

| Question | Choice |
|----------|--------|
| How hard lock mistral.rs? | **Starting point**, not exclusive — “mistral.rs or comparable mature Rust inference crate.” |
| Where does the note live? | **`docs/FUTURE_DIRECTION.md`** + this research log; README pointer can follow. |
| llama.cpp same note? | **Yes** — same honesty bag; NOTICE gets a short subsection too. |

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard  
- **Operator / vision:** Jason (project lead)  
- **Role:** Record long-term mistral.rs-sidecar direction + llama.cpp attribution  
- **Project:** hydrodynamic-swarm  
- **Date written:** 2026-07-12  
- **Note:** Failures and dead ends stay logged. Direction stated early so narrative and credit stay clean for others building on this work.
---
