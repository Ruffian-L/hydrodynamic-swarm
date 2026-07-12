# Future direction

**Status:** Intent recorded 2026-07-12 — not a timeline commitment, not a rewrite in progress.  
**Authors:** Jason (project lead) · Grok (xAI) (co-engineer / drafting)

---

## Why say this now

The current standalone Rust harness is the **validation vehicle** for residual physics steering (splats, forces, Diderot field, shared ocean, quality gates, SplatLens museum).

We are stating the long-term shape **before** any port lands so it is clearly planned, not an after-the-fact hop onto someone else’s stack. Niodoo’s history is a cautionary tale: interest without a runnable, maintainable path for others — and then loss of the codebase. We do not want to repeat that.

---

## Long-term shape

1. **Keep** the Niodoo physics layer as the reusable core: splat memory, force composition, field wake, motif/scar policy, steering loop, telemetry, museum/viz contracts.
2. **Move inference** onto a more mature Rust GGUF/runtime backend — **starting with [mistral.rs](https://github.com/EricLBuehler/mistral.rs)** (or a comparable crate if the integration fit is better), used as dependency or sidecar rather than maintaining a full custom stack forever.
3. **Expose** that combination so others can run and extend the steering without owning our entire loader history.

Short form for citations / README:

> The current standalone Rust implementation develops and validates the core physics-steering and memory architecture. Once stable, the plan is to host the Niodoo physics layer on a mature inference backend (starting with mistral.rs) as a sidecar/library, so the work is more accessible and maintainable for others.

---

## Staged path (indicative, not a calendar)

| Stage | Focus |
|-------|--------|
| **Near term** | Finish 4B (B4d-q) + 27B (B27) validation on the **current** custom loader. Museum checkpoints when clean. |
| **Medium term** | Map mistral.rs (or peer) integration points: where residual/hidden is available, hooks for per-token force injection, GGUF load path we can stop owning. |
| **Longer term** | Physics as library/sidecar; demo path still `./splat-lens` style (watch without GPU when possible). |

We are **not** freezing physics R&D to wait on a port. We are **not** abandoning the interesting layer for “just another chat UI.”

---

## What this is *not*

- Not a claim that the port is done or scheduled for a fixed date.  
- Not abandonment of Gemma/Llama GGUF work already validated.  
- Not a license change for model weights (Meta / Google terms still apply via [`NOTICE`](../NOTICE)).  
- Not “we rewrote llama.cpp.” See attribution below.

---

## Attribution: GGUF loading and llama.cpp

This repo uses a **custom Rust GGUF / model path** (Candle + our `llama.rs` / Gemma loaders). It was **inspired by** and developed with **reference to** [llama.cpp](https://github.com/ggerganov/llama.cpp)’s open GGUF parsing and ecosystem — especially early loader work — **not** a wholesale copy of large llama.cpp sources into this tree.

**Safe public line:**

> GGUF loading and model metadata handling were developed with reference to llama.cpp’s open-source implementation.

Candle / tokenizers / hf-hub remain separately licensed (see [`NOTICE`](../NOTICE)). Model weights are separate again.

---

## Related live docs

- Research narrative: `research_logs/2026-07-12_future-direction-mistral-sidecar.md`  
- Physics size scaling: `docs/MODEL_SIZE_PHYSICS_SCALING.md`  
- Demo door: `./splat-lens` · `research_logs/2026-07-12_splat-lens-museum-structure.md`  
- Third-party / model notices: [`NOTICE`](../NOTICE)
