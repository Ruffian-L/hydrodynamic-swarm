# Gemma 4 loader — attribution + hard path

**Date:** 2026-07-28  
**Tree:** `/media/ruffianl/ghost_team/projects/hydrodynamic-swarm`  
**Authors:** Jason · Grok (xAI) co-engineer this session  

## Rule (Jason)

**Credit everyone who helped. Do not over-credit.**

- **Llama + Gemma 3 loaders:** earned here (Jason + Grok + Claude + Gemini and
  lineage) by tuning and trial-and-error. **Not** copied from foreign C++.
- **Gemma 4 — hard path:** Derive from **our** `gemma.rs`, real GGUF
  metadata/tensors, local `data/google/gemma4_assets/`, and garbage smokes.
  Jason deleted the local llama.cpp tree so we **stop peeking** maps. Early
  notes that listed a map path were the easy route — abandoned.
- Never write comments that sound like “we took large chunks of their code”
  unless that actually happened (it did not). GGUF format history may still
  name llama.cpp; that is not authorship of our forward.

## What hydro uses today

| Arch | Loader | Status |
|------|--------|--------|
| Llama 3.1 | `src/llama.rs` | Wired |
| Gemma 3 | `src/gemma.rs` | Wired |
| Gemma 4 | `src/gemma4.rs` | Wired; English one-shot works; multi-turn still frays — see `2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md` |

## Evidence sources for G4 deltas (hard path)

| Source | What we use |
|--------|-------------|
| `src/gemma.rs` | Residual/attn/FFN skeleton, QK norms, hydro hooks |
| GGUF `gemma4.*` keys + tensor names/shapes | block count, dual head dims, SWA pattern, softcap, optional `rope_freqs`, missing `attn_v` |
| `data/google/gemma4_assets/config.json` | partial_rotary_factor, proportional vs default rope, gelu_pytorch_tanh, k_eq_v, dual head dims |
| Smoke jsonl / readable.txt | “still garbage” receipt — fix until short English |

### Observed G4 needs vs Gemma 3 (from those sources)

- Sliding-window + full-attn layers (dual head dims / dual rope bases)
- Full-attn: proportional + **partial** RoPE
- Optional V ← K when `attn_v` absent
- FFN GELU (tanh form), not SiLU
- Attn scale often 1.0; final logit softcap
- Hydro must keep: `forward_with_hidden` + `project_to_logits` + `token_embeddings`

## Models for personality A/B (after loader works)

- unsloth / bartowski 31B Q4_K_M pair (on disk as available)
- 12B IT GGUF for cheap forward smokes first

## License reminder

- **Our Rust port**: MIT-0 (this repo).  
- **Gemma 4 weights**: Apache-2.0 (Google) + quantizer credit (Unsloth / bartowski).  
- **Gemma 3** remains under Gemma Terms — do not mix notices.  
- **GGUF format ecosystem** (incl. llama.cpp historically): format nod only.
