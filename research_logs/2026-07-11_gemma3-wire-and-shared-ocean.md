# Gemma 3 27B live wire + Shared Ocean (Lane A→C)

**Date:** 2026-07-11  
**Scope:** hydrodynamic-swarm v0.2

## What landed

1. **Lane A — Gemma 3 loader wired into `main`**
   - GGUF `general.architecture` detection (`gemma3` → `gemma::ModelWeights`, else Llama).
   - Default model preference: `data/google/gemma-3-27b-it-Q8_0.gguf` (copied from SanDisk backup).
   - Gemma 3 IT chat template + EOS ids `{1, 106}`.
   - Explicit reject for Gemma 4 (`gemma4` / E2B) until a separate loader exists.

2. **Lane C — Shared Ocean foundation (`src/ocean.rs`)**
   - `FieldPacket` deposits from host residual every N tokens.
   - Diffusion-style `refine_step`: consensus blend + residual-noise decay (crystallization).
   - Fourth force in `NiodooEngine::steer`: ocean pull toward crystallized packets.
   - Pad/truncate projector for foreign dims (stub for learned Bridge matrices).

## Smoke run (2026-07-11)

```
cargo run --release -- \
  --model data/google/gemma-3-27b-it-Q8_0.gguf \
  --tokenizer data/google/tokenizer.json \
  --prompt "In one sentence, what is hydrodynamic memory?" \
  --tokens 24 --clear-memory
```

| Metric | Value |
|--------|--------|
| Load | Gemma3 heads=32 kv=16 blocks=62 hidden=5376 head_dim=128 |
| Field | 262144 × 5376, sigma≈0.72 |
| Goal norm | 423.52 |
| Tokens | 24 |
| Ocean | 5 packets, deposits=5, mean_noise≈0.56 |
| Log | `logs/2026-07-11_13-14-02_gemma3-27b_v3-forcecap8_T0_9_s35_a2_d100.jsonl` |

Decoded sample (steered): mentions fluid-flow memory / viscosity — coherent under physics.

## Not yet

- Load secondary Llama/Qwen minds into the same ocean (Goal 3 multi-mind).
- Learned projection matrices (384→D, 4096→D).
- Ocean force in JSONL step schema.
- Field sigma on Gemma is very tight (0.72); may need retune for 5376-d space.

## Model note

**E2B is Gemma 4** (effective ~2B edge). Wrong for this harness.  
**This run used Gemma 3 27B IT Q8** from the backup swarm tree — the original multi-mind kit model.

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---

