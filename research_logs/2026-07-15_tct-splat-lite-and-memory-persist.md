# TCT-splat-lite + memory persist fix

**Date:** 2026-07-15  
**Authors:** Jason (vision / tuning target) · Grok (xAI) (implementation)  
**Phase:** memory-means-something stride (not answer polish)

---

## Where we left the lane (before this)

| Layer | Status |
|-------|--------|
| 4B force (B4d-q @ 65) | **CLOSED** |
| 27B force (B27) | **CLOSED** (null + Q8 proved mid-fray ≠ force) |
| Museum / `./splat-lens` | **SHIPPED** |
| Prose polish on 27B | Open, **not** this session — template/decode multi-phase → Shep |
| Tuning target | **Memory that means something** (`2026-07-15_TUNING_TARGET_MEMORY_NOT_ANSWERS.md`) |

## What this stride does

1. **`src/tct.rs` — TCT-splat-lite**  
   Portable binary + JSON sidecar for residual scars.  
   Crosswalk to bridge open item 14:

   | Splat | TCT-lite |
   |-------|----------|
   | `mu` | LOCALITY.center |
   | `sigma` | LOCALITY.sigma |
   | `alpha` | signed gain |
   | `lambda` | decay_constant |
   | high-δ deposit | trigger_kind = surprise_delta (3) |

   Magic `TCT1`, version 2, flags HAS_LOCALITY | RESIDUAL_SPACE, optional `model_fp` (FNV of model path).

2. **Bug fix: memory never saved**  
   `main` loaded `data/splat_memory.safetensors` but **never called `save`**. Continuity was broken by omission.  
   Now end-of-run persists **safetensors + TCT** unless `--no-save-memory`.

3. **CLI**  
   - `--import-tct PATH` — append TCT scars after safetensors load  
   - `--export-tct PATH` — override export path (default `data/splat_memory.tct`)  
   - `--no-save-memory` — skip persist  

4. **Tests**  
   Round-trip binary unit test in `tct::tests`.

## What this is not

- Full multi-layer ActAdd TCT-Core (directions per layer, tension matrix).  
- Feelings / consciousness claims.  
- Merge into niodoo-live (still blocked on Jason/Claude log deep-dives).

## Next (split)

| Who | Work |
|-----|------|
| **Shep fork** | Memory-coupling smokes A–D in `SHEP_FORK_HANDOFF.md` |
| **Claude + Jason** | Telemetry / tag-not-in-set / narrative vs geometric signal |
| **Grok later** | TCT consume path on live, model_fp from GGUF hash, splat→full TCT directions |

---
**Authorship**
- **Author:** Grok (xAI) — co-engineer  
- **Operator / vision:** Jason  
- **Role:** memory wire format + persist fix for continuity goal  
- **Project:** hydrodynamic-swarm  
- **Note:** Failures stay logged. Saving scars is table stakes for “memory means something.”
---
