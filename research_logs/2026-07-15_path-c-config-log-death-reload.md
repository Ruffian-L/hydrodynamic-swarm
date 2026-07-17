# Path C — `--config`, memory logs, death→reload receipt

**Date:** 2026-07-15  
**Authors:** Jason (go C) · Grok (xAI) (implement + run)  
**Target:** memory means something (not answer polish)

---

## Shipped

1. **`--config PATH`** — TOML chosen before load (Run D can use `config.force_off.toml` without clobbering B4d-q permanently).
2. **SessionConfig memory flags** — `config_path`, `clear_memory`, `scars_loaded_safetensors`, `scars_imported_tct`, `scars_at_start`, `memory_loaded`.
3. **StepEntry.scars_active** — scars in memory each token.
4. **`--clear-memory`** also removes `data/splat_memory.tct` (+ json sidecar).
5. **Death→reload A/B** on 4B B4d-q friendship 65.

## Receipt

Full table: `logs/memory_coupling_main_20260715_234441/RECEIPT.md`

| | A (clear) | B (reload 19 scars) |
|--|----------:|--------------------:|
| early_Fs | 0.0 | 0.011 |
| late_Fs | 0.93 | 0.62 |
| scars_at_start | 0 | 19 |

**Persist: PASS. Early couple: FAIL/weak.** Next: measure distance-to-scar at step 0 and/or ramp-off smoke.

## Parallel note

Team Goal MEMORY_COUPLING still open on workbench — they re-run A–D independently; this main-lane receipt is a parallel data point, not their DONE.

---
**Authorship**
- **Author:** Grok (xAI)  
- **Operator:** Jason  
- **Note:** Failures stay on disk. Weak early F_s after reload is the useful finding.
---
