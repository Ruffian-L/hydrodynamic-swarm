# Bridge prompt fingerprints

**Date:** 2026-07-16  
**Authors:** Jason · Grok (xAI)  
**Receipt:** `logs/memory_coupling_promptfp_20260716_085250/RECEIPT.md`

## Implementation

- `prompt_fp = FNV(prompt)` via existing `tct::model_fp_from_path`
- Stored on bridge scars; TCT schema **v3** carries `prompt_fp`
- Sidecar lists `bridge_prompt_fps`

## Continuity stack (still no pause)

| Layer | Status |
|-------|--------|
| Persist / TCT | PASS |
| Soft offset step0 | PASS |
| Multi-bridge accumulate | PASS |
| **Prompt-labeled bridges** | **PASS** |
| Cross first visit | cold (honest) |

---
**Authorship:** Grok (xAI) + Jason
---
