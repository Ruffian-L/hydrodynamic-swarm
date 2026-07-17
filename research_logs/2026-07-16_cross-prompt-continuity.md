# Cross-prompt continuity — bridge is basin-local

**Date:** 2026-07-16  
**Authors:** Jason (watched museum · continue) · Grok (xAI)  
**Receipt:** `logs/memory_coupling_cross_20260716_071348/RECEIPT.md`

## Finding

| Arm | pot | nearest | early F_s |
|-----|-----|---------|-----------|
| Same prompt reload | 0.66 | 31.5 (0.35σ) | 0.19 |
| TCT import same prompt | 0.66 | 31.5 | 0.20 |
| **Cross prompt** (CUDA tips on friendship store) | **0.02** | **172** | **0.03** |

Prefill-bridge + soft offset **works for same-prompt death→reload** and **TCT round-trip**.  
It does **not** warm an unrelated prompt’s prefill basin.

## Public claim boundary

Safe: “Memory survives process death and steers the **next run of a similar start**.”  
Not safe: “Any scar store pulls every new prompt.”

## Continuity stack status

| Layer | Status |
|-------|--------|
| Persist | PASS |
| LOCALITY COLD diagnosed | PASS |
| Prefill-bridge | PASS |
| Soft offset step0 F_s | PASS |
| TCT import | PASS |
| Cross-prompt | **FAIL (honest)** — open for multi-bridge later |

---
**Authorship:** Grok (xAI) + Jason · failures logged on purpose
---
