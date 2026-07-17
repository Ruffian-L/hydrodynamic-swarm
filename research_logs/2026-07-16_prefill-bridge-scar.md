# Prefill-bridge scar — fix LOCALITY COLD for death→reload

**Date:** 2026-07-16  
**Authors:** Jason (go) · Grok (xAI) (implement + smoke)  
**Receipt:** `logs/memory_coupling_bridge_20260716_004806/RECEIPT.md`

## Problem

Trail scars load but sit ~180 L2 from next prefill residual → early F_s ≈ 0 (Gaussian dead).

## Fix

`SplatMemory::deposit_prefill_bridge` after final decay, before save: scar at **goal/prefill residual**, wide σ, replace prior bridges (flux mark).

Config (physics):

```toml
prefill_bridge_scar = true
prefill_bridge_sigma = 90.0
prefill_bridge_alpha = 0.75
prefill_bridge_lambda = 0.005
```

## Smoke

| | nearest_L2 | early_Fs | couples_hint |
|--|------------|----------|--------------|
| Before | ~180 | ~0.006 | false |
| After reload | **0.0** | **0.084** | **true** |

## Next (optional)

- Ramp-off + bridge for step-0 magnitude  
- Team re-runs A–D with bridge on  
- TCT mark bridge records in sidecar  

---
**Authorship:** Grok (xAI) + Jason · failures stay logged
---
