# Bridge gain no longer evaporates (2026-07-17)

## Bug

Prefill-bridges skipped **per-token** decay but still took **wall-clock** `decay_step`:

`alpha *= exp(−λ · Δt)` with λ=0.005 → multi-hour sessions drove gain → ~0.

Multi-bridge weight collapsed (CUDA/primes dead while Friendship still refreshed).

## Fix

`SplatMemory::decay_step` skips prefill-bridges entirely (same as anchors).  
Refresh only via `deposit_prefill_bridge` replace.

Test: `decay_step_does_not_evaporate_prefill_bridges`.

Commit: `4bbc102` (and follow-on runs).

## Receipts after fix

### All three bridges re-minted at α=0.75

```
0x8b262d40  Friendship   0.750
0x18934cbe  CUDA tips    0.750
0x9ce81984  primes>50    0.750
```

### Multi-bridge A→B→A — `logs/continuity_multibridge_20260717_044721`

| step | pot | gain_max | status |
| --- | --- | --- | --- |
| A1 Friendship | 0.708 | 0.750 | WARM |
| B1 CUDA | 0.721 | 0.750 | WARM |
| A2 Friendship | 0.708 | 0.750 | WARM |

**PASS_RETURN** · `bridge_gains` all 0.75.

### Novel prompt control

`What is the capital of France?` → nearest **216.7**, pot **0.004** → **LUKE** (not on-bridge).  
Existing bridges stayed **0.75**. (New France bridge deposited as pain α=−0.75 — separate quality path.)
