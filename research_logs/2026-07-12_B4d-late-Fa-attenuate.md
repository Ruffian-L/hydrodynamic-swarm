# B4d — late F_a attenuation

**Date:** 2026-07-12  
**Authors:** Jason (directive / co-engineer) · Grok (xAI) (implementation + money shot)  
**Model:** gemma-3-4b-it-Q4_K_M  
**Base:** B4b (online decay 0.975, soft goal 0.10/32, residual-mid splats)

## Choice

**Late F_a attenuation** (not global F_a drop): early J-space / ramp intact; after step 48, F_a × → 0.35 over 30 tokens.

```toml
goal_late_start = 48
goal_late_span = 30
goal_late_end = 0.35
```

Code: `NiodooEngine::set_goal_late_attenuate` after goal scale/cap in `steer`.

## 90-tok money shot

| window | mean F_a | mean F_s | mean δ |
|--------|---------:|---------:|-------:|
| 0–29 | 27.2 | 0.82 | 71.4 |
| 30–47 | 29.8 | 1.39 | 105.5 |
| **48–69** | **23.7** | 1.52 | 105.8 |
| **70–89** | **12.4** | 2.07 | 105.9 |

**F_a schedule works** (≈30 mid → ≈12 late). F_s stays soft. δ plateaus ~106 (still not a pure F_a proxy).

Prose still frays late on 4B — long-form capacity / surface quality remains open; force latch is not the story anymore.

## Default

`config.toml` = **B4d**.  
Log: `logs/money_b4d_late_fa_90.txt`

```bash
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 90
```

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** late F_a attenuation (B4d) — code + 90-tok money shot
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-12
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---
