# Field logit bias (surface bridge)

**Date:** 2026-07-11

## Formula (wired)

\[
\hat u_g = F_g / \|F_g\|,\quad
s = E\,\hat u_g,\quad
z \leftarrow z + \alpha\cdot\frac{s}{\max|s|}
\]

- Applied **after** `project_to_logits`, **before** rep-penalty + softmax.
- \(E\) = `token_embd` (same D as residual for Gemma 3 27B — no \(W\)).
- Config: `field_logit_alpha` (default **0.15**, `0` = off).

## Code

| piece | location |
|-------|----------|
| unit \(F_g\) | `SteerResult.field_dir` in `niodoo.rs` |
| \(z += \alpha \hat s\) | `main.rs` generation loop |
| knob | `config.toml` → `field_logit_alpha` |

## Smoke (80 tok)

Banner: `Field logit bias: α=0.15`  
`[50/80] F_g=8.6 F_s=35.0 F_a=50.0` — residual forces unchanged; surface tip on top.

Log: `logs/2026-07-11_14-57-01_gemma3-27b_v3-forcecap3_T0_8_s40_a1_d30.jsonl`

## Note

Residual physics remains primary. This only *tips* vocab scores toward tokens aligned with the field direction. Cost: one `(V,D)×(D,1)` matmul per token.
---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---

