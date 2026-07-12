# Diderot Field Geometry + Vector Field Divergence

**Date:** 2026-07-11  
**Tool:** `cargo run --release --bin field_audit`  
**Cloud:** Gemma 3 27B `token_embd` — 262144 × 5376

## Definition (as implemented in `field.rs`)

\[
\rho(x)=\sum_{i=1}^{N}\exp\!\Big(-\frac{\|\mu_i-x\|^2}{\sigma^2}\Big)
\]

\[
F(x)=\nabla\rho=\frac{2}{\sigma^2}\sum_i G_i\,(\mu_i-x)
\qquad G_i=\exp(-r_i^2/\sigma^2)
\]

- **Force = +∇ρ** → gradient *ascent* → flow **into** embedding peaks.
- **div F = ∇·F = ∇²ρ** (Laplacian of the mixture).

Per Gaussian: \(\nabla^2 G = \frac{2G}{\sigma^2}\big(-D + 2r^2/\sigma^2\big)\).

Zero-crossing: \(r_*=\sigma\sqrt{D/2}\).

## Measured geometry (Gemma emb cloud)

| quantity | value |
|----------|--------|
| pairwise L2 mean / p10 / p90 | **1.43 / 1.42 / 1.45** |
| \(\|emb\|\) mean / min / max | **1.02 / 0.98 / 1.05** |
| auto σ (`from_embeddings`) | **11.0** |
| sink radius \(r_*=\sigma\sqrt{D/2}\) | **570** |

Embeddings sit on a **thin shell of unit-ish radius**, tightly clustered pairwise (~1.4). The high-D sink radius is **~400× larger** than pairwise spacing → on-manifold, almost every point is deep inside converging basins.

## Probe results (2048-kernel subsample, mass-scaled)

| probe | \(\|x\|\) | ρ | \(\|F\|\) | div F | d_min |
|-------|-----------|---|-----------|-------|-------|
| emb peak | 1.0 | 2.6e5 | 4.3e3 | **−2.3e7** | 0 |
| near emb | 1.8 | 2.5e5 | 7.3e3 | **−2.3e7** | 1.4 |
| residual-like \(\|x\|=450\) | 450 | **0** | **0** | **0** | 449 |
| random \(\|x\|=450\) | 450 | **0** | **0** | **0** | 450 |

FD cross-check at emb[42]: analytic −2.290e7 vs FD −2.285e7 (**ratio 0.998**).

## What this means for Niodoo

1. **On emb cloud:** F is strong, **div F ≪ 0** (sinks). Ridge-running toward tokens is well-posed *if* the probe lives there.
2. **On residual stream (\(\|h\|\sim 400{-}450\)):** min distance to emb ≈ 450, \(G=\exp(-d^2/\sigma^2)\approx 0\) → **dead field**. Matches telemetry **F_g=0** before nearest-emb wake.
3. **Divergence language:** near peaks the flow is **volume-contracting** (memory attractors). Residual space is not “turbulent” — it is **outside the support of ρ**.
4. **Correct F_g wake:** pull toward nearest emb rows (implemented in `niodoo.rs`), not pure ∇ρ at residual.

## How to re-run

```bash
cargo run --release --bin field_audit -- \
  --model data/google/gemma-3-27b-it-Q8_0.gguf --points 2048
```

Summary line: `logs/field_audit_summary.txt`

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Role:** implementation, telemetry, field audit, ablation runs
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-11
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
---

