# Live apply wire-up — residual TCT → niodoo-live

**Date:** 2026-07-16  
**Lane:** hydro continuity → niodoo-live residual apply  
**Status:** landed in `niodoo-live` (loader + force + CLI + telemetry)

## What shipped

niodoo-live now consumes **TCT-splat-lite** (`TCT1` v3) residual scars and applies
Gaussian force on the last-token attention residual probe — same physics as
hydro `SplatMemory::query_force`, independent of the 64D correction-packet path.

| Piece | Location |
| --- | --- |
| Loader + force | `niodoo-live/niodoo/src/tct_splat_lite.rs` |
| CLI | `--tct-splat-path`, `--tct-splat-gain`, `--tct-splat-clamp`, `--tct-splat-bridge-only` |
| Engine field | `PrincipiaEngine.residual_tct` |
| Apply hook | `apply_forces` → `try_apply_residual_tct_force` (stacks with packets) |
| Load site | `simulation.rs` after VQ block; dim guard `[INV-5]` |
| Telemetry | `tct_force_norm`, `tct_potential`, `tct_nearest_dist`, `tct_n_active`, `tct_n_considered` |

## Force (matches hydro)

```
F = Σ_i α_i · exp(−‖μ_i − p‖² / σ_i²) · (μ_i − p)
if n_active > 1: F *= 1/√n_active
then: F *= gain; L2-clamp to --tct-splat-clamp (default 0.05)
```

`--tct-splat-bridge-only` keeps only `trigger_kind=5` prefill bridges.

## Dim guard (hard)

Current hydro export on this machine: **model_dim=2560** (Gemma path).  
Live Llama-3.1-8B: **4096**. Load refuses with clear message — no silent truncate.

To actually *feel* scars on live Llama, re-export TCT from a residual-dim-matched
run (or later: explicit projection — not this wire-up).

## Smoke

```bash
cd niodoo-live/niodoo
RUSTFLAGS="-C target-feature=+fp16" cargo test --lib tct_splat_lite \
  --no-default-features --features niodv4_bridge
# 6 passed (roundtrip, force direction, dim guard, hydro file load if present)

# Live (when you have a 4096-dim .tct):
./target/release/niodoo ... --tct-splat-path /path/to/splat_memory.tct \
  --tct-splat-bridge-only
# Expect startup: [TCT] Loaded N residual scars (... prefill_bridge) dim=4096 ...
# Expect telemetry.jsonl keys: tct_force_norm, tct_potential, tct_nearest_dist
```

## Non-claims

No consciousness. No feelings. Continuity = measurable residual geometry + force KPIs.
Packets (64D VQ) and residual TCT (full hidden) are **two parallel memory languages**;
this wire-up opens the residual language on live without touching packet semantics.
