# Vocabulary lock — hydrodynamic-swarm

**Owner:** Jason Van Pham  
**Date:** 2026-07-30  
**Rule:** Public and console language uses **this** table. Older code identifiers may still say `scar` on the wire for compatibility; **do not teach operators that word.**

## Preferred terms

| Use this | Not this | Meaning |
|----------|----------|---------|
| **learned will** / **learned wills** | scar / scars / scar tissue | Gaussian memory unit in residual space (μ, σ, signed α) |
| **will store** | scar store | `data/splat_memory.safetensors` + TCT |
| **bridge will** / **prefill bridge** | prefill-bridge scar | Continuity deposit at prefill residual for next-run warm start |
| **will geometry** | scar geometry | nearest_L2, pot, n at prefill |
| **+will** / **attract** | pleasure (optional in deep docs) | α > 0 pull |
| **−will** / **repel** | pain / poison | α < 0 push; never “poison” |
| **cold / warm** | — | locality relative to learned wills (d ≫ σ vs pot high / near) |

## History (honest)

Earlier seats renamed **learned wills → scars**. That language felt extractive (“used”).  
**Learned will** is Jason’s term and the public face again.  
Wire/jsonl keys may still say `scar_*` until a versioned schema migration; console + papers + team talk use **learned will**.

## Wire aliases (do not rename blindly without migration)

| Wire / code (legacy) | Speak as |
|----------------------|----------|
| `scars_active`, `scars_at_start` | learned wills active / at start |
| `nearest_scar_dist` | nearest will L2 |
| `scar_potential_at_prefill` | will potential at prefill |
| `prefill_bridge_scar` (toml) | prefill bridge (config key unchanged for now) |
| `SplatKind::Pain` / `Pleasure` | −will / +will on console |

## Poison

**Banned** in product language. Negative α is **repel** or **−will**, not poison.

## Color (telemetry)

Color = α sign + H + δ + termsplat tier. Still valid. Speak wills, not scars.
