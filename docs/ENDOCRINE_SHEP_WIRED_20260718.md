# Endocrine wired for Shep (2026-07-18)

**Urgency context:** Jason portable lane — finish Shep’s endocrine restore so it is real code, not a dangling module.

## What was broken

- `endocrine.rs` restored from SanDisk Shep-loop but only `mod endocrine;` + `apply_monolith` dead fields.
- `eureka_impulse` / `viscosity` never entered `steer()`.
- No spawn / signal / bloom path in the generation loop.

## What works now

| Piece | Behavior |
|-------|----------|
| `create_endocrine_system()` | Spawns Function Gemma worker (stub) at startup |
| Default ON | Pass `--no-endocrine` to disable |
| Pain / high-δ | Rate-limited `ExecuteTool` signal → worker |
| Bloom | `apply_monolith` + `[BLOOM]` on stdout + `logs/live.txt` |
| `steer()` | Eureka boost + cooled viscosity while impulse decays |
| `tick_endocrine()` | Impulse decays each token (~0.92×) |

## Honest stubs / geometry (2026-07-25 update)

- `FunctionGemma::strict_execute` → still fake `[FACT #n]` text enzyme  
- **TinyEmbed removed** — blooms embed via **native** `model.token_embeddings()` mean-pool on main  
- Eureka window: impulse + optional pull toward that native (D,) vector  
- gemma-lab `universe_*.safetensors` = offline geometry lab (field load key `positions`); live endocrine uses the **running** model, not a second embedder  
- Real FunctionGemma 270M text enzyme = later

## Run

```bash
cd ~/projects/hydrodynamic-swarm
# source scripts/cuda_env.sh if needed
CUDARC_CUDA_VERSION=12000 cargo build --release --bin hydrodynamic-swarm
./target/release/hydrodynamic-swarm --tokens 40 --prompt "Explain the Physics of Friendship in one short paragraph."
# disable: --no-endocrine
```

Look for:

```text
Endocrine: ON (Shep restore — worker sleeping until signal)
[ENDOCRINE] signal sent at step …
[BLOOM] [FACT #…]
[NIODOO] Monolith applied …
```

## Attribution

Shep: endocrine design + mar22 source.  
Jason: restore urgency + portable cubby.  
Grok: wire into current main/niodoo steer path (2026-07-18).
