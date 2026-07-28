# Configs layout

Runtime always loads a single path via `--config`.  
**Active machine config** stays at repo root: `config.toml` (gitignored).  
**Template** at root: `config.example.toml`.

Everything else lives here so the root is not a pile of experiment TOMLs.

| Directory | Contents |
|-----------|----------|
| `profiles/` | Stable named profiles (`27b`, `force_off`, `ramp_off`) |
| `gemma4/` | Near-vanilla / greedy / stable Gemma 4 probes |
| `gates/` | G3–G6 ablation gate TOMLs (pain/dissipation, attractor break, …) |
| `ablation/` | Force/temp/ocean/splat ablations + sweeps |
| `experiments/` | Letter-run scratch (`E`–`I`, `runE`) — ephemeral knobs |
| `archive/` | Local bak/baseline snapshots — **do not treat as truth** |

## Examples

```bash
cp config.example.toml config.toml
cp configs/profiles/config.27b.toml config.toml
./target/release/hydrodynamic-swarm --config configs/gemma4/config.gemma4_greedy.toml ...
./scripts/chat_gemma4.sh   # defaults to configs/gemma4/config.gemma4_greedy.toml
```

## Naming

- `pleasure_*` / `pain_*` in TOML are **legacy ± splat labels** (attract / repel), not emotions.  
- `dream_*` knobs mean **offline consolidation / replay weight**, not sleep mysticism.
