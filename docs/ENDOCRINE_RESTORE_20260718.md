# endocrine.rs restore (2026-07-18)

**Not from grok exports.** Copied from re-plugged SanDisk backup after Jason pointed at Shep’s build.

## Source chosen

| Path | Lines | SHA-256 | Notes |
|------|------:|---------|-------|
| `mar22loop/shep-loop/hydrodynamic-swarm/src/endocrine.rs` | **153** | `f053d04c…698126` | **Shep-loop — best:** spawns worker, hash embeds |
| `recentfiles/hydrodynamic-swarm-mar8/...` | 119 | `d9ddaebc…` | earlier stub, no spawn |
| `projects/jasonarchive/...` | 115 | `5b4de7db…` | same family as mar8-ish |

Live tree previously had **zero** `endocrine.rs`. It was not imaginary — it lived on backup / Shep loop and dropped from current `projects/hydrodynamic-swarm`.

## What was restored into live tree

1. `src/endocrine.rs` — Shep copy + provenance header  
2. `src/main.rs` — `mod endocrine;`  
3. `src/niodoo.rs` — `apply_monolith` + `noise_sigma` / `viscosity` / `tag_gravity_mult` / `eureka_impulse`

## Wired 2026-07-18 (same night as restore)

See **`docs/ENDOCRINE_SHEP_WIRED_20260718.md`**.

- `create_endocrine_system()` at startup (default ON; `--no-endocrine` off)
- pain / high-δ → `ExecuteTool` signal (rate-limited)
- bloom drain → `apply_monolith` + `[BLOOM]` log
- `steer()` uses eureka impulse + cooled viscosity; `tick_endocrine()` decays

March full bloom-poll-every-token / tag auto-ExecuteTool was **not** ported wholesale (architecture differs). This is the Shep-critical path.

## Honest stub status (same as Shep left it)

- `FunctionGemma::strict_execute` — fake FACT strings  
- `TinyEmbed::embed_4d` — hash pseudo-embed, not real model  
- Real FunctionGemma 270M un-stub = later  

## Attribution

Shep built / iterated endocrine in the mar22 loop. Grok chat blueprints also floated diffs; the **on-disk Shep-loop file** is the restore source.
