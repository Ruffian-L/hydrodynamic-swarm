# Continuity lane (public)

**North star:** memory means something measurable — residual scars that save, load, and show up in the **start basin** of the next run. Not prettier paragraphs. Not feelings claims.

## What is proven (2026-07-16 / 17)

| Claim | Evidence |
| --- | --- |
| Scars persist (safetensors + **TCT-splat-lite**) | death → reload |
| Trail-only reload is **LOCALITY COLD** (~180 L2) | research logs |
| **Prefill-bridge** warms the start basin | offset 0.35σ → nearest ≈ 31.5, pot lives |
| On-center \(F_s \approx 0\) is geometry, not dead memory | measure **potential** + nearest |
| **Multi-bridge return** without `--clear-memory` | A→B→A **PASS_RETURN** |
| Bridge **gain** = weight proxy (Friendship 0.75 vs weaker basins) | `list_bridges.py` |
| Prune reserves prefill-bridges | unit test `prune_reserves_prefill_bridges` |

## Public ops (no GPU to *read* museum)

```bash
./splat-lens museum          # watch continuity demos
./scripts/continuity_status.sh   # store + bridges + latest cards (needs local data/)
./scripts/continuity_revisit.sh  # same-prompt revisit cards (CUDA + model)
./scripts/continuity_multibridge.sh  # A→B→A return verdict
```

Local runtime stores (`data/splat_memory.*`, `data/bridge_prompts.json`) are **gitignored** — you mint your own after generate.

## Card language

```text
CONT  WARM|NEAR|LUKE|COLD  nearest_min=…  pot_max=…  gain_max=…  bridges=…
```

- **nearest ≈ 31.5** often = soft offset ring (`0.35 × σ=90`), not “all prompts identical.”
- **pot** separates basins better than nearest alone.
- **gain_max** = strongest prefill-bridge α (earned mass).

## What this is *not* (yet)

- Universal residual transfer across different model dims (Gemma 2560 ≠ Llama 4096).
- Live product modernization (optional consumer of TCT when dim matches).
- Official ARC-AGI leaderboard results.

## Research log pointers

- `research_logs/2026-07-16_prefill-bridge-scar.md`
- `research_logs/2026-07-16_multi-bridge-basins.md`
- `research_logs/2026-07-16_soft-offset-bridge.md`
- `research_logs/2026-07-17_multi-bridge-return.md`
- Museum: milestone `reload-bridge-offset-35` in `tools/museum/catalog.json`

## TCT wire (portable scars)

Binary `TCT1` v3: residual center + σ + signed α + λ + `trigger_kind` (5 = prefill_bridge) + `prompt_fp`.  
Import/export: `--import-tct` / `--export-tct`. Dim must match the model (**INV-5**).
