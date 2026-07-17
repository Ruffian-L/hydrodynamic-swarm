# Continuity lane (public)

**North star:** memory means something measurable — residual scars that save, load, and show up in the **start basin** of the next run. Not prettier paragraphs. Not feelings claims.

## What is proven (2026-07-16 / 17)

| Claim | Evidence |
| --- | --- |
| Scars persist (safetensors + **TCT-splat-lite**) | death → reload |
| Trail-only reload is **LOCALITY COLD** (~180 L2) | research logs |
| **Prefill-bridge** warms the start basin | offset 0.35σ → nearest ≈ 31.5, pot lives |
| On-center \(F_s \approx 0\) is geometry, not dead memory | measure **potential** + nearest |
| **Multi-bridge return** without `--clear-memory` | A→B→A **PASS_RETURN** (e.g. 2026-07-17 pot≈0.71 all steps) |
| Bridge **gain** = weight proxy (stay 0.75; not evaporated) | `list_bridges.py` · fix `decay_step` skips bridges |
| Prune reserves prefill-bridges | unit test `prune_reserves_prefill_bridges` |
| Novel prompt not false-warm | e.g. capital-of-France → **LUKE** nearest≈217 pot≈0.004 |

## Public ops (no GPU to *read* museum)

```bash
./splat-lens museum          # watch continuity demos
python3 scripts/continuity_selftest.py  # no GPU — card + list_bridges fixtures
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

### Optional consumer: niodoo-live

Live can load the same binary via residual apply (not the 64D packet path):

```text
--tct-splat-path path/to/splat_memory.tct
--tct-splat-bridge-only
--tct-splat-gain 1.0
--tct-splat-clamp 0.05
```

Requires a **matching residual dim** (e.g. Gemma 2560 store on a 2560 hidden model). Dim mismatch is refused, not padded.  
Telemetry keys: `tct_force_norm`, `tct_potential`, `tct_nearest_dist`, `tct_n_active`.  
First visit may still be **COLD** (locality) even when load succeeds — same physics as hydro.

Wire-up lives in the `niodoo-live` tree (`tct_splat_lite` + **final post-norm** residual apply). Hydro remains the continuity **mint** and KPI home.

**Live residual geometry (2026-07-17):** scars must be queried at the **final RMSNorm last-token residual** (hydro mint space, `‖μ‖~O(100)` on Gemma-4B). Mid-layer post-attn is a different site → false COLD (`nearest~1e4`). After final-hidden apply: Friendship bridge KPI **WARM** `nearest_min≈184` `pot_max≈0.013` `force_max=0.05` (clamp). See `niodoo-live/research_logs/2026-07-17_residual_tct_final_hidden_warm.md`.

**Gemma speech on live (2026-07-17):** use low mid-layer blend (0.02), free system, not February theta. Smoke: `niodoo-live/scripts/gemma_speak_smoke.sh` — mode 0 clean **SPEECH PASS**; mode 2 loads hydro TCT and still **SPEECH PASS** on Friendship. See `niodoo-live/research_logs/2026-07-17_gemma_speak_not_garbage.md`.
