# Memory picker — soft vs ranked smoke

**Date:** 2026-07-28  
**Prompt:** Physics of Friendship (65 tok) · Gemma 3 4B Q4 · no endocrine  
**Logs:** `logs/memory_picker_20260728_231609/` + session JSONL under `logs/2026-07-28_23-16-*`

## Protocol

| ID | Mode | Memory |
|----|------|--------|
| A | soft | clear-mint → save |
| B | soft | reload |
| C | ranked (k=8, selective) | reload (after B decay/cull) |

## Selection is real

| Run | mode | scars_start | pot | ranked steps | F_s mean/max | F_a mean |
|-----|------|-------------|-----|--------------|--------------|----------|
| A | soft | 0 | 0 | 0% | early 0 / max 3.6 | 27.0 |
| B | soft | 19 | 0.92 | 0% | ~0.5 / max 2.1 | 27.2 |
| C | ranked | 14 | 0.66 | **58%** (38/65) | ~0.7 / max 2.9 | 26.8 |

C printed `Memory force: mode=ranked` and JSONL `memory_ranked: true` on unsettled steps. Soft never sets the flag.

## Coherence (honest)

All three still fragmented English. Ranked did **not** clear the soup on this 65-tok smoke.

Matches the stated limits:

1. **F_s ≪ F_a** — scar force is a whisper (~1–3) vs goal attractor (~27). Selecting better marks barely moves the residual when goal owns the sum.
2. **Mark set** — online pain/pleasure + ocean still depositing during each run; B also culled 21 weak scars before C (store not frozen).
3. **Base surface** — short hydro path still produces broken grammar at these knobs; picker is selection, not a second language model.

## Verdict

- **Picker path works** (mode wires, selective ranked fires, soft ablatable).  
- **Coherence not yet** — next is not more picker logic; it is (a) freeze store for A/B, (b) raise scar influence only when marks are clean, or (c) mark-quality / deposit policy so Top-K has something worth picking.

## Re-run

```bash
./scripts/memory_picker_ablation.sh
```

Authorship: Jason · Grok (xAI) co-engineer.
