# Multi-bridge return (continuity first)

**Run:** `logs/continuity_multibridge_20260717_002752`  
**Script:** `scripts/continuity_multibridge.sh`  
**No** `--clear-memory`.

## Sequence

| Step | Prompt | Card | nearest | pot |
| --- | --- | --- | --- | --- |
| A1 | Physics of Friendship | WARM | 31.5 | 0.038 |
| B1 | CUDA kernel tips | WARM | 31.5 | 0.210 |
| A2 | Friendship again | WARM | 31.5 | 0.217 |

**Verdict:** `PASS_RETURN` (A2 stays WARM after B; pot recovers to ~0.22).

Bridges after: 3 fps (`0x8b262d40`, `0x18934cbe`, `0x9ce81984`).

## Notes

1. **nearest_L2 ≈ 31.5 always on bridge revisits**  
   That is `prefill_bridge_offset_frac * sigma = 0.35 * 90`. Soft-offset deposit → prefill sits on the ring, not at μ. Nearest alone does not separate A vs B; **pot** does.

2. **Pot is the multi-bridge signal**  
   A1 pot weak (0.038), B1/A2 stronger (~0.21). Return to A still WARM with pot recovery.

3. **Cull pressure**  
   A2 session culled 19 dead splats mid-run (bridges reserved). Store end: 12 records TCT. Watch prune vs bridge protect under long multi-bridge sessions.

4. **Text quality** not scored here (continuity geometry only).

## Ops

```bash
./scripts/continuity_multibridge.sh
# card: logs/continuity_multibridge_*/CONTINUITY_CARD.md
```
