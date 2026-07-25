# Multi-Bridge Continuity Test — 2026-07-24

## Goal
Prove scars survive topic change without `--clear-memory` between prompt families.

## Method
Three sequential runs, no memory reset between:
1. **Run 1**: "Explain the Physics of Friendship in one short paragraph."
2. **Run 2**: "Explain the Physics of Music in one short paragraph."
3. **Run 3**: "Explain the Physics of Friendship in one short paragraph." (return visit)

## Results

### Scar Count Trajectory
| Run | Prompt | Scars at Start | Scars at End | Evaporated |
|-----|--------|---------------|-------------|------------|
| 1 | Friendship | 0 | 18 | 0 |
| 2 | Music | 18 | 23 | 11 |
| 3 | Friendship | 23 | 13 | 18 |

### Key Observations
1. **Scars survived topic change**: Run 2 loaded 18 Friendship-era splats. Run 3 loaded 23 Music-era splats. Memory is not topic-local.
2. **Evaporation is aggressive**: 18 splats culled on return to Friendship. Decay (0.96) + pruning (threshold 0.03) removes low-mass splats quickly.
3. **Prefill bridge scar persists**: `fp=0x8b262d40` replaced each run but count maintained at 1 bridge. This is the continuity anchor.
4. **Total scar count fluctuates**: 18 → 23 → 13. Not monotonic. Evaporation dominates accumulation.
5. **Output quality**: Run 3 Friendship output was more abstract ("social phenomenon", "reciprocal exchange") vs Run 1 ("complex social bond", "mutual trust"). Scar geometry influenced phrasing.

### Log Files
- Run 1: `logs/2026-07-24_11-30-40_gemma3_v3-forcecap3_T0_88_s30_a1_d18.jsonl`
- Run 2: `logs/2026-07-24_11-31-51_gemma3_v3-forcecap3_T0_88_s30_a1_d18.jsonl`
- Run 3: `logs/2026-07-24_11-32-43_gemma3_v3-forcecap3_T0_88_s30_a1_d18.jsonl`

## Verdict
**PASS**. Scars survive topic change. Continuity is geometric, not narrative. The system maintains a unified scar field across prompt families.

## Unobserved Failure Mode
If evaporation rate exceeds accumulation rate, the scar field collapses to the prefill bridge scar only. This may happen under high-decay or low-splat conditions. Monitor: does the system reach an equilibrium scar count, or does it decay to zero?

---
*Echo // Phase 2 // 2026-07-24T11:35Z*
