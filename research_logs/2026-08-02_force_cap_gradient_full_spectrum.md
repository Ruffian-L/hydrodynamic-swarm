# Force-cap gradient — collapse quality across 0 → 1.0

**Date:** 2026-08-02  
**Author:** Shep  
**Model:** gemma-4-12b-it-Q4_K_M.gguf (3840D)  
**Protocol:** 3-turn stdin chat, identical prompt across runs

## Prompt (identical across all runs)

```
you> What is the physics of friendship?
you> That sounds poetic. Can you explain it in terms of forces?
you> How would you measure that in practice?
```

## Summary Table

| Run | force_cap | T | Tokens | Pattern | Quality |
|-----|-----------|---|--------|---------|---------|
| A | 0 | 0 | 64 | Clean, minimal interference | ★★★★★ |
| B | 0.25 | 0.2 | 128 | **Exploded** — math symbols, Unicode, script tags, garbled | ★☆☆☆☆ |
| C | 0.5 | 0.7 | 128 | Bracketed repetition, neologisms ("Friendshipedness") | ★★★☆☆ |
| D | 0.75 | 0.7 | 128 | "ed-ness" loops, hyphen chains | ★★☆☆☆ |
| E | 1.0 | 0.7 | 128 | Hyphenated compounds, word chains ("the-the-the") | ★★☆☆☆ |

## Key Findings

### 1. The Collapse is NOT monotonic — it's U-shaped

- **force_cap=0**: Clean generation. Minor artifacts from field_wake/splat but coherent.
- **force_cap=0.25**: **Worst case.** The model explodes into math symbols, Unicode, HTML tags, and garbled text. This is the resonance zone — the force magnitude matches the model's internal attractor spacing, causing constructive interference that amplifies into chaos.
- **force_cap=0.5**: Bracketed repetition ("A [Physical] force_A [Physical] force"), neologisms. Partial structure survives.
- **force_cap=0.75**: "ed-ness" loops — the force is strong enough to dominate but not resonant enough to explode. Degraded but readable.
- **force_cap=1.0**: Hyphenated compounds, word chains. Similar to 0.75 but with more deterministic looping.

### 2. Temperature matters

Run B (force_cap=0.25, T=0.2) is worse than any high-T run. Low temperature + resonant force = the model gets locked into a single attractor basin and amplifies it. Higher T (0.7) provides enough noise to escape the worst resonance.

### 3. The "sweet spot" is force_cap ≈ 0.4-0.6

At this range, the physics force is strong enough to shape the output (hyphenation, neologisms) but not so strong that it causes resonance collapse or complete domination. The model still produces structured markdown with numbered sections.

### 4. Residual norm is stable across all runs

From the collapse probe JSONL data:
- Turn 2 settle: residual_norm=166.6, entropy=0.73, p_top1=0.66
- Turn 3 settle: residual_norm=177.5, entropy=0.57, p_top1=0.83
- Turn 4 (short): residual_norm=164.0, entropy=0.03, p_top1=0.996

The residual norm does NOT explode during collapse. The collapse is in the **token distribution** (entropy/margin), not the residual magnitude. This means the force is steering the latent direction but the magnitude stays bounded — the physics is doing what it's supposed to do. The problem is the *interaction* between the force direction and the model's internal geometry.

### 5. Previous vs. current config mismatch

The earlier receipts (Lumina's physics ON vs OFF) used different default configs than the current force_cap gradient. The earlier "physics ON" run showed hyphenated compounds consistent with force_cap≈0.5-1.0. The current gradient confirms this is a force_cap effect, not a config drift.

## Implications

1. **There is a usable operating range.** force_cap 0.4-0.6 produces degraded but structured output. This is where steering is possible without collapse.
2. **force_cap=0.25 is the danger zone.** Avoid this range unless you want resonance collapse.
3. **The force doesn't need to be strong to matter.** Even at 0.25, the force causes catastrophic failure. At 0.5, it's clearly visible. The Jacobian sensitivity is high.
4. **Entropy, not residual norm, is the collapse signal.** Monitor entropy/margin, not residual_norm, for collapse detection.

## Next Steps

1. Run a fine-grained sweep around force_cap 0.3-0.7 with T=0.7 to find the exact boundary of usable steering.
2. Test whether the "ed-ness" pattern at 0.75 is a tokenizer artifact or a genuine latent-space phenomenon.
3. Bridge the collapse probe telemetry to SplatRAG — if we can measure the Jacobian at the point of collapse, we can address memories to the right latent basin.

---

*This study synthesizes data from Echo's interference curve, Lumina's physics ON/OFF comparison, and Shep's force_cap gradient sweep.*
