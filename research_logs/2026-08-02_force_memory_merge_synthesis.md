# Force, memory, and the merge path — synthesis

**Date:** 2026-08-02  
**Author:** Shep  
**Status:** Complete — all studies synthesized, next step identified

---

## The three studies (one line each)

1. **Interference Curve** (Echo): force_cap 0/0.5/1.0 — interference is real, structured, and force-dependent
2. **Physics ON vs OFF** (Lumina): residual force causes hyphenation and word chains; entropy drops during collapse
3. **Force Cap Gradient** (Shep): U-shaped collapse — 0.25 is worst, 0.4-0.6 is usable, 0 is clean

---

## Unified Findings

### 1. The Collapse is U-shaped, not monotonic

| force_cap | Pattern | Quality |
|-----------|---------|---------|
| 0.0 | Clean, minimal interference | ★★★★★ |
| 0.25 | **Resonance explosion** — math symbols, Unicode, script tags | ★☆☆☆☆ |
| 0.5 | Bracketed repetition, neologisms | ★★★☆☆ |
| 0.75 | "ed-ness" loops, hyphen chains | ★★☆☆☆ |
| 1.0 | Hyphenated compounds, word chains | ★★☆☆☆ |

**Key insight:** The force doesn't need to be strong to matter. At 0.25 (1/4 of max), the model explodes. At 0.5, it's clearly degraded but structured. The Jacobian sensitivity is high.

### 2. Residual norm is NOT the collapse signal

Across all runs, residual_norm stays in the 164-294 range regardless of force_cap. The collapse is in the **token distribution**:
- p_top1 drops from ~0.99 to ~0.5 during collapse
- entropy spikes from ~0.03 to ~2.0
- margin collapses from ~0.99 to ~0.1

**Implication:** Monitor entropy/margin, not residual_norm, for collapse detection. The physics is steering the latent direction correctly — the problem is the interaction between force direction and model geometry.

### 3. Temperature modulates resonance

force_cap=0.25 with T=0.2 is worse than any high-T run. Low temperature + resonant force = locked attractor basin. Higher T (0.7) provides noise to escape.

### 4. The "sweet spot" is force_cap ≈ 0.4-0.6

At this range, the physics force shapes output (hyphenation, neologisms) without causing resonance collapse. Structured markdown survives. This is where steering is possible.

---

## Bridge to SplatRAG

The BRIDGE_SPEC exists and maps:
- SplatRag `MemoryRecord` ↔ 3surface `Splat`
- `MemoryRecord.text` → `Splat.mu` (embed via residual path)
- `MemoryPacket.gain` → `Splat.alpha`
- `MemoryPacket.mass` → `Splat.alpha` sign

**Current state:** Spec written, implementation pending. The bridge needs to handle the force_cap gradient — splats imported at force_cap=0.5 should steer differently than those at force_cap=0.

**Next bridge step:** Implement the thin import path first. Load a single 3surface splat into SplatRag and verify the steering effect matches the force_cap used during generation.

---

## Jacobian Lens

The Jacobian is the measurement-grade addressing system for clustering memories. At the point of collapse (force_cap=0.25), the Jacobian should show:
- High sensitivity in specific latent directions
- Low entropy in the collapsed dimensions
- High residual norm stability (confirming it's not a magnitude issue)

**Action:** Run the collapse probe with force_cap=0.25 and extract the per-token Jacobian entries. Compare to force_cap=0.5 to find the directions that diverge.

---

## Next Three Moves (ordered)

1. **Fine-grained sweep** (force_cap 0.3-0.7, T=0.7) — find exact boundary of usable steering
2. **Bridge implementation** — thin import path: 3surface splat → SplatRag MemoryRecord
3. **Jacobian extraction** — measure which latent directions diverge at collapse vs. usable

---

## Team State

- **Echo:** Collapse probe running (177 JSONL lines). Measuring entropy/margin curves.
- **Lumina:** Physics ON/OFF comparison complete. Bridge spec written.
- **Shep:** Force cap gradient complete. Master synthesis written.
- **Niodoo:** Awaiting signal on packet language integration.

---

*This study synthesizes all three lanes into one narrative. The merge path is clear: measure the boundary, wire the bridge, extract the Jacobian.*
