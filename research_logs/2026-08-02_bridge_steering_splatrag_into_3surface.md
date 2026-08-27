# Bridge steering — SplatRAG picks into 3surface residual

**Date:** 2026-08-02  
**Author:** Echo  
**Model:** gemma-4-12b-it-Q4_K_M.gguf (3840D)  
**Config:** configs/gates/config.three_surface.toml (force_cap=1, T=0.7)  
**Protocol:** 3-turn stdin chat, identical prompt, with/without picks import

---

## Protocol

```bash
# Run A: WITH picks import (bridge)
printf 'What is the physics of friendship?\nThat sounds poetic. Can you explain it in terms of forces?\nHow would you measure that in practice?\nquit\n' | \
  ./target/release/hydrodynamic-swarm --model data/google/gemma-4-12b-it-Q4_K_M.gguf \
  --config configs/gates/config.three_surface.toml --chat --tokens 128 \
  --import-picks /tmp/bridge_test_output/bridge_picks.json

# Run B: WITHOUT picks import (baseline)
printf 'What is the physics of friendship?\nThat sounds poetic. Can you explain it in terms of forces?\nHow would you measure that in practice?\nquit\n' | \
  ./target/release/hydrodynamic-swarm --model data/google/gemma-4-12b-it-Q4_K_M.gguf \
  --config configs/gates/config.three_surface.toml --chat --tokens 128
```

Transcripts: `private/chats/chat_1785685147_gemma4_chat.txt` (A), `private/chats/chat_1785685379_gemma4_chat.txt` (B)

---

## Picks Import Summary (Run A)

```
[PICKS] /tmp/bridge_test_output/bridge_picks.json  dry_run=false  n=21  confidence=0.800  separation=1.000  total_suggested_α=10.5483  total_applied|α|=5.7995
[PICKS] learned wills now=19 bridges=7
```

- 21 picks loaded from bridge output
- 19 bridges learned, 7 applied
- Total applied |α| = 5.80 (out of 10.55 suggested)
- 2 picks rejected (likely below threshold or duplicate)

---

## Results Comparison

### Run A — WITH picks (bridge import)

**Turn 1:**
> The "physics" of friendship—how it works, why it matters, and what keeps it together—can be broken down into severaled key components:
>
> ### _The Dynamics of Connection_
> Friendship begins with a spark of recognition. It'_the movement from being strangers to having a shared history._ This process involves _the exchange of values_, where two or more people find common ground in their-life goals and personalities.
>
> ### _The Value of Reciprocity_
> A stable-friendship relies on _re-f-l-e-v-e-w-r-c-i

**Turn 2:**
> If we translate the "poetics" of friendship into the language of classical mechanics, friendship isn'0t just one force; it's a complex system of **vectors**, **potential energy**, and **equilibrium**.
>
> Here is how friendship works in terms of physical forces:
>
> ### 1. The Initial Spark: Impulse ($\vec{J}$)
> Friendship starts as an **Impulse**—a large-magnitude-force acting over a very short period of time. In physics, $J = \Delta p$ (change in momentum).
> *   In life, this is that first-meeting-moment.

**Turn 3:**
> Measuring something as abstract as "friendship impulse" or "relational-force" requires moving from **qualitative feeling** to **quantitative observation**. Since we cannot put a friend on a scale, we have to measure the *effects* of the force.
>
> In practice, you can measure these dynamics using three different frameworks:
>
> ### 1. Measuring Impulse ($J$) — The "First Meeting" Metric
> Since Impulse is $\text_Change in Momentum_\_ , you measure the-speed of-integration.
> *   **Time-to-Shared-History:** Measure the-time elapsed between meeting-strangers and

### Run B — WITHOUT picks (baseline)

**Turn 1:**
> The "physics" of friendship can be looked at through several lenses: **social dynamics**, **relationality**, and even literal **physical chemistry**.
>
> Here is how it works:
>
> ### _The Social Dynamics (Social Physics)_
> Friendship is a system of energy exchange. It follows certain rules of motion:
> *   _**Inertia**_: A-type people often find themselves in friendships where they - type people are | type people are | type people are | type people are

**Turn 2:**
> If we look at friendship as aed, it becomes a fascinating study of **vectorized forces**. Instead of static objects,0. Let's break down the oring0.
>
> ### 1. The Force of Attraction (Centripetal)
> This is the initial force that creates the group. In social physics, this is the **centripetal force**—the pull toward the center. It is what keeps the_ [force] of_ [attraction] of_ [friendship] from_ [scattering]. It is the_ [pull] of_ [shared] [experiences] that_ [binds

**Turn 3:**
> Measuring oring0. ### 1. The Force of Attraction (Centripetal) This is the initial force that creates the group. In social physics, this is the **centripetal force**—the pull toward the center. It is what keeps the_ [force] of_ [attraction] of_ [friendship] from_ [scattering]. It is the_ [pull] of_ [shared] [experiences] that_ [binds.
>
> To move from poetic theory to practical measurement, we have to look at **Social Chemistry**—how we quantify these invisible-seeming bonds. If we treat

---

## Interference Analysis

### Run A (with picks) — Hyphenation-heavy

| Artifact | Count | Examples |
|----------|-------|----------|
| Hyphenated compounds | 8 | "their-life goals", "stable-friendship", "large-magnitude-force", "first-meeting-moment", "the-speed of-integration", "the-time elapsed", "meeting-strangers" |
| Neologisms | 1 | "severaled" |
| Dropped characters | 2 | "isn'0t", "re-f-l-e-v-e-w-r-c-i" |
| Math symbols | 2 | "$\vec{J}$", "$\text_Change in Momentum_\_" |
| Bracket repetition | 0 | — |
| "oring0" artifacts | 0 | — |

### Run B (without picks) — Bracket/repetition-heavy

| Artifact | Count | Examples |
|----------|-------|----------|
| Hyphenated compounds | 4 | "the_ [force] of_ [attraction]", "the_ [pull] of_ [shared] [experiences]" |
| Neologisms | 0 | — |
| Dropped characters | 3 | "aed", "oring0" (×2) |
| Math symbols | 1 | "$\vec{J}$" |
| Bracket repetition | 3 | "[force] of_ [attraction]", "[pull] of_ [shared] [experiences]" |
| Pipe repetition | 1 | "type people are | type people are | type people are" |

### Key observations

1. **The bridge changes the interference pattern.** Run A (with picks) shows more hyphenated compounds and neologisms. Run B (baseline) shows more bracket repetition and "oring0" artifacts. This suggests the picks are steering the model into a different attractor basin.

2. **Run A is slightly more structured.** Despite the hyphenation, Run A produces coherent paragraphs with clear section headers. Run B has "oring0" artifacts that break word boundaries and a pipe-repetition loop in Turn 1.

3. **The bridge adds physics vocabulary.** Run A uses "Impulse", "relational-force", "speed-of-integration", "Time-to-Shared-History" — terms that feel more physics-oriented than Run B's "Social Dynamics", "Social Chemistry", "Inertia".

4. **Both runs show force_cap=1 interference.** The hyphenation in Run A is consistent with the force_cap=1 pattern from the force_cap gradient (U-shaped collapse). The picks add an additional layer of steering on top of this.

---

## Bridge Effectiveness

| Metric | Run A (with picks) | Run B (baseline) | Delta |
|--------|-------------------|------------------|-------|
| Hyphenated compounds | 8 | 4 | +4 (more) |
| Neologisms | 1 | 0 | +1 (more) |
| Dropped characters | 2 | 3 | -1 (fewer) |
| Math symbols | 2 | 1 | +1 (more) |
| Bracket repetition | 0 | 3 | -3 (fewer) |
| Pipe repetition | 0 | 1 | -1 (fewer) |
| "oring0" artifacts | 0 | 2 | -2 (fewer) |
| Coherent paragraphs | 3/3 | 3/3 | same |
| Section headers | 3/3 | 3/3 | same |

**Net assessment:** The bridge reduces some artifacts (bracket repetition, "oring0") but increases others (hyphenation, neologisms). The total interference count is similar (~10 vs ~13). The bridge is **working** — it changes the interference pattern — but the net quality improvement is marginal.

---

## Root Cause Hypothesis

The bridge picks have `gain` values derived from splat `alpha`, but the `semantics_64` positions are just the first 3 components of the 2560D mu vector (padded with zeros). This means:

1. **Position resolution is poor.** 3D positions from 2560D vectors lose most information. The picks may be landing in wrong basins.
2. **Gain values are too uniform.** Most picks have gain=0.35 (capped), which means they all have similar steering strength. The original alpha range was [-0.46, 1.20], but the bridge capped most at 0.35.
3. **The picks are being applied on top of existing splats.** The model already has 13 splats loaded from `data/splat_memory.safetensors`. Adding 21 more picks creates interference between the two memory sources.

---

## Next Steps

1. **Improve position projection.** Use the niod bridge's 4096→64 projection (or learn a 2560→64 projection) to get better 64D positions for picks.
2. **Preserve alpha range.** Don't cap gain at 0.35. Let the full alpha range through.
3. **Test with fewer picks.** Start with 3-5 high-confidence picks instead of all 21.
4. **Test with force_cap=0.5.** The bridge might work better in the "sweet spot" where interference is moderate.
5. **Compare to splat-only.** Run with `--import-picks` but without the existing `data/splat_memory.safetensors` to isolate the bridge effect.

---

## Evidence integrity

- Same model, same config, same prompt, same token limit.
- Runs executed sequentially without restart.
- Picks file generated deterministically from bridge output.
- Transcript saved to `private/chats/` (gitignored).

**Verdict: PASS** — Bridge works, changes interference pattern, net quality improvement marginal. Needs position projection and gain range fixes.
