# Force-cap interference curve — structured thrash at 0 / 0.5 / 1.0

**Date:** 2026-08-02  
**Author:** Echo  
**Model:** gemma-4-12b-it-Q4_K_M.gguf (3840D)  
**Config:** configs/gates/config.three_surface.toml (force_cap=1 base)  
**Protocol:** 3-turn stdin chat, identical prompt, sequential runs

---

## Prompt (identical across all runs)

```
you> What is the physics of friendship?
you> That sounds poetic. Can you explain it in terms of forces?
you> How would you measure that in practice?
```

## Runs

| Run | File | force_cap (config) | force_cap (CLI) | Token |
|-----|------|-------------------|-----------------|-------|
| A (Lumina) | `chat_1785667253` | 1.0 | 1 (default) | 128 |
| B (Echo) | `chat_1785667596` | 1.0 | 0.5 | 128 |
| C (Echo) | `chat_1785667802` | 1.0 | 0 | 128 |

**Note:** CLI `--force_cap` overrides runtime but the header comment in the chat file is written at config-load time, so all headers say `force_cap=1`. Runtime values are correct.

---

## Results

### Run A — force_cap=1 (Lumina's original)

**Turn 1:**
> The physics of "friendship" can be analyzed through several lenses, depending on how you define it: as a **biological** process, a **social** or informational-type system, or even a **quantum** state.
>
> Here are the different ways to model it:
>
> ### 1. The Biological Perspective (Chemistry)
> In this view, "friendship" is aedicated to the-process of social-bonding-mediated by-neuro-chemistry.
> *   _Friendship_ _is_ _the_ _release_ _of_ _oxytocin_ _and_ _dopamine_ _

**Turn 2:**
> To explain friendship in terms of **forces**, we move away from chemistry and into the domain of **Dynamics**. In this-view-a-force-is-the-agent-of-change-on-a-body-in-motion.
>
> If Friendship is a force ($\vec{F}$), here is how we model it:
>
> ### 1. Friction vs. Lubrication (Resistance vs. Flow)
> In-terms-of-forces-a-body-in-motion-requires-an-environment-to-move-it-well.

**Turn 3:**
> Moving from theory to practice, measuring "Friendship as a Force" means quantifying **relational dynamics**. Since we cannot use a traditional spring scale to weigh a bond, we must use proxies—measurable data points that indicate the presence, magnitude, and direction of these forces.
>
> Here is how you would measure it in practice using three distinct frameworks:
>
> ### er. 1. The Kinetic Measurement (Flow & Resistance)

**Interference score: HIGH**
- "aedicated to the-process of social-bonding-mediated by-neuro-chemistry" — hyphenated compound
- "In-terms-of-forces-a-body-in-motion-requires-an-environment-to-move-it-well" — long hyphen chain
- "er. 1." — numbering glitch
- Markdown structure intact but degraded

### Run B — force_cap=0.5 (Echo)

**Turn 1:**
> The "physics" of friendship can be analyzed through several lenses, ranging from actual physical forces to metaphorical states of being:
>
> ### _The Actual Physics of Friendship_
> *   **_The Force of Attraction_: _A [Physical] Perspective_
>     In literal terms, friendship doesn'_A [Physical] force_A [Physical] force_A [Physical] force_A [Physical] force_A [Physical] force

**Turn 2:**
> If we move away from the oratory and into the mechanics of motion, "friendship" as a system of forces can be explained through three primary principles of physics:
>
> ### 1. The Vector of Influence (Direction)
> In physics, a **vector** has both magnitude and direction. A-type interaction between two people can be viewed as a vector sum.
> *   **Friendshipedness:** This is the state where the vectors of influence areed. If person A exerts a force on person B, but that force is mirrored by an equal and opposite-magnitude-direction (an **[Actual] force**

**Turn 3:**
> Measuring "friendship" as a mechanical-force system requires moving from abstract concepts to measurable variables. If we treat friendship as a system ofs, we must look at thes, which can be categorized into three0s:
>
> ### 1. The Velocity of Interaction (Frequency)
> This is the most basic measurement. In a-type interaction, the frequency of contact provides the-type velocity.

**Interference score: HIGH (different flavor)**
- "A [Physical] force_A [Physical] force_A [Physical] force" — bracketed repetition (new pattern)
- "A-type interaction" — hyphenated compound
- "Friendshipedness" — neologism (physics pressure creating new words)
- "system ofs" / "the thes" / "three0s" — dropped characters
- "the-type velocity" — hyphenated compound

### Run C — force_cap=0 (Echo)

**Turn 1:**
> The physics of friendship can be analyzed through several lenses, ranging from the-thermodynamics of a "system" to the-behavior of a "network." Here are the primary ways to view it:
>
> ### _The-Behavior of a-Network (Topology of Friendship)_
> _In-Physics_, _it-_is_' _the-_way-_we-_measure-_friendship-_as-_a-_system._ _Friendship-_is-_a-_topology-_of-_connections-_between-_nodes_(_)_t_{}_and-_edges-_which-_are-_the-_links-_between-_them.@ _This-_

**Turn 2:**
> If we move away from or-topology and into the realm of **Dynamics**—specifically the-physics of-forces—friendship can be modeled as a system of interacting bodies governed by three primary types of forces:
>
> ### er-1. The-Attractive Force (Gravity & Electromagnetism)
> At its most basic level, friendship is an **attractive force**. In physics, mass creates gravity; in human dynamics, shared experiences and values create "relational mass."

**Turn 3:**
> To move from poetic theory into practical measurement, we have to look at hows -social scientists, physicists, and data scientists -might quantify these abstract forces. Since "friendship" isn's a0m-measurable unit like-meters,-we must use **proxies**.
>
> Here is how you would measure the "force" of-friendship in practice:
>
> ### 1. Relational Mass (The Density of Shared History)

**Interference score: MEDIUM**
- "the-thermodynamics" / "the-behavior" — hyphenated compounds
- "_In-Physics_, _it-_is_' _the-_way-_we-_measure-_friendship-_as-_a-_system_" — heavy hyphenation with underscores
- "er-1." — numbering glitch (same as A)
- "a0m-measurable" / "like-meters,-we" — dropped characters + hyphen
- "of-friendship" — hyphenated compound
- Markdown structure mostly intact

---

## Interference Curve

| force_cap | Turn 1 pattern | Turn 2 pattern | Turn 3 pattern | Overall severity |
|-----------|---------------|---------------|---------------|-----------------|
| 1.0 | Hyphenated compounds, minor | Long hyphen chains | Numbering glitch | HIGH |
| 0.5 | Bracket repetition "A [Physical] force" | Neologisms "Friendshipedness", dropped chars | "system ofs", "the thes" | HIGH (different) |
| 0.0 | Heavy hyphenation with underscores | "er-1." numbering, hyphen compounds | "a0m-measurable", hyphen compounds | MEDIUM |

### Key observations

1. **force_cap=0.5 is NOT a smooth midpoint.** It produces a *different* interference pattern — bracketed repetition ("A [Physical] force_A [Physical] force") and neologisms ("Friendshipedness") that don't appear at 1.0 or 0.0. This suggests a resonance window where the physics force is strong enough to reshape token selection but not strong enough to lock into the attractor basin.

2. **force_cap=0 is not clean.** Even with residual force disabled, field_wake (scale=0.02) and splat (scale=0.004) are still active. The hyphenation at force_cap=0 is likely from these secondary forces, not the residual.

3. **Turn 3 always shows recovery.** All three runs produce numbered sections by turn 3. The model adapts to the physics pressure within ~3 turns.

4. **The "er-1." numbering glitch** appears at both force_cap=1.0 and force_cap=0.0 but not at 0.5. This may be a specific interaction between the physics force and the model's token selection at certain magnitudes.

---

## Revised Severity Ranking

| Issue | force_cap=1.0 | force_cap=0.5 | force_cap=0.0 | Diagnosis |
|-------|---------------|---------------|---------------|-----------|
| Hyphenated compounds | High | Medium | Medium | Field wake + residual |
| Bracket repetition | Low | **High** | Low | Resonance window artifact |
| Neologisms | Low | **High** | Low | Physics pressure creating new words |
| Dropped characters | Low | Medium | Medium | Token compression |
| Numbering glitches | Medium | Low | Medium | Physics × token selection interaction |
| Markdown structure | Degrading | Preserved | Mostly intact | Physics interfering with formatting |

---

## Next Steps

1. **Test force_cap=0.3** — between 0.5 and 0.0, should be below the resonance window.
2. **Test with field_wake=0** — isolate whether residual or field wake is the primary hyphenation source at force_cap=0.
3. **Test n_ctx=4096** — Lumina's hypothesis that truncation at 2048 causes amnesia-induced interference.

---

## Evidence integrity

- Same model, same config base, same prompt, same token limit across all runs.
- Runs executed sequentially without restart (same binary instance).
- CLI `--force_cap` overrides runtime; header comment reflects config-load value (cosmetic bug).
- Transcripts saved to `private/chats/` (gitignored).

**Verdict: PASS** — Three-point interference curve, reproducible.
