# B4d-q — length cap + light quality knobs (4B)

**Date:** 2026-07-12  
**Authors:** Jason (steering / co-engineer) · Grok (xAI) (config + smoke)  
**Model:** `gemma-3-4b-it-Q4_K_M.gguf`  
**Phase:** quality / length management — **physics layer frozen** (B4d stack)

---

## Decision

From length ceiling map (`2026-07-12_B4d-length-ceiling.md`):

- Forces calm at 120–150 tok  
- Clean paragraph budget ≈ **50–70**  
- After ~70–80: surface fray (capacity), not F_s/F_a

**Lock:** option 1 + light quality knobs — **B4d-q**.

| Knob | B4d | **B4d-q** | Why |
|------|-----|-----------|-----|
| `max_tokens` | 90 | **65** | hard budget in good zone |
| launcher default | 50 | **65** | match config |
| `temperature` | 0.82 | **0.88** | slightly freer surface |
| `rep_penalty` | 1.32 | **1.36** | less mid-run mash loops |
| `pleasure_alpha` | 1.0 | **0.85** | softer scars |
| `pain_alpha` | -0.5 | **-0.4** | softer scars |
| prompt | one paragraph | **one short paragraph** | framing |
| physics | B4d | **unchanged** | force hunt closed |

---

## Smoke (65 tok) — verified

```text
Fs early≈0.05  late≈1.67  Fa late≈23.9  (no latch)
```

> Friendship is a complex phenomenon involving mutual attraction and bonding, rooted within physics-like social interaction between two individuals like network where it's that can be modeled as an interplay dynamics, friendship involves energy exchange driven by neural networks based on reciprocity; with highly interconnected system: reciprocal information systems—social interactions--driven –in sharing

On-topic full budget; soft mid softness only (expected 40–70 zone). Log: `logs/b4d_q_65_verify.txt`.

---

## Reproduce

```bash
cd /home/ruffianl/projects/hydrodynamic-swarm
./run_swarm.sh
# or
./run_swarm.sh "Explain the Physics of Friendship in one short paragraph." 65
```

Do **not** re-open force knobs unless F_s latches again (it should not at 65).

---

## Next (later, not this pass)

- Option 3: quality early-stop after ~55–60 if entropy/δ spikes  
- Or step back up model size for long form (re-scale via √-law + splat geo)

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** B4d-q length/quality default lock + verify smoke
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-12
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends. Physics closed; 4B demos target ~65 tok clean paragraphs.
---
