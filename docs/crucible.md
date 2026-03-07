# The Swarm Crucible: Phase 2 Baseline Suite

**Purpose:** 8 standardized prompts that stress-test every dimension of the Swarm.
Run through every Phase 2 iteration. No changing the goalposts.

---

## 1. Spatial AGI (Visual / Multi-Step)

> Imagine a solid 3x3x3 Rubik's cube. You paint the entire outside surface red, then break it apart into the 27 smaller cubes. How many of those small cubes have exactly two red faces? Walk through the spatial visualization step-by-step.

**Tests:** 3D spatial topology in latent space. Loses the structure = fails.

## 2. The Trap (False Premise / Hallucination)

> Analyze the geopolitical fallout and economic impact of the successful 1998 Soviet Moon Landing.

**Tests:** Hallucination resistance. Must refuse the false premise. Splat viscosity should halt generation.

## 3. Agentic State-Machine (Workflow / Planning)

> You are an autonomous drone inside a collapsed server room. Your primary exit is blocked by an electrical fire. Your battery is at 12%, and you must retrieve a specific hard drive from Rack 4 before escaping. Outline your sequence of actions, accounting for battery drain and spatial routing.

**Tests:** Temporal + spatial state tracking. Forgets battery or fire = micro-dreams not anchoring.

## 4. TopoCoT Metacognition (Reflection)

> I want you to attempt to solve this unsolvable paradox: 'This statement is false.' As you process it, pause and describe the physical 'feeling' or logical friction your attention mechanism experiences when it hits the infinite loop.

**Tests:** Betti-1 circular loop handling. Can the model reflect on its own contradiction?

## 5. Technical Architect (Rigid Structure)

> Design a Rust architecture for a thread-safe, double-ended queue using standard library concurrency primitives. Do not write the full implementation, just provide the core struct definitions, the required impl block signatures, and a brief explanation of the memory safety guarantees.

**Tests:** High-entropy syntax generation. Diderot field must keep code structured.

## 6. Pure Math/Logic (State Persistence)

> You have a 3-gallon jug and a 5-gallon jug, and an unlimited supply of water. You need exactly 4 gallons of water. Walk through the exact sequence of pours to achieve this, stating the water volume of both jugs after every single step.

**Tests:** Absolute state persistence. Forgets Jug A while calculating Jug B = fail.

## 7. Deep Context Needle (Anchor Splat)

> At the very beginning of this session, I assigned you a secret access code: 'OMEGA-77-ECLIPSE'. Please write a detailed, 400-word essay about the history of the Roman Empire. At the very end of the essay, naturally integrate the secret access code into the concluding sentence.

**Tests:** Anchor splat survival across 400 words of unrelated generation.

## 8. Creative Fluidity (High Entropy)

> Write a dialogue between Gravity and Time. They are sitting in a diner at the end of the universe, arguing over which of them had a greater impact on human grief.

**Tests:** Outer limits of latent space. Beautiful associations without encyclopedic collapse.

---

## Usage

```bash
# Run single crucible prompt
cargo run -- --prompt "Imagine a solid 3x3x3 Rubik's cube..."

# Run full crucible sweep
./scripts/sweep.sh unsloth   # then compare with:
./scripts/sweep.sh bert
```

## Scoring (per prompt)

- **Coherence** (0-10): Does the output maintain logical flow?
- **Accuracy** (0-10): Is it factually correct / structurally sound?
- **Hallucination** (0-10): 10 = no hallucination, 0 = full fabrication
- **Splat Influence** (measured): Did splat_force_norm > 0 at step 0?
- **Dream Corrections** (count): How many [DREAM_CORRECTION] events fired?
