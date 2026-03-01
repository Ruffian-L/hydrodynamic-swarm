# Phase 2 Roadmap: Hydrodynamic Swarm -- Emergent Long-Context Mastery

**Date:** March 1, 2026
**Status:** v1 Complete (Continuous Diderot field + persistent splat memory + Niodoo residual steering online)
**Phase 2 Vision:** Evolve the V1 token steering engine into a self-correcting long-context system (32k-128k+ tokens) that uses fluid dynamics to beat 70B+ models on hallucination and retention benchmarks using only sub-billion/8B parameters.

---

## Core Problems We Solve in Phase 2

- **Hidden State Drift** -- Small models lose coherence after ~8k-16k tokens because there is no mechanism to "feel" when the trajectory is leaving stable ridges.
- **Memory Interference** -- Old context overwrites new; no persistent spatial memory of what worked.
- **Lack of Self-Correction** -- No real-time consolidation during generation (only full offline dream replay).
- **Hallucination in Long Context** -- No physical repulsion from past failure scars.

**Our advantage:** We already have the physics primitives. Phase 2 scales them.

---

## Phase 2.1: The Memory Physics Engine

If the memory doesn't breathe, it gridlocks. If it doesn't anchor, it drifts.

- [ ] **Build the Evaporation Engine (Time-Degrading Memory)**
  - Implement organic decay using V(t) = V_0 * exp(-lambda * delta_t). Old habits (Flux) wash away unless actively reinforced, preventing topological gridlock.
  - Establish a Culling Horizon: Auto-purge splats that drop below a set threshold (e.g., 5% of original Viscosity) to keep the safetensors file lightning fast.

- [ ] **Engineer Anchor Splats (Variable Viscosity)**
  - Assign lambda = 0 (zero decay) and extreme Viscosity to the core facts of a prompt.
  - Use these massive gravitational points to physically wrench the token trajectory back to reality when the small model's attention starts to drift in deep context.

- [ ] **Implement Semantic Diderot Fields**
  - Shift from purely spatial/token proximity to cosine similarity against semantic embeddings. Generate massive SKIP waves to block semantic hallucinations before a single token finalizes.

- [ ] **Unified Emergent Splat Memory**
  - Move from per-prompt silos to a shared, cross-domain splat pool. Ensure physics scars from a creative prompt organically influence the viscosity of a technical coding prompt.

---

## Phase 2.2: Advanced Swarm Dynamics

Giving the cybernetic loop the ability to consolidate and vote.

- [ ] **Micro-Dream Real-Time Consolidation**
  - Trigger short, high-speed physics bursts during natural pauses (low-entropy steps or specific tokens).
  - Use forward prediction + backward anchoring for recursive collisions, allowing the model to quickly synthesize its own context.

- [ ] **Real Hidden-State Steering + KV-Cache Awareness**
  - Extend candle's ModelWeights to expose the pre-output hidden state. Steer the true residual stream instead of just the logits.
  - Implement full KV-cache support to handle the 128k+ scale.

- [ ] **Temperature + Force-Aware Sampling**
  - Dynamically shift the sampling temperature based on the splat force magnitude in real-time.

- [ ] **Multi-Agent Consensus Layer**
  - Introduce 3-5 small models voting on steering force (Floor General, Ridge-Runner, Historian).

---

## Phase 2.3: The Deep Water Protocol (Benchmarks)

We don't just say it works; we prove it against 2026 industry standards.

- [ ] **The Swarm Benchmark Crucible (Needle-In-A-Haystack)**
  - Pass the extended 128k+ NIAH retrieval test. Prove the model finds the "needle" via the physical pull of Anchor Splats rather than relying on a massive, bloated KV cache.

- [ ] **Long-Context Evaluation Suite**
  - Build automated runners for LongBench Pro (bilingual, multi-task), MuSiQue (multi-hop reasoning), and GovReport / RepoQA.
  - Target: Outperform 70B+ models while using an 8B (or 0.5B) base.

- [ ] **Hallucination-Specific Evals**
  - Run PHANTOM (financial QA) and Mu-SHROOM (multilingual).
  - Target: Reduce hallucination rates by >= 40% against baseline Llama 3.1.

---

## Phase 2.4: The Endgame

We take the physics of alignment public.

- [ ] **Paper-Ready Release**
  - Draft "Hydrodynamic Swarm: Continuous Fields and Volumetric Memory for Long-Context Reasoning".

- [ ] **Open Source Drop**
  - Polish the repo. Release the architecture, the custom weights, and the raw JSONL logs.
  - Submit to arXiv and target an ICLR/NeurIPS 2026 workshop track.

---

## Measurable Milestones (Timeline)

### Milestone 1: Persistent Cross-Session Memory Engine (Weeks 1-2)

- [x] Full save/load of SplatMemory to disk (safetensors format, versioned by model_variant)
- [x] Splat field grows and influences across multiple sessions and prompts
- [x] Add model_variant to every JSONL log entry
- [ ] Test: Run the same prompt 5 times over 3 days. Measure if early splats improve later runs (lower deltas, better coherence)

### Milestone 2: Micro-Dream Real-Time Consolidation (Weeks 2-3)

- [ ] Trigger short, high-speed physics bursts during natural pauses (low-entropy steps, steering_delta > 18.0)
- [ ] Forward prediction (1-3 steps) + backward anchoring to prompt goal -- recursive collision -- quick synthesis spike
- [ ] Test: Run 32k token tasks. Measure coherence score and hallucination rate before/after micro-dreams enabled

### Milestone 3: Unified Emergent Splat Memory + Cross-Domain Synthesis (Weeks 3-4)

- [ ] Move from per-prompt silos to a shared, cross-domain splat pool
- [ ] Physics scars from "Physics of Friendship" prompt influence technical or creative prompts
- [ ] Test: 5-prompt sweep (Physics Metaphor, Technical Explanation, Creative Writing, Multi-Hop Reasoning, Abstract Philosophy). Run 5 times each. Compare cross-domain improvement

### Milestone 4: Real Hidden-State Steering + Full KV-Cache Awareness (Weeks 4-6)

- [ ] Extend/fork candle's ModelWeights to expose pre-output-projection hidden state
- [ ] Steer the true residual stream instead of logits
- [ ] Full KV-cache support for 128k+ context
- [ ] Test: Needle-in-a-Haystack at 64k and 128k. Measure retrieval accuracy via physics anchoring vs baseline

### Milestone 5: Full Long-Context Evaluation Suite (Weeks 6-8)

- [ ] Automated runners for: LongBench Pro, MuSiQue, GovReport / QMSum, PHANTOM, Extended NIAH (128k+)
- [ ] Target: Beat 70B+ models on LongBench Pro while using only 8B base
- [ ] Target: Reduce hallucination rate by >= 40% on PHANTOM

### Milestone 6: Paper and Open Release (Weeks 8-10)

- [ ] Draft "Hydrodynamic Swarm: Continuous Latent Fields, Volumetric Scar Memory, and Physics-Guided Steering for Long-Context Reasoning"
- [ ] Release full repo + weights + raw JSONL logs + evaluation harness
- [ ] Submit to arXiv + ICLR/NeurIPS 2026 workshop track

---

## Immediate Next Steps (The Wake-Up Protocol)

Before we boil the ocean, we tighten the bolts on V1.

- [x] ~~Update Logger: Add model_variant to the JSONL logger~~
- [ ] **The 5-Prompt Sweep:** Run bert with the current config across 5 distinct domains (Physics, Technical, Creative, Reasoning, Abstract)
- [ ] **Log and Compare:** Analyze the cross-domain splat behavior
