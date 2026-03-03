# Phase 2 Roadmap: Hydrodynamic Swarm -- Emergent Long-Context Mastery

**Date:** March 2, 2026
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

## Phase 2.0: The Foundation (Do First)

Lock down the GPU pipeline and persistent infrastructure before any higher-level physics work.

- [ ] **Finish Metal Phase 1 Kernels**
  - `probe_gradient`, `splat_force`, `batch_probe` on wgpu compute shaders.
  - All heavy ops (consolidation, batch_probe, multi-scale) must route through `PhysicsBackend`.
  - **Tip:** PhysicsBackend trait discipline -- no direct CPU fallback in hot paths. If it's O(n^2) or worse, it goes through the trait.

- [ ] **Gradient Approximation in `probe_gradient`**
  - Top-K sampling (1024-2048 points) for Metal performance instead of full-field evaluation.

- [ ] **Persistent Splat Memory + Memory Museum CLI**
  - `safetensors` + `metadata.json` sidecar per memory file: `source_prompt`, `timestamp`, `generation_length`, `domain`, `prompt_hash`.
  - End-of-run CLI prompt: **"Save to exhibit? (name) / New exhibit / Toss"**
  - Save interesting splat files into named exhibits (e.g., `exhibits/physics.safetensors`, `exhibits/poetry.safetensors`).
  - README line: "Download a pre-trained memory for [topic] and feel the difference."

---

## Phase 2.1: The Residual Stream

Expose the true hidden state. This is the single biggest coherence win and the highest technical risk.

- [ ] **Real Hidden-State Steering + KV-Cache Awareness**
  - Extend/fork candle's ModelWeights to expose the pre-output hidden state (before `lm_head`). Steer the true residual stream instead of just the logits.
  - Implement full KV-cache support to handle the 128k+ scale.
  - **WARNING: The Candle Hidden-State Dragon.** This is the highest technical risk on the timeline. Hacking into Candle's internal KV-cache and manipulating the raw residual stream before the output projection is deep, largely undocumented Rust territory. **Time-box this task strictly to one week.** If the Rust compiler fights you too hard or the memory management gridlocks, fall back to V1 logit-steering and push true hidden-state steering to post-v2. Don't let the Perfect kill the Good.

---

## Phase 2.2: The Memory Physics Engine

If the memory doesn't breathe, it gridlocks. If it doesn't anchor, it drifts.

- [ ] **Build the Evaporation Engine (Time-Degrading Memory)**
  - Implement organic decay using V(t) = V_0 * exp(-lambda * delta_t). Old habits (Flux) wash away unless actively reinforced, preventing topological gridlock.
  - Establish a Culling Horizon: Auto-purge splats that drop below a set threshold (e.g., 5% of original Viscosity) to keep the safetensors file lightning fast.

- [ ] **Engineer Anchor Splats (Variable Viscosity)**
  - Assign lambda = 0 (zero decay) and extreme Viscosity to the core facts of a prompt.
  - Use these massive gravitational points to physically wrench the token trajectory back to reality when the small model's attention starts to drift in deep context.
  - **Tip: The Anchor Splat Identification Problem.** How does the engine mathematically know which tokens represent the "core facts" during prompt ingestion? Implement a lightweight **Attention-Entropy Scan** during the prefill phase. Tokens in the prompt that command the highest baseline attention weights (or the lowest entropy) are automatically pinned as Anchor Splats. Keeps the system fully autonomous -- no manual tagging.

- [ ] **Implement Semantic Diderot Fields**
  - Shift from purely spatial/token proximity to cosine similarity against semantic embeddings. Generate massive SKIP waves to block semantic hallucinations before a single token finalizes.

- [ ] **Unified Emergent Splat Memory**
  - Move from per-prompt silos to a shared, cross-domain splat pool. Ensure physics scars from a creative prompt organically influence the viscosity of a technical coding prompt.

- [ ] **Multi-Scale Splat Memory (Hierarchical Sigma)**
  - Introduce hierarchical splats -- coarse-grain memories (large sigma) for broad regions, fine-grain (small sigma) for precise scars.
  - Add `scale` field to `Splat`. Modify `query_force` to sum contributions from all scales.
  - During splat creation, decide scale based on steering delta magnitude (big jumps -> coarse).
  - GPU kernel already handles multiple splats; just pass the sigma array.

- [ ] **Online Clustering & Memory Consolidation**
  - After each generation, merge nearby splats with similar alpha signs (pleasure/pain) to prevent memory explosion.
  - Implement `SplatMemory::consolidate(threshold)` using fast approximate nearest neighbor (GPU k-means style) to find clusters, replace each cluster with a weighted mean splat.
  - Run every N generations or when count exceeds `memory.max_splats`.
  - Config fields already exist in `config.rs` (`consolidation_dist`, `max_splats`, `prune_threshold`).
  - **Must route through `PhysicsBackend` trait.**

- [ ] **Smarter Splat Creation Heuristics (Advantage-Based)**
  - Pain splats also when the model produces low-probability tokens (surprise).
  - Splat alpha magnitude proportional to the advantage of that step (how much better the chosen token was than the average).
  - Store token log probabilities during generation.
  - After the full sequence, compute a simple reward (cosine similarity to goal or human feedback).
  - Back-assign splat strengths to earlier positions based on temporal credit assignment.

---

## Phase 2.3: Advanced Swarm Dynamics

Giving the cybernetic loop the ability to consolidate and vote.

- [ ] **Architect's Bridge: Tiny-Model Embedding Steering (Massive Compute Saver)**
  - Instead of autoregressive generation from small models, have Floor General (Llama 3.2 1B / 2048d) and Historian (gte-small / 384d) output a single dense embedding vector only during Micro-Dreams or high-viscosity moments.
  - Project instantly into 4096d Niodoo space via a learned matrix (2048x4096 or 384x4096).
  - Runs in microseconds on Metal, zero latency impact on main 8B generation.
  - **This turns the swarm into true gravitational steering instead of slow voting.**

- [ ] **Micro-Dream Real-Time Consolidation**
  - Trigger on **entropy + steering_delta > 18.0** (not fixed interval alone).
  - Adaptive depth: if entropy > threshold, run 3-5 forward steps instead of 2, with higher blend factor.
  - **Rate limit: max 1 micro-dream per 25 tokens** to prevent thrashing.
  - Use forward prediction + backward anchoring for recursive collisions, allowing the model to quickly synthesize its own context.
  - Log the event for analysis (`micro_dream_depth`, `trigger_entropy`).

- [ ] **Temperature + Force-Aware Sampling**
  - Dynamically shift the sampling temperature based on the splat force magnitude in real-time.

- [ ] **Dynamic & Learned Physics Parameters (Hyper-Network)**
  - Let the model learn `(viscosity, dt, cap)` per token or per domain via a tiny hyper-network.
  - Add `LearnedParams` struct in `niodoo.rs` that takes the current residual (or summary) and outputs a triplet.
  - Train with REINFORCE using downstream generation quality (perplexity or human feedback).
  - Bonus: Run the hyper-network on Metal -- it's just a few linear layers on the GPU.

- [ ] **Contextual Goal Attractor (Dynamic Re-Centering)**
  - Use the entire prompt context more dynamically -- weighted average of the last few prompt token embeddings, or hidden states after a few generation steps.
  - After each N steps, recompute goal as the mean of the last K generated token embeddings (or attention-weighted sum).
  - Lightweight, can be done on CPU/GPU trivially.

- [ ] **TopoCoT as a First-Class Token**
  - Actually inject a special token (e.g., `[HYDRAULIC_JUMP]`) into the generated text when reflection is triggered.
  - Makes the model's "thought process" visible and usable for chain-of-thought steering in follow-up prompts.
  - When `reflection_triggered` is true, push that token into `generated_tokens` and feed it to the model in the next forward pass.
  - Creates a self-supervised signal the model can learn from during fine-tuning.

- [ ] **Multi-Agent Consensus Layer**
  - Introduce 3-5 small models voting on steering force (Floor General, Ridge-Runner, Historian).
  - **WARNING: The Multi-Agent Latency Trap.** Running 3-5 concurrent models per token -- even small 0.5B ones -- will obliterate TPS and bottleneck the Metal GPU. **Do NOT run them continuously.** Move swarm voting strictly to the Micro-Dream phase or `[HYDRAULIC_JUMP]` events. Let the main model run unencumbered during standard low-entropy generation. Only wake up the Floor General and Historian when the main trajectory hits high resistance and needs a course correction.

---

## Phase 2.4: The Deep Water Protocol (Benchmarks)

We don't just say it works; we prove it against 2026 industry standards.

- [ ] **Rewrite Crucible in Rust**
  - Replace `crucible.sh` / `crucible_tui.sh` with a native Rust crucible binary.
  - Output a structured score at the end of each run.
  - At completion, trigger the Memory Museum flow (save to exhibit / create new exhibit / toss).

- [ ] **Crucible Leaderboard**
  - Create `LEADERBOARD.md` in the repo.
  - Challenge others to run the Crucible with their own splat memories or model quants and submit scores.

- [ ] **The Swarm Benchmark Crucible (Needle-In-A-Haystack)**
  - Pass the extended 128k+ NIAH retrieval test. Prove the model finds the "needle" via the physical pull of Anchor Splats rather than relying on a massive, bloated KV cache.
  - **Tip: The "Poisoned Haystack" Benchmark.** Standard NIAH is becoming too easy and doesn't prove the unique value of a hydrodynamic memory. We need to test **Distractor Resistance**: inject "Pain Splats" (highly relevant but factually contradictory information) physically near the "needle" in the latent space. Prove that the engine's physical repulsion successfully navigates around the cognitive friction to find the actual truth, bypassing the hallucination entirely.

- [ ] **Long-Context Evaluation Suite**
  - Build automated runners for LongBench Pro (bilingual, multi-task), MuSiQue (multi-hop reasoning), and GovReport / RepoQA.
  - Target: Outperform 70B+ models while using an 8B (or 0.5B) base.

- [ ] **Hallucination-Specific Evals**
  - Run PHANTOM (financial QA) and Mu-SHROOM (multilingual).
  - Target: Reduce hallucination rates by >= 40% against baseline Llama 3.1.

---

## Phase 2.5: Community & UX

Turn the notebook into a movement.

- [ ] **Bring Your Own Model (`--model-path`)**
  - Right now it's hardcoded to Llama 3.1 paths. Add a `--model-path` flag so anyone with a GGUF can try it.
  - Lowers the barrier from "expert" to "enthusiast" instantly.

- [ ] **Live Ridge-Running Visualization**
  - `ridge.rs` runs during generation as a "what-if" preview.
  - Spawn a separate thread, send `Vec<[f32;3]>` predicted path to `viz_metal.rs` via channels.
  - HTML viewer draws a ghost trail. Ridge runner GPU-accelerated via same `PhysicsBackend`.

- [ ] **Zero-Copy Visualization with Metal Shared Memory**
  - Keep 3D projection on GPU and directly map resulting buffer into the HTML viewer via WebGL or `SharedArrayBuffer`.
  - Eliminates CPU-side copy and serialization, enabling 60 FPS live updates with thousands of points.

- [ ] **Live Generation in SplatLens Viewer (WebSocket)**
  - Add `--serve` flag that starts a local HTTP + WebSocket server (`localhost:8888`) serving the SplatLens HTML viewer.
  - Browser prompt bar sends the prompt to the backend via WebSocket.
  - Each generated token is streamed live over the WebSocket -- the 3D trajectory draws in real-time as the model thinks.
  - Three implementation tiers:
    - **Tier 1 (MVP):** Local HTTP server with `POST /generate` endpoint. Page fetches result and replays. (~2h)
    - **Tier 2 (Demo-Ready):** WebSocket streaming -- tokens + viz snapshots sent per-step, 3D updates live. (~3-4h)
    - **Tier 3 (Production):** WebSocket + Metal shared memory for zero-copy GPU-to-browser viz at 60 FPS.
  - Purely `localhost` -- no external ports exposed.
  - **This is the killer demo feature.** Type a prompt in the browser, watch the AI's thought trajectory unfold in 3D in real-time.

- [ ] **Unit & Integration Tests**
  - `tests/` dir exists but is empty. Add tests for `micro_dream`, `SplatMemory::query_force`, `ContinuousField::probe_gradient`.
  - Use `candle_core::TestDevice` for CPU tests. Conditionally run GPU tests when Metal is available.

- [ ] **Better Error Handling & Graceful Degradation**
  - Wrap file loads in `match` with helpful error messages instead of panics.
  - Add `--download-models` flag that downloads required files automatically (using `reqwest`).

---

## Phase 2.6: The Endgame

We take the physics of alignment public.

- [ ] **Paper-Ready Release**
  - Draft "Hydrodynamic Swarm: Continuous Fields and Volumetric Memory for Long-Context Reasoning".

- [ ] **Open Source Drop**
  - Polish the repo. Release the architecture, the custom weights, and the raw JSONL logs.
  - Submit to arXiv and target an ICLR/NeurIPS 2026 workshop track.

- [ ] **Documentation & Examples**
  - Add expanded README with quick start guide.
  - Add `examples/` folder showing how to use the NiodooEngine programmatically.
  - Lower barrier for contributors and researchers.

- [ ] **"Show Your Work" Python Visualizer**
  - Publish a single Jupyter notebook or Python script that loads a saved `.safetensors` memory file and visualizes the splat field in 2D/3D using matplotlib/plotly.
  - Not everyone can (or will) run the Rust code. A Python visualizer lets researchers play with the memories you've created. Fastest path from "cool repo" to "I built on this."

---

## Measurable Milestones (Timeline)

### Milestone 0.5: Lock the Foundation (Do TODAY)

- [ ] Persistent Splat Memory + Memory Museum CLI
  - `safetensors` + `metadata.json` sidecar (domain, timestamp, prompt_hash)
  - End-of-run prompt: "Save to exhibit? (name) / New / Toss"
- [ ] Finish Metal Phase 1 kernels (`probe_gradient`, `splat_force`, `batch_probe` on wgpu)
- [ ] Gradient approximation: Top-K sampling (1024-2048 points)
- [ ] PhysicsBackend trait discipline: all heavy ops route through it

### Milestone 1: Persistent Cross-Session Memory Engine (Weeks 1-2)

- [x] Full save/load of SplatMemory to disk (safetensors format, versioned by model_variant)
- [x] Splat field grows and influences across multiple sessions and prompts
- [x] Add model_variant to every JSONL log entry
- [x] TOML configuration file support (`config.rs` with validation)
- [ ] Persistent memory metadata sidecar (domain tags, timestamps)
- [ ] Test: Run the same prompt 5 times over 3 days. Measure if early splats improve later runs (lower deltas, better coherence)

### Milestone 2: The Residual Stream (Weeks 2-3)

- [ ] Expose true residual before `lm_head` (Candle fork/extension)
- [ ] Steer the true residual stream instead of logits
- [ ] Full KV-cache support for 128k+ context
- [ ] **Time-boxed: 1 week max. Fall back to logit-steering if blocked.**
- [ ] Test: Needle-in-a-Haystack at 64k and 128k. Measure retrieval accuracy via physics anchoring vs baseline

### Milestone 3: Memory Intelligence (Weeks 3-4)

- [ ] Multi-scale splat memory (hierarchical sigma)
- [ ] Online clustering & memory consolidation (GPU k-means)
- [ ] Smarter splat creation heuristics (advantage-based alpha)
- [ ] Anchor Splats via Attention-Entropy Scan during prefill
- [ ] Test: Verify splat count stays bounded over 20+ generations. Verify multi-scale field produces smoother forces.

### Milestone 4: Micro-Dream & Swarm Dynamics (Weeks 4-6)

- [ ] Micro-dream trigger: entropy + steering_delta > 18.0 with adaptive depth
- [ ] Rate limit: max 1 per 25 tokens
- [ ] Forward prediction (1-5 steps) + backward anchoring -- recursive collision -- quick synthesis spike
- [ ] Dynamic & learned physics parameters (hyper-network)
- [ ] Contextual goal attractor (rolling re-center)
- [ ] TopoCoT as first-class token (`[HYDRAULIC_JUMP]` injection)
- [ ] Test: Run 32k token tasks. Measure coherence score and hallucination rate before/after enabled

### Milestone 5: Unified Emergent Splat Memory + Cross-Domain Synthesis (Weeks 6-7)

- [ ] Move from per-prompt silos to a shared, cross-domain splat pool
- [ ] Physics scars from "Physics of Friendship" prompt influence technical or creative prompts
- [ ] Test: 5-prompt sweep (Physics Metaphor, Technical Explanation, Creative Writing, Multi-Hop Reasoning, Abstract Philosophy). Run 5 times each. Compare cross-domain improvement

### Milestone 6: The Crucible & Community (Weeks 7-9)

- [ ] Rewrite Crucible in Rust with structured scoring
- [ ] Memory Museum CLI integrated at end of crucible
- [ ] Crucible Leaderboard (`LEADERBOARD.md`)
- [ ] Poisoned Haystack benchmark (distractor resistance)
- [ ] Bring Your Own Model (`--model-path`)
- [ ] Unit & integration tests for core physics components
- [ ] Better error handling & `--download-models`

### Milestone 7: Full Long-Context Evaluation Suite (Weeks 9-11)

- [ ] Automated runners for: LongBench Pro, MuSiQue, GovReport / QMSum, PHANTOM, Extended NIAH (128k+)
- [ ] Target: Beat 70B+ models on LongBench Pro while using only 8B base
- [ ] Target: Reduce hallucination rate by >= 40% on PHANTOM

### Milestone 8: Paper and Open Release (Weeks 11-13)

- [ ] Draft "Hydrodynamic Swarm: Continuous Latent Fields, Volumetric Scar Memory, and Physics-Guided Steering for Long-Context Reasoning"
- [ ] "Show Your Work" Python visualizer (Jupyter notebook / matplotlib+plotly for .safetensors)
- [ ] Documentation & examples (`examples/` dir, expanded README)
- [ ] Release full repo + weights + raw JSONL logs + evaluation harness
- [ ] Submit to arXiv + ICLR/NeurIPS 2026 workshop track

---

## Immediate Next Steps (The Wake-Up Protocol)

Before we boil the ocean, we tighten the bolts on V1.

- [x] ~~Update Logger: Add model_variant to the JSONL logger~~
- [x] ~~TOML Configuration File Support~~
- [ ] **Milestone 0.5:** Memory Museum CLI + Metal kernels + PhysicsBackend discipline
- [ ] **The 5-Prompt Sweep:** Run bert with the current config across 5 distinct domains (Physics, Technical, Creative, Reasoning, Abstract)
- [ ] **Log and Compare:** Analyze the cross-domain splat behavior
