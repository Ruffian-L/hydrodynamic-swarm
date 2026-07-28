# Telemetry & Physics Vocabulary

**Date:** 2026-07-11  
**Context:** Quick reference for reading hydrodynamic-swarm generation logs and telemetry.

---

## Steering forces

| Term | Technical meaning | In practice |
|------|-------------------|-------------|
| **F_g** | Field gradient force (∇ρ from Diderot Field) | How strongly the token embedding cloud pulls on the current hidden state. Was ~0 before the wake; now ~35 when the wake is active. |
| **F_s** | Splat force (from current PLEASURE/PAIN memory particles) | How much memory splats deposited earlier are shaping generation right now. High = memory is actively interfering. |
| **F_a** | Goal attractor force (from prompt prefill hidden state) | How strongly the original prompt still pulls generation. Currently capped at 50 and often dominates. |
| **F_ocean** | Shared Ocean force (packets deposited every 4 steps) | Background memory/context injected from the persistent ocean. Lower noise = cleaner background signal. |
| **Goal attractor norm** | Size/strength of the initial prompt hidden state vector | ~430–460 in current runs. The big anchor the model starts from. If too strong, it can overpower everything else. |

---

## Trajectory & stability

| Term | Technical meaning | In practice |
|------|-------------------|-------------|
| **δ (delta)** | Deviation / drift of the hidden state trajectory | How far the current hidden state has moved from where the physics layer wants it. High δ often precedes Pain events. |
| **REFLEX** | Emergency corrective blend triggered by VR H1 collapse | Safety override. When the hidden-state trajectory becomes too unstable, it forcibly blends in a correction. Usually too late. |

---

## Memory events

| Term | Technical meaning | In practice |
|------|-------------------|-------------|
| **SPLAT Pleasure** | Positive memory deposit triggered during generation | The system decided this token/region was "good" and stored a splat. Usually early in runs when things are coherent. |
| **SPLAT Pain** | Negative / corrective memory event | The system detected chaos, high entropy, or bad trajectory and deposited a "pain" splat. Often followed by garble. |
| **Dream replay** | Post-generation memory consolidation step | Takes the 200 generated points and turns them into a smaller number of splats. Decay is then applied. |

---

## Field geometry & wake

| Term | Technical meaning | In practice |
|------|-------------------|-------------|
| **Diderot Field (ρ)** | Gaussian mixture density built from all token embeddings | The "map" of the model's vocabulary in high-dimensional space. Field audit showed a thin shell with huge dead zones outside it. |
| **σ (sigma)** | Width of each Gaussian in the Diderot Field | Currently ~11. Controls how far field influence reaches. Too small = dead outside the cloud. Too big = blurry attractors. |
| **Field wake (k=1)** | Nearest-embedding wake mode | Instead of pure math gradient (which was near-zero outside the embedding shell), pull toward the single closest token embedding. This is what made F_g non-zero and usable. |
| **field_wake_max** | Hard cap on how strong the wake force can get | Currently 40. If too high it can yank the hidden state too violently → late-run instability. |

---

## Shared Ocean

| Term | Technical meaning | In practice |
|------|-------------------|-------------|
| **Shared Ocean** | Persistent background memory layer (deposits every 4 tokens) | Long-term context that slowly accumulates. Currently secondary to the goal attractor. |
| **ocean_n** | Number of ocean packets currently active | How much background memory is being injected right now. |
| **mean_noise** | Average noise level in the ocean deposits | Lower = cleaner, more stable background memory signal. |