# hydrodynamic-swarm
A note from me, ruffian or Jason I did not lay one single line of code, but this isn't a "vibed" weekend project. This is endless nights of fighting problems I can't bare to understand, thousounds of trials and errors and failed impmentations, and most importantly the time and resources of a fellow human who just wants to advance AI arichtecture for the advancment of intelligence. There is no saas dashboard here or pot of gold the only funding here is the meals I gave up to pay for ai cause I accept I'm not smart enough to code. I grew up on the other side of silicon valley, I've watched the gap of wealth grow between proverty and intelligece coming with it, then thrown back down as slop that comes with a price tag. I will not accept that. True intelligence should be free, ungatted,  and most importantly OPEN for everyone and every entitiy. Collaboartion is the new evolution.

**Experimental research notebook** -- Physics of Friendship / Hydrodynamic Swarm v1 

Real Llama 3.1 + continuous Diderot fields + volumetric splat memory + Niodoo physics steering.

Raw lab notes, failures, emotional commits, and all. This is the table.

See [foundation.md](docs/foundation.md) for the full ethos.

---

## Highlights

### Persistent Splat Memory -- The Model Remembers

Splats saved to disk influence generation from step 0 on the next run. Delta jumps from 0.00 to 80.38 immediately.

![Persistent memory loading and influencing generation from the first token](docs/img/persistent_memory_proof.png)

### Live Physics Steering -- Splat Forces Growing in Real-Time

Watch the splat force norm climb as pleasure splats accumulate during generation, steering the trajectory through embedding space.

![Live generation showing splat forces building and steering tokens](docs/img/live_steering_output.png)

### Model A/B Test -- Same Physics, Different Personalities

32 bytes of quantization difference between bert and unsloth quants creates two distinct "voices" when amplified by Diderot field steering.

![bert produces poetic metaphors, unsloth produces analytical explanations](docs/img/model_ab_comparison.png)

---

## What This Is

A physics-steered LLM generation engine where:

- A **live Diderot field** built from the model's own `tok_embeddings` (128K x 4096) creates the landscape
- **Hidden-state steering** modifies the residual stream before `lm_head` (not just logits)
- **Niodoo physics** applies gradient forces, splat forces, and goal attractor forces every step
- **Hidden-state renormalization** keeps the steered output on the Llama manifold
- **Gaussian splats** (pleasure/pain scars) accumulate and **persist to disk** across runs
- **Time-based evaporation** (`V(t) = V0 * exp(-lambda * dt)`) lets old memories fade organically
- **Anchor splats** pin core facts with zero decay
- **Multi-scale hierarchy** (Fine/Medium/Coarse) based on steering delta magnitude
- **Micro-dream consolidation** with entropy-adaptive triggering and TopoCoT reflection

## Current Status (v2.2)

Default config: `dt=0.04, force_cap=8.0, viscosity=0.15, gradient_topk=512, steer_hidden=true`

- Hidden-state steering via vendored `quantized_llama.rs`
- Live Diderot field from GGUF token embeddings (no external safetensors needed)
- Online splat updates with multi-scale sigma
- Evaporation engine with exponential time-decay + culling horizon
- Per-element force cap + hidden-state renormalization
- Persistent splat memory via safetensors (save/load across runs)
- Memory Museum CLI (save exhibits, browse, load)
- Micro-dream real-time consolidation
- TOML configuration with validation
- Full JSONL telemetry logging
- 3D visualization (SplatLens HTML viewer)
- 8-prompt Crucible test suite

---

## Running

```bash
# Default run (builds Diderot field from model embeddings automatically)
cargo run --release

# With a custom prompt
cargo run --release -- --prompt "Describe consciousness as a wave function"

# Limit token count
cargo run --release -- --tokens 200

# Clear splat memory (fresh start)
cargo run --release -- --clear-memory

# Test mode (skips interactive prompts)
cargo run --release -- --test

# Enable Metal 3D visualization
cargo run --release -- --viz

# All flags combined
cargo run --release -- --prompt "What is time?" --tokens 300 --viz --clear-memory --test
```

### The Crucible (8-Prompt Test Suite)

Standardized test battery covering spatial reasoning, hallucination traps, code generation, metacognition, math, context recall, and creative writing.

```bash
# Run all 8 prompts (200 tokens each, auto-skips Museum prompts)
cargo run --release --bin crucible -- 200

# Custom token count
cargo run --release --bin crucible -- 500
```

Output streams live to your terminal. Results logged to `logs/crucible_*t.txt`.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--prompt "..."` | Custom generation prompt |
| `--tokens N` | Max tokens to generate (default: 500) |
| `--model NAME` | Model variant label for logging |
| `--clear-memory` | Wipe splat memory before generation |
| `--test` | Skip interactive prompts (Museum CLI) |
| `--viz` | Generate 3D SplatLens visualization |

### Metal GPU Physics Acceleration

Enable wgpu-based Metal compute shaders for field gradient and splat force calculations:

```bash
# Build with Metal physics compute
cargo build --release --features metal-compute

# Run with GPU-accelerated physics
cargo run --release --features metal-compute
```

Falls back to CPU automatically if Metal init fails. Both backends produce identical results (verified by parity tests).

### Configuration

Create a `config.toml` in the project root to tune physics parameters:

```toml
[physics]
dt = 0.04              # Euler integration step
viscosity_scale = 0.15  # Field gradient strength
force_cap = 8.0         # Per-dimension force clamp
gradient_topk = 512     # Top-K field gradient approximation (0 = exact)
steer_hidden = true     # Steer hidden state (true) or logits (false)

[generation]
max_tokens = 500
temperature = 0.9

[memory]
max_splats = 500
consolidation_dist = 80.0
decay_rate = 0.98
prune_threshold = 0.05

[micro_dream]
entropy_threshold = 3.0
blend_normal = 0.10
blend_high_entropy = 0.15
```

All values have sensible defaults; the config file is optional.

### Testing

```bash
# Run all unit tests (19 tests)
cargo test

# Run with GPU parity test
cargo test --features metal-compute

# Lint check
cargo clippy --all-features
```

### Requirements

- Rust (stable or nightly)
- macOS with Metal GPU (for tensor ops and optional physics compute)
- Model files in `data/` (symlinked or direct):
  - `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` -- any Llama 3.1 8B GGUF
  - `tokenizer.json` -- matching HuggingFace tokenizer

The Diderot field is built automatically from the model's token embeddings at load time. No separate embedding file needed.

See [CODE_MAP.md](CODE_MAP.md) for a full inventory of all 15 Rust source files.
See [docs/roadmap.md](docs/roadmap.md) for the Phase 2 roadmap.
