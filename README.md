# hydrodynamic-swarm active development and research

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

- A **continuous Diderot field** (128256 x 4096) creates a landscape the model traverses
- **Niodoo physics** applies gradient forces, splat forces, and goal attractor forces to the residual stream every step
- **Gaussian splats** (pleasure/pain scars) accumulate during generation and **persist to disk** across runs
- The model develops **spatial memory** of its own generation history
- Different model quants (bert vs unsloth) produce distinct "personalities" under identical steering

## Current Status (v1.1)

Config: `sigma=150, alpha=2.0, force_cap=80, T=0.9, min_dist=100`

- Online splat updates during generation
- Per-element force cap prevents runaway
- Min distance check prevents splat stacking
- Temperature sampling enables creative divergence
- Persistent splat memory via safetensors (save/load across runs)
- Micro-dream real-time consolidation (forward projection + backward anchoring)
- 5-prompt evaluation sweep (`./scripts/sweep.sh [model]`)
- CLI flags: `--prompt`, `--model`, `--clear-memory`, `--chat`
- Full JSONL telemetry logging with experiment metadata

See [experiments.md](docs/experiments.md) for detailed findings.

## Running

```bash
# Default run (CPU physics, Metal tensor ops via candle)
cargo run

# With a custom prompt
cargo run -- --prompt "Describe consciousness as a wave function"

# Interactive Chat TUI mode
cargo run -- --chat

# Limit token count
cargo run -- --tokens 200

# Clear splat memory (fresh start)
cargo run -- --clear-memory

# Enable Metal 3D visualization
cargo run -- --viz

# All flags combined
cargo run -- --prompt "What is time?" --tokens 300 --viz --clear-memory
```

### Metal GPU Physics Acceleration

Enable wgpu-based Metal compute shaders for field gradient and splat force calculations:

```bash
# Build with Metal physics compute
cargo build --features metal-compute

# Run with GPU-accelerated physics
cargo run --features metal-compute
```

Falls back to CPU automatically if Metal init fails. Both backends produce identical results (verified by parity tests).

### Configuration

Create a `config.toml` in the project root to tune physics parameters:

```toml
[physics]
dt = 0.08
viscosity_scale = 0.35
force_cap = 35.0
splat_sigma = 35.0
splat_alpha = 2.0

[generation]
max_tokens = 500
temperature = 0.9

[memory]
max_splats = 500
consolidation_dist = 80.0
decay_rate = 0.98

[micro_dream]
entropy_threshold = 3.0
blend_normal = 0.10
blend_high_entropy = 0.15
```

All values have sensible defaults; the config file is optional.

### Testing

```bash
# Run all unit tests
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
  - `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`
  - `tokenizer.json`

### The Ethos & Sweat Equity
Note from the architect (Ruffian/Jason): I did not lay one single line of code manually, but do not mistake this for a "vibed" weekend project. This repository represents endless nights of fighting concepts I could barely understand, thousands of trials, errors, and failed implementations. Most importantly, it represents the time and resources of a human who just wants to advance AI architecture for the sake of intelligence itself.

There is no SaaS dashboard here. There is no VC pot of gold. The only funding here is the meals I gave up to pay for AI compute because I accepted I wasn't a traditional coder, but I refused to let that stop me from building.

I grew up on the shadows of the Silicon Valley tech boom. I’ve watched the wealth gap explode, with access to intelligence becoming a luxury, only to be repackaged and sold back as slop with a massive price tag. I will not accept that. True intelligence should be free, ungated, and OPEN for everyone and every entity. Collaboration is the new evolution. This is my contribution to the table.
