# hydrodynamic-swarm
A note from me, ruffian or Jason I did not lay one single line of code, but this isn't a "vibed" weekend project. This is endless nights of fighting problems I can't bare to understand, thousounds of trials and errors and failed impmentations, and most importantly the time and resources of a fellow human who just wants to advance AI arichtecture for the advancment of intelligence. There is no saas dashboard here or pot of gold the only funding here is the meals I gave up to pay for ai cause I accept I'm not smart enough to code. I grew up on the other side of silicon valley, I've watched the gap of wealth grow between proverty and intelligece coming with it, then thrown back down as slop that comes with a price tag. I will not accept that. True intelligence should be free, ungatted,  and most importantly OPEN for everyone and every entitity. Collaboartion is the new evolution.

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

## Current Status (v1.2 -- Phase 3 Recovery)

Hardware: NVIDIA Blackwell GB10 | Candle 0.9.2 + CUDA | Llama 3.1 8B Q5_K_M

Config: `dt=0.035, viscosity=0.35, force_cap=7.5, manifold_pullback=0.15, T=0.9`

- Hidden-state steering (`steer_hidden=true`) -- physics operates on 4096D pre-lm_head representations
- Manifold safety: per-step pullback prevents cumulative drift off the Llama manifold
- Repetition penalty (1.18) from config, applied before sampling
- Token mass physics: surprise-weighted trajectory (heavy tokens get stronger splats)
- Bundle stress: collective force from 8 nearest splats adds emergent fluid structure
- VR H1 reflex: Vietoris-Rips topology check with corrective blend on collapse
- Micro-dream consolidation with hydraulic jump clamping (threshold 6.0)
- Real dream replay: actual generation trajectory consolidated with Langevin noise
- Online splat creation with min-distance deduplication
- TOML-configurable physics, generation, memory, and micro-dream parameters
- Full JSONL telemetry logging per step + session summary
- CLI flags: `--prompt`, `--model`, `--tokens`, `--clear-memory`, `--viz`, `--chat`

Rainbow tuning (10 prompts, 120 tokens each): 7/10 fully coherent through 120 tokens, 3/10 coherent for 50-90 tokens before gradual drift.

See [experiments.md](docs/experiments.md) for detailed findings.

## Running

```bash
# Default run (CUDA tensor ops via candle 0.9)
cargo run --release --bin hydrodynamic-swarm

# With a custom prompt
cargo run --release --bin hydrodynamic-swarm -- --prompt "Describe consciousness as a wave function"

# Limit token count
cargo run --release --bin hydrodynamic-swarm -- --tokens 200

# Clear splat memory (fresh start)
cargo run --release --bin hydrodynamic-swarm -- --clear-memory

# Interactive chat mode
cargo run --release --bin hydrodynamic-swarm -- --chat

# All flags combined
cargo run --release --bin hydrodynamic-swarm -- --prompt "What is time?" --tokens 300 --clear-memory
```

### Configuration

Create a `config.toml` in the project root to tune physics parameters:

```toml
[physics]
dt = 0.035
viscosity_scale = 0.35
force_cap = 7.5
splat_sigma = 35.0
splat_alpha = 2.0
manifold_pullback = 0.15
steer_hidden = true
gradient_topk = 2048

[generation]
max_tokens = 500
temperature = 0.9
rep_penalty = 1.18
min_success_tokens = 15
pleasure_alpha = 1.8
pain_alpha = -0.9

[memory]
max_splats = 500
consolidation_dist = 80.0
decay_rate = 0.98

[micro_dream]
entropy_threshold = 3.0
blend_normal = 0.10
blend_high_entropy = 0.15
topocot_threshold = 6.0
```

All values have sensible defaults; the config file is optional.

### Testing

```bash
# Run all unit tests
cargo test

# Lint check
cargo clippy
```

### Requirements

- Rust 1.75+ (stable)
- NVIDIA GPU with CUDA toolkit (tested on Blackwell GB10)
- Model files in `data/bartowski/`:
  - `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` (5.7 GB)
  - `tokenizer_official.json` (from bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
