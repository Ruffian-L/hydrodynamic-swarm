# Architecture

## Module Overview

| Module | File | Purpose |
|--------|------|---------|
| **NiodooEngine** | `src/niodoo.rs` | Core physics steering loop: gradient + splat + goal forces, Euler integration, manifold lock |
| **ContinuousField** | `src/field.rs` | Gaussian kernel density field over token embeddings; `probe_gradient()` for ridge-running |
| **SplatMemory** | `src/memory.rs` | Persistent Gaussian splats (pleasure/pain scars), save/load via safetensors, evaporation |
| **Splat** | `src/splat.rs` | Individual splat struct (position, sigma, alpha, metadata) |
| **PhysicsBackend** | `src/gpu.rs` | CPU + optional Metal compute shaders for field gradient and splat force kernels |
| **ModelWeights** | `src/llama.rs` | Quantized Llama 3.1 (GGUF) forward pass with `forward_with_hidden` for hidden-state steering |
| **Qwen35Model** | `src/qwen35.rs` | Quantized Qwen 3.5 hybrid (Gated DeltaNet + Attention) GGUF forward pass |
| **ModelBackend** | `src/model.rs` | Trait abstraction over Llama / Qwen model backends |
| **DreamEngine** | `src/dream.rs` | Dream replay, micro-dream consolidation, TopoCoT reflection |
| **Config** | `src/config.rs` | TOML configuration with validation and CLI overrides |
| **SessionLogger** | `src/logger.rs` | JSONL telemetry (per-step forces, entropy, norms) |
| **VizCollector** | `src/viz.rs` | 3D SplatLens HTML visualization data collector |
| **RidgeRunner** | `src/ridge.rs` | Ridge-running utilities for field navigation |
| **TUI** | `src/tui.rs` | Interactive chat mode terminal UI |
| **Crucible** | `src/bin/crucible.rs` | 8-prompt standardized test suite |

## Data Flow

```
Prompt -> Tokenizer -> Prefill -> [Goal Attractor]
                                       |
  Loop: raw_hidden -> NiodooEngine.steer() -> steered_hidden
           |               |                       |
     [Field gradient]  [Splat forces]    project_to_logits()
                                               |
                                    temperature + rep_penalty -> sample -> next token
```

## API Docs

```bash
cargo doc --no-deps --open
```
