# CODE MAP -- Hydrodynamic Swarm

> 15 Rust source files, ~4800 lines total.
> SplatRAG v1 -- Physics-steered LLM generation with Gaussian splat memory.

---

## Architecture Overview

```
src/
  main.rs                 -- Entry point, CLI, generation loop
  config.rs               -- TOML config (physics, generation, memory, micro-dream)
  bin/crucible.rs         -- 8-prompt standardized test suite
  physics/
    niodoo.rs             -- Physics steering engine (3-force model)
    field.rs              -- Continuous Diderot embedding field
    gpu.rs                -- PhysicsBackend trait + CPU/Metal backends
    ridge.rs              -- Ridge-running particle simulation
  memory/
    memory.rs             -- Splat memory (scar tissue)
    splat.rs              -- Gaussian splat data type
    dream.rs              -- Dream replay + micro-dream + TopoCoT
  model/
    llama.rs              -- Vendored quantized Llama (hidden-state steering)
  telemetry/
    logger.rs             -- JSONL telemetry logger
  ui/
    tui.rs                -- Chat-style TUI mode
    viz.rs                -- 3D visualization data collector
    viz_metal.rs          -- HTML/Canvas 3D viewer generator
```

---

## File Details

### `main.rs` (803 lines)

Entry point. CLI argument parsing, model loading, generation loop.

| Function | Lines | Description |
|----------|-------|-------------|
| `main()` | 36-786 | Full pipeline: load field, load Llama, build engine, run generation |
| `find_file()` | 788-802 | Path resolution with fallback |

**Key responsibilities:**
- Metal device selection (`Device::new_metal(0)`)
- GGUF model + tokenizer loading
- Prompt prefill with `forward_with_hidden()`
- Per-token generation loop with physics steering
- Hidden-state steering (`steer_hidden=true`): steer hidden -> `project_to_logits()`
- Logit-space fallback (`steer_hidden=false`)
- Micro-dream triggering (entropy-adaptive + fixed interval)
- Online splat creation from steering delta
- TopoCoT reflection injection
- Dream replay (post-generation)
- Memory Museum CLI (save/load/browse splat memory)
- Visualization export + optional Metal 3D viewer

---

### `config.rs` (270 lines)

TOML-deserializable configuration with validation.

| Struct | Fields |
|--------|--------|
| `Config` | `physics`, `generation`, `memory`, `micro_dream` |
| `PhysicsConfig` | `dt`, `viscosity_scale`, `force_cap`, `splat_sigma`, `splat_alpha`, `min_splat_dist`, `splat_delta_threshold`, `gradient_topk`, `steer_hidden` |
| `GenerationConfig` | `max_tokens`, `temperature`, `default_prompt`, `eos_token_ids` |
| `MemoryConfig` | `max_splats`, `consolidation_dist`, `decay_rate`, `prune_threshold` |
| `MicroDreamConfig` | `entropy_threshold`, `fixed_interval`, `adaptive_interval`, `blend_normal`, `blend_high_entropy`, `topocot_threshold` |

| Method | Description |
|--------|-------------|
| `Config::load()` | Load TOML, validate, return defaults if missing |
| `Config::validate()` | Check all numeric invariants |

**Tests:** 5 (default validates, TOML parsing, negative dt, zero tokens, EOS defaults)

---

### `llama.rs` (526 lines)

Vendored `candle-transformers/quantized_llama.rs` with hidden-state access.

| Struct | Description |
|--------|-------------|
| `QMatMul` | Traced quantized matrix multiply wrapper |
| `Mlp` | Feed-forward block (w1, w2, w3) |
| `MlpOrMoe` | Mlp or Mixture-of-Experts dispatch |
| `LayerWeights` | Single transformer layer (attention + MLP + KV cache) |
| `ModelWeights` | Full model: embeddings, layers, norm, lm_head |

| Method | Lines | Description |
|--------|-------|-------------|
| `ModelWeights::from_ggml()` | 263-325 | Load from GGML format |
| `ModelWeights::from_gguf()` | 327-446 | Load from GGUF format |
| `ModelWeights::run_layers()` | 461-491 | Shared: run all layers, return hidden state `(1, D)` |
| `ModelWeights::forward()` | 493-498 | Standard: hidden -> lm_head -> logits |
| `ModelWeights::forward_hidden()` | 500-505 | Hidden state only (pre-lm_head) |
| `ModelWeights::forward_with_hidden()` | 507-518 | Both logits AND hidden state (no wasted compute) |
| `ModelWeights::project_to_logits()` | 520-523 | Project steered hidden state through lm_head |

**Key detail:** `run_layers()` uses `narrow(1, last, 1).squeeze(1)` to guarantee `(1, D)` shape.

---

### `niodoo.rs` (175 lines)

Core physics steering engine. Three forces act on the residual stream:

1. **Field gradient** -- ridge-running (viscosity-scaled)
2. **Splat scar tissue** -- Gaussian pull/push from memory
3. **Goal attractor** -- linear pull toward prompt's semantic goal

| Method | Description |
|--------|-------------|
| `NiodooEngine::new()` | Build with field, memory, backend, physics params |
| `NiodooEngine::steer()` | Core: apply 3-force physics to `(1, D)` residual |
| `NiodooEngine::set_gradient_topk()` | Set Top-K gradient approximation (0 = exact) |
| `NiodooEngine::field()` | Read-only field access |
| `NiodooEngine::memory()` / `memory_mut()` | Splat memory access |
| `NiodooEngine::dim()` | Embedding dimension |

**Force equation:** `steered = baseline + dt * clamp(grad*viscosity + splat + goal, -cap, cap)`

---

### `field.rs` (362 lines)

Continuous Diderot embedding field. Sum of Gaussian kernels over stored positions.

| Method | Description |
|--------|-------------|
| `ContinuousField::load_real()` | Load safetensors (positions, mass, charge), auto-tune sigma |
| `ContinuousField::load_dummy()` | Random embeddings for testing |
| `ContinuousField::probe()` | Scalar density at position |
| `ContinuousField::probe_gradient()` | Full gradient (all N points, NaN-safe) |
| `ContinuousField::probe_gradient_topk()` | Approximate gradient (K nearest, O(N) partial sort) |
| `ContinuousField::nearest_tokens()` | K nearest token IDs by cosine similarity |

**Tests:** 4 (gradient direction, density positivity, zero far away, topk matches exact)

---

### `memory.rs` (487 lines)

Splat memory -- accumulated experience ("scar tissue").

| Method | Description |
|--------|-------------|
| `SplatMemory::new()` | Empty memory on given device |
| `SplatMemory::add_splat()` | Insert a splat |
| `SplatMemory::decay_step()` | Asymmetric decay (pain lasts 70% longer) |
| `SplatMemory::query_force()` | Summed Gaussian pull/push from all splats |
| `SplatMemory::has_nearby()` | Check if any splat within min_dist (samples 50) |
| `SplatMemory::consolidate()` | Greedy merge nearby same-sign splats |
| `SplatMemory::prune_to_limit()` | Keep N strongest by abs(alpha) |
| `SplatMemory::save()` / `load()` | Safetensors persistence |
| `SplatMemory::save_metadata()` / `load_metadata()` | JSON sidecar |

**Tests:** 7 (attract, repel, zero force, merge, preserve distant, no merge opposite, prune)

---

### `splat.rs` (26 lines)

Minimal data type for a Gaussian splat.

```rust
pub struct Splat {
    pub mu: Tensor,   // position (D,)
    pub sigma: f32,   // isotropic width
    pub alpha: f32,   // signed: + = pleasure, - = pain
}
```

---

### `gpu.rs` (858 lines)

Physics backend abstraction with CPU and Metal implementations.

| Trait Method | Description |
|-------------|-------------|
| `field_gradient()` | Exact field gradient at position |
| `field_gradient_topk()` | Top-K approximate gradient |
| `splat_force()` | Aggregate splat force at position |
| `batch_field_gradient()` | Batched gradient for multiple positions |
| `name()` | Backend identifier string |

| Backend | Description |
|---------|-------------|
| `CpuBackend` | Wraps field/memory code directly |
| `MetalBackend` | WGSL compute shaders via wgpu (feature-gated) |

**Metal shaders:** Two WGSL kernels embedded as string literals:
- Field gradient: per-query workgroups over all field points
- Splat force: per-query workgroups over all splats

**Tests:** 5 (CPU gradient matches direct, splat force matches, batch shape)

---

### `dream.rs` (107 lines)

Dream replay + micro-dream consolidation + TopoCoT.

| Function/Struct | Description |
|----------------|-------------|
| `DreamEngine` | Post-generation replay with Langevin noise |
| `DreamEngine::run()` | Replay trajectories, apply global decay |
| `micro_dream()` | Real-time: forward project N steps, backward anchor to goal, blend correction |
| `MicroDreamResult` | Consolidated position + correction norm + reflection flag |

**TopoCoT:** When `correction_norm > DREAM_CORRECTION_THRESHOLD` (18.0), a reflection marker is injected.

---

### `ridge.rs` (222 lines)

Ridge-running particle simulation. Proves physics works without the LLM.

| Struct | Description |
|--------|-------------|
| `QueryParticle` | Position, velocity, mass in embedding space |
| `RidgeRunner` | Drives particle through field until settled on ridge |
| `RunStats` | Steps, settled flag, final speed/density |

| Method | Description |
|--------|-------------|
| `RidgeRunner::run()` | Full loop: gradient + splats + goal -> Euler integration -> damping |
| `RidgeRunner::run_with_memory()` | Simplified test loop |

---

### `logger.rs` (250 lines)

JSONL telemetry logger for reproducibility.

| Struct | Description |
|--------|-------------|
| `StepEntry` | Per-token: token_id, text, steering_delta, force magnitudes |
| `SessionConfig` | Snapshot of all physics/generation params |
| `SessionSummary` | Token counts, splat stats, delta stats, decoded output |
| `LogEntry` | Top-level wrapper (config/step/summary discriminated) |
| `SessionLogger` | File writer with session ID and delta tracking |

**File naming:** `logs/{YYYY-MM-DD}_{HH-MM-SS}_{label}.jsonl`

---

### `tui.rs` (234 lines)

Chat-style terminal UI for interactive single-prompt generation.

| Function | Description |
|----------|-------------|
| `run_chat()` | Clear screen, show banner, take prompt, generate with live token streaming |

**Features:** ANSI colors, live streaming, micro-dream integration, online splat creation.

---

### `viz.rs` (343 lines)

3D visualization data collector. Random projection from 4096D to 3D.

| Struct | Description |
|--------|-------------|
| `TokenNeighbor` | Token ID + text + probability |
| `VizSnapshot` | Per-step: position_3d, token, delta, neighbors |
| `VizSplat` | Projected splat scar |
| `VizSession` | Full session: prompt, snapshots, field points, ridge ghost |
| `VizRenderData` | Lightweight data for Metal renderer |
| `VizCollector` | Accumulates snapshots during generation |

| Method | Description |
|--------|-------------|
| `VizCollector::new()` | Build projection matrix (seed=42), project field points |
| `VizCollector::snapshot()` | Capture per-step data |
| `VizCollector::export_json()` | Write session JSON |
| `VizCollector::into_render_data()` | Convert for renderer |

---

### `viz_metal.rs` (698 lines)

HTML/Canvas 3D viewer generator. "Architectural Fluidity" palette.

| Function | Description |
|----------|-------------|
| `launch()` | Generate self-contained HTML with embedded JS, open in browser |
| `points_to_js()` | Serialize 3D points to JS array |
| `floats_to_js()` | Serialize float array to JS |

**Aesthetic:** Deep space/ocean depths. Ice-blue trail, teal (pleasure) / rust (pain) splats, gold attractors.

---

### `bin/crucible.rs` (123 lines)

Standardized 8-prompt test suite for baseline comparison.

| Prompt | Category |
|--------|----------|
| 1_SpatialAGI | 3D spatial reasoning (Rubik's cube) |
| 2_TheTrap | Historical hallucination trap (1998 Soviet Moon Landing) |
| 3_AgenticStateMachine | Multi-constraint planning (drone escape) |
| 4_TopoCoT_Metacognition | Self-referential paradox |
| 5_TechnicalArchitect | Rust architecture design |
| 6_PureMathLogic | Water jug problem (step-by-step) |
| 7_DeepContextNeedle | Long-form + context recall |
| 8_CreativeFluidity | Creative dialogue writing |

**Usage:** `cargo run --release --bin crucible [-- tokens]`
Inherits stdout/stderr for live token streaming.

---

## Test Summary

| Module | Tests |
|--------|-------|
| `config.rs` | 5 |
| `field.rs` | 4 |
| `memory.rs` | 7 |
| `gpu.rs` | 3 |
| **Total** | **19** |

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor ops, Metal GPU, quantized models |
| `candle-nn` | Neural network ops (softmax, RoPE, SDPA) |
| `candle-transformers` | Quantized model utilities (RmsNorm, QMatMul) |
| `tokenizers` | HuggingFace tokenizer |
| `serde` / `serde_json` / `toml` | Config + logging serialization |
| `safetensors` | Embedding + memory persistence |
| `rand` | Sampling, Langevin noise |
| `tracing` | Performance tracing spans |
| `wgpu` / `bytemuck` / `pollster` | Metal compute (feature-gated) |
