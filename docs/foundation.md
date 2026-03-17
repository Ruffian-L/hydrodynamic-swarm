# SplatRAG v1 — Hydrodynamic Swarm

## North Star

Retrieval and generation are the same physical process.
A particle sliding down a continuous Diderot field, guided by splat scar tissue.

## Core Runtime Loop (every token)

```rust
let baseline_residual = naked_llama.forward(input);

let query_pos = get_current_hidden_position();

let grad_force = field.probe_gradient(&query_pos) * viscosity_scale;
let splat_force = splat_memory.query_force(&query_pos);
let goal_force = prompt_embedding - query_pos;
let total_force = grad_force + splat_force + goal_force + momentum + noise;

let steered_residual = baseline_residual + (total_force * dt);

next_token_logits = final_layer(steered_residual);
```

## Architecture Phases

### Phase 1: Ingestion (run once / on new memory)
- Raw text → embed
- Optional UMAP to 3D for initial placement
- Gaussian Splat per memory: μ (position), Σ (covariance), α (opacity/viscosity)
- Build continuous field from splats (Gaussian kernel sum)

### Phase 2: Continuous Field (Diderot core)
- `probe(pos)` → density scalar
- `probe_gradient(pos)` → vector slope (ridge-running force)

### Phase 3: Splat Memory (scar tissue)
- Asymmetric decay (pain lasts longer)
- `query_force(pos)` → summed Gaussian pull/push from nearby splats

### Phase 4: Dream Replay (after generation)
- Replay top-K successful trajectories with Langevin noise
- Update splat opacities

## 7-Day Plan

| Day | Milestone |
|-----|-----------|
| 1 | `cargo init` + FOUNDATION.md + `field.rs` with `probe_gradient` |
| 2 | Query particle + ridge-running loop (prove physics works alone) |
| 3 | Full splat memory with asymmetric decay |
| 4 | Hook Niodoo steering into residual stream |
| 5 | Dream replay |
| 6 | Basic coherence eval (8k token test vs baseline) |
| 7 | Push v1 + README |
