# Milestone 2: Interactive Model Loader, 3-Surface Physics Board, & HookControls

> Date: 2026-08-27
> Agent: `worker_m2`
> Repo: `hydrodynamic-swarm-3surface`

## Context
Milestone 2 delivers full interactive controls and real-time state synchronization for the first two tabs of the unified 6-tab Ratatui frontend:
1. **Tab 1: Model Loader & Config** (`src/frontend/tabs/tab1_model.rs`)
2. **Tab 2: Physics Board** (`src/frontend/tabs/tab2_physics.rs`)
3. **Engine Bridge Live State & Sync** (`src/frontend/engine_bridge.rs`)

## What changed

### 1. Tab 1: Model Loader & Config (`src/frontend/tabs/tab1_model.rs`)
- Implemented `Tab1State` and `ModelEntry` structures tracking discovered GGUF models (`gemma-4-9b`, `gemma-3-4b`, `qwen2.5-7b`, `llama-3.1-8b`) with automatic architecture and parameter sniffing.
- Added interactive sliders with bounded step adjustment (single step and Shift 10x):
  - Temperature ($0.0 \dots 2.0$, step 0.05 / 0.20)
  - Repetition Penalty ($1.0 \dots 2.0$, step 0.05 / 0.20)
  - Max Tokens ($16 \dots 8192$, step 64 / 512)
  - Top-P ($0.0 \dots 1.0$, step 0.05 / 0.20)
  - Top-K ($1 \dots 256$, step 8 / 32)
- Added live Algo Scale formula preview (`algo_scale.rs`) comparing:
  - Legacy 3B $\sqrt{}$ transform with tight clamps and coupled temperature
  - July 2026 8B $\sqrt{}$ transform with coupled temperature
  - Current worktree Piecewise $\sqrt{}$ to 8B / log-softened above 8B with decoupled temperature
  - Model archetype selector (`Standard`, `Instruct`, `Chat`, `Thinking`, `Coding`)
  - Live computed predicted knobs: $\sigma$ (force cap), $\theta$ (goal force scale), $\beta$, and loop repulsion.
- Added hotkeys: `[L]` (Load selected model), `[U]` (Unload model), `[S]` (Save config), `[C]` (Clear KV cache).

### 2. Tab 2: Physics Board (`src/frontend/tabs/tab2_physics.rs`)
- Implemented `Tab2State` managing 28 interactive controls across 3 physical intervention surfaces:
  - **Surface 1: Residual Forces**:
    - `residual.cap` ($0.0 \dots 20.0$)
    - `residual.goal` ($0.0 \dots 2.0$)
    - `residual.field` ($0.0 \dots 2.0$)
    - `residual.splat` ($0.0 \dots 2.0$)
    - `residual.dt` ($0.001 \dots 0.20$)
    - `force_ramp_len` ($0 \dots 100$) and `force_ramp_str` ($0.0 \dots 1.0$)
    - Ceiling limits: `residual.field_max`, `residual.splat_max`, `residual.goal_max`
    - Live force monitors: $F_{\text{grad}}$, $F_{\text{splat}}$, $F_{\text{goal}}$, and scar count.
  - **Surface 2: Logit Biases & Fluid Governor & Hands**:
    - `field.alpha`, `splat.scale`, `splat.top_m`, `splat.top_k`
    - `gov.on` toggle, `gov.velocity`, `gov.brake`, `gov.visc_gain`, `gov.max_bias`
    - `backslash.pen`
    - `Hands` dynamic repulsion, beta, and blend.
  - **Surface 3: Layer Hook (`HookControls`)**:
    - `hook.on` toggle
    - `hook.site` selector (`PreLayer`, `PostAttn`, `PostMlp`, `FinalNorm`)
    - `hook.norm_fraction` ($0.0 \dots 0.10$, scale-free $\|\Delta h\|/\|h\|$)
    - `hook.start_frac` and `hook.end_frac` layer bands
    - Resolved layer band display (e.g. `L18..L35` for 36-layer stacks)
  - **Stability Verdicts Panel**:
    - Live evaluation of $\sigma \to \text{cap}$, $\theta \to \text{goal}$, and $\beta \to \text{temp}$ against predicted bounds, color-coded with status tags `[IN]`, `[HOT]`, `[COOL]`, `[SAT]`.
- Added hotkeys: `[H]` (Toggle Hook), `[G]` (Toggle Governor), `[R]` (Reset physics to predicted scale).

### 3. Engine Bridge & Concurrency Core (`src/frontend/engine_bridge.rs` & `src/frontend/mod.rs`)
- Background engine worker thread now manages `LiveEngineState` with live parameter maps.
- Receives `UiToEngineMsg::SetLiveParam` and `UiToEngineMsg::SetHookControl`, updates active physics state, and emits `EngineToUiMsg::TelemetryUpdate`.
- Receives `UiToEngineMsg::LoadModel`, sniffs architecture and layer counts, and emits progress and `ModelLoaded` events.
- `App::handle_key_event` routes vertical navigation and horizontal slider adjustments directly, dispatching real-time channel messages without blocking rendering.

### 4. Comprehensive Testing (`tests/test_ratatui_frontend.rs`)
- Added Tier 5 E2E test suite covering:
  - Model loader navigation, hotkeys, and slider adjustments
  - Algo scale preview mathematical accuracy and temperature coupling/decoupling
  - 3-surface physics board slider adjustments and Governor toggling
  - HookControls panel layer band resolution and site cycling
  - Real-time bidirectional message exchange and model hot-swapping

## Hypothesis
Connecting interactive crossterm keybindings to unbounded message channels allows instant continuous tuning of residual, logit, and inter-layer physics while maintaining a fluid 60 FPS UI.

## Findings
- `cargo check --bin hydrodynamic-swarm`: Compiles cleanly with 0 errors.
- `cargo test --test test_ratatui_frontend`: All 62 test cases pass.
- Terminal rendering remains completely responsive and non-blocking during parameter updates and model loading.

## Next
Milestone 3:
- Tab 3: System Deck (Live system prompt injection & control tag inspector for `<spike>`, `<focus>`, `<remember>`, `<lock>`)
- Tab 4: Debug Matrix (Entropy/margin sparklines, TDA loop pressure & homology, active hook logs)
