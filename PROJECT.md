# Project: Unified 6-Tab Ratatui Frontend for Hydrodynamic Swarm

## Architecture
A decoupled multi-threaded Rust terminal user interface built on Ratatui (v0.29) and Crossterm (v0.29).
- **UI Thread (Main)**: Runs Crossterm event polling and Ratatui frame rendering at a responsive rate (~30-60 FPS). Renders the 6 tabs, manages widget focus, keyboard navigation, text editing, and slider adjustments.
- **Engine Worker Thread**: Owns `Model`, `NiodooEngine`, `LogitChain`, and `HookControls`. Executes `generate_turn_ex` and parameter adjustments without blocking the UI thread.
- **Bidirectional Channels**:
  - `UiToEngine`: Commands including `LoadModel`, `GenerateTurn`, `SetParam`, `SetHookControl`, `InjectSystemPrompt`, `SnapshotKv`, `ClearKv`, `UpsertRememberLine`, `AbortGeneration`.
  - `EngineToUi`: Events including `TokenGenerated(&str, HudFrame)`, `GenerationFinished`, `ModelLoaded(ModelInfo)`, `EngineError(String)`, `HookFired(HookReport)`, `TdaUpdate(TdaMetrics)`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Ratatui/Crossterm Scaffold | Multi-threaded async event loop, clean panic hooks, 6-tab header, status footer | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Tab 1: Model Loader & Config | Model swap selector, temperature, top-p, context length sliders, scaling preview | M2 | ORIGINAL_REQUEST §R2 Tab 1 |
| 3 | Tab 2: Physics Board | Goal force, repulsion, hand beta sliders, HookControls layer band & norm fraction | M2 | ORIGINAL_REQUEST §R2 Tab 2 |
| 4 | Tab 3: System Deck | System prompt live text editor, control tag (<spike>, <focus>, etc.) inspector | M3 | ORIGINAL_REQUEST §R2 Tab 3 |
| 5 | Tab 4: Debug Matrix | Live entropy/margin sparklines, TDA loop pressure & homology, active hook logs | M3 | ORIGINAL_REQUEST §R2 Tab 4 |
| 6 | Tab 5: Compare Arena | Side-by-side comparative generation (Llama.cpp API vs Hydro Swarm steered) | M4 | ORIGINAL_REQUEST §R2 Tab 5 |
| 7 | Tab 6: Misc (KV & Remember) | Choice-driven KV cache snapshot/restore/clear toggles, RememberStore JSONL editor | M4 | ORIGINAL_REQUEST §R2 Tab 6 |
| 8 | Engine Integration & CLI | Add `--ratatui` / `--tui-unified` CLI flag, ensure zero regression on `talk.sh` & `generate_turn_ex` | M4 | ORIGINAL_REQUEST Acceptance |
| 9 | Headless Verification & E2E Suite | 100% E2E testing suite, verification under headless/startup mode, non-panicking UI | M5 | ORIGINAL_REQUEST Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Scaffold & Concurrency Core | Ratatui dependency integration, decoupled UI/Engine threads, channels, 6-tab layout frame, key event router | none | DONE (`src/frontend/`, 57/57 tests passing) |
| M2 | Tabs 1 & 2 (Model & Physics) | Tab 1 (Model Loader, Config sliders) + Tab 2 (Physics Board, HookControls, Niodoo live params). Dry-run worker writes `Config::set_live_param` / `HookControls::set_param` (same names as `/set`). No GGUF load, no `generate_turn_ex`. | M1 | DONE (dry-run seat; live generation is M4) |
| M3 | Tabs 3 & 4 (System & Debug) | Tab 3 (System prompt editor, Tag inspector) + Tab 4 (Debug Matrix, TDA homology sparklines, Hook reports) | M1 | PLANNED |
| M4 | Tabs 5 & 6 (Compare & Misc) + CLI | Tab 5 (Compare Arena) + Tab 6 (KV Cache, Remember Store) + CLI entrypoints and integration | M2, M3 | PLANNED |
| M5 | Final E2E Verification & Hardening | Opaque-box & unit tests, startup smoke check, changelog & research log pair | M4 | PLANNED |

## Interface Contracts

### UI ↔ Engine Channel Message Contract
```rust
pub enum UiToEngineMsg {
    LoadModel { path: String, tokenizer: Option<String> },
    SetLiveParam { key: String, val: f32 },
    SetHookControl { enabled: bool, site: f32, start_frac: f32, end_frac: f32, norm_fraction: f32 },
    SetSystemPrompt(String),
    StartGeneration { prompt: String, temperature: f32, max_tokens: usize },
    AbortGeneration,
    SnapshotKv,
    RestoreKv,
    ClearKv,
    UpsertRememberLine { key: String, val: String },
    CompareVanilla { prompt: String, endpoint: String },
}

pub enum EngineToUiMsg {
    EngineReady,
    ModelLoaded { name: String, n_layers: usize },
    TokenGenerated { text: String, frame: hud::HudFrame },
    GenerationComplete { total_tokens: usize, elapsed_sec: f32 },
    Error(String),
    CompareResult { vanilla_text: String, hydro_text: String },
    RememberStoreUpdated(Vec<(String, String)>),
    KvSnapshotStatus { state: String },
}
```

## Code Layout
- `src/frontend/`:
  - `mod.rs`: App struct, main entrypoint `pub fn run_ratatui_frontend(...) -> Result<()>`
  - `event.rs`: Input event loop and crossterm polling
  - `channel.rs`: `UiToEngineMsg` and `EngineToUiMsg` definitions
  - `engine_bridge.rs`: Background engine worker thread runner interfacing with `generate_turn_ex`
  - `tabs/`:
    - `mod.rs`: Tab enum and shared trait/widgets
    - `tab1_model.rs`: Model Loader & Config tab
    - `tab2_physics.rs`: Physics Board & HookControls tab
    - `tab3_system.rs`: System Deck & Tag Inspector tab
    - `tab4_debug.rs`: Debug Matrix, TDA sparklines & Hook reports tab
    - `tab5_compare.rs`: Compare Arena tab (side-by-side viewer)
    - `tab6_misc.rs`: KV Cache & Remember Store tab
- `src/main.rs`: CLI argument parsing for `--ratatui` / `--tui-unified` dispatching to `run_ratatui_frontend`.
