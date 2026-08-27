# Unified 6-Tab Ratatui Scaffold and Decoupled Concurrency Core

> Date: 2026-08-27
> Agent: worker_m1
> Repo: hydrodynamic-swarm-3surface

## Context
The Hydrodynamic Swarm engine contains rich backend multi-surface steering, TDA topological monitoring, KV cache snapshotting, and model hot-swapping, but lacked a unified, non-blocking terminal frontend. Generation tensor operations take 15ms-200ms per token, which previously stuttered or blocked terminal UI loops. Milestone 1 establishes the scaffold and concurrency core for a unified 6-tab Ratatui frontend.

## What changed
1. Integrated `ratatui = { version = "0.29", default-features = false, features = ["crossterm"] }` in `Cargo.toml`.
2. Created `src/frontend/` module hierarchy:
   - `mod.rs`: `App` state struct, `TerminalGuard` with RAII raw-mode cleanup and panic hooks, `run_ratatui_frontend` main loop (~60 FPS).
   - `channel.rs`: `UiToEngineMsg` and `EngineToUiMsg` channel message contracts.
   - `event.rs`: Input event loop handling keyboard navigation (`Tab`/`Shift+Tab`, `1-6`, arrow keys, `Esc`/`q`, `Ctrl+C`).
   - `engine_bridge.rs`: Background engine worker thread spawner managing crossbeam channels and handling non-blocking generation/commands.
   - `tabs/mod.rs`: `Tab` enum (ModelLoader, PhysicsBoard, SystemDeck, DebugMatrix, CompareArena, Misc), top tab bar, and status footer renderer.
   - `tabs/tab1_model.rs` ... `tab6_misc.rs`: All 6 tab renderers providing modular container blocks and live UI components.
3. Added `pub mod frontend;` to `src/main.rs`.
4. Verified compilation and comprehensive test coverage (57 integration tests in `test_ratatui_frontend`, 12 binary unit tests).

## Hypothesis
By isolating the Ratatui frame rendering and Crossterm event loop on the main thread while delegating all inference, physics, and model swapping to a background worker thread via unbounded channels, the UI maintains a constant 60 FPS without frame stutter, dropped keypresses, or blocked redraws during heavy token generation.

## Findings
- All 57 tests in `test_ratatui_frontend` passed.
- Headless testing on `TestBackend` verified that tab cycling, direct number jump, slider boundary limits, rapid message bursts, and unicode inputs execute with zero panics.
- Terminal restoration via RAII `TerminalGuard` and panic hooks safely reverts raw mode and alternate screen on exit.

## Next
Milestone 2 (Tabs 1 & 2): Wire up live model loader file browsing/hot-swapping and interactive slider bindings for the 3-surface physics controls (Niodoo residual forces, logit chain governor, and transformer layer hooks).
