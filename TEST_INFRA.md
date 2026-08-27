# E2E Test Infra: Unified 6-Tab Ratatui Frontend for Hydrodynamic Swarm

## Test Philosophy
- Opaque-box, requirement-driven testing. Verification of non-blocking thread behavior, tab navigation, parameter synchronization, and non-panicking startup.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 |
|---|---------|--------|:------:|:------:|:------:|
| 1 | Ratatui/Crossterm Scaffold & Concurrency | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 2 | Tab 1: Model Loader & Config | ORIGINAL_REQUEST §R2 Tab 1 | 5 | 5 | ✓ |
| 3 | Tab 2: Physics Board & HookControls | ORIGINAL_REQUEST §R2 Tab 2 | 5 | 5 | ✓ |
| 4 | Tab 3: System Deck & Tags | ORIGINAL_REQUEST §R2 Tab 3 | 5 | 5 | ✓ |
| 5 | Tab 4: Debug Matrix & TDA | ORIGINAL_REQUEST §R2 Tab 4 | 5 | 5 | ✓ |
| 6 | Tab 5: Compare Arena | ORIGINAL_REQUEST §R2 Tab 5 | 5 | 5 | ✓ |
| 7 | Tab 6: Misc (KV Cache & Remember Store) | ORIGINAL_REQUEST §R2 Tab 6 | 5 | 5 | ✓ |
| 8 | CLI Integration & Non-Regression | ORIGINAL_REQUEST Acceptance | 5 | 5 | ✓ |

## Test Architecture
- Unit / Integration Test Suites under `tests/` and within modules.
- Headless terminal test harness (`ratatui::backend::TestBackend`) testing frame rendering, tab switching, and event dispatch without opening physical tty.
- Integration tests executing `cargo check --bin hydrodynamic-swarm`, `cargo check --tests`, and `--ratatui --dry-run` or headless verification.
- Shell script integrity validation (`bash -n scripts/talk.sh`, etc.).

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full UI Lifecycle (Startup -> Tab Navigation [1->2->3->4->5->6] -> Parameter Tweak -> Quit) | F1, F2, F3, F4, F5, F6, F7, F8 | Medium |
| 2 | Non-blocking Generation Stream (Prompt submission on Tab 3 -> streaming tokens on UI while adjusting sliders on Tab 2) | F1, F3, F4, F5 | High |
| 3 | Live Physics & Hook Tuning (Adjusting goal/repulsion force and hook band, verifying message receipt and state sync) | F1, F3, F5 | Medium |
| 4 | Model Config & KV Snapshot Roundtrip (Changing temperature, triggering KV snapshot on Tab 6, verifying state update) | F1, F2, F7 | Medium |
| 5 | Existing CLI Non-Regression (`generate_turn_ex` / `talk.sh` invocation compatibility) | F8 | Medium |
