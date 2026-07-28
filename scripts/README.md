# Scripts

Run from **repo root** unless noted.

| Script | Role |
|--------|------|
| `run_swarm.sh` | Main generate entry (also `./run_swarm.sh` root wrapper) |
| `demo_slice.sh` | Short viz slice + museum |
| `chat.sh` / `chat_gemma4.sh` | Multi-turn REPL diagnostics |
| `cuda_env.sh` | CUDA env helper (sourced by others) |
| `ab_*.sh` | A/B comparisons |
| `g3_ablation.sh` / `g4_*.sh` | Gate ablations (use `configs/gates/`) |
| `ablation_sweep.sh` / `force_temp_sweep.sh` / `goal_force_ablation.sh` | Parameter sweeps |
| `continuity_*.sh` / `continuity_*.py` | Continuity / bridge smokes |
| `memory_coupling_smokes.sh` | Memory reload smokes |
| `learning_lane_*.sh` / `splat_lane_4b.sh` / `f_decay_lane_4b.sh` | Lane experiments |
| `endocrine_enzyme.sh` | Endocrine path helper |
| `console_watch.sh` | Console watch |
| `crucible.sh` / `crucible_tui.sh` | Crucible harness |
| `scale_physics_for_model.py` | Scale knobs by model size |
| `extract_force_windows.py` / `list_bridges.py` | Telemetry helpers |

Logs go under `logs/` (gitignored). Private chats under `private/` (gitignored).
