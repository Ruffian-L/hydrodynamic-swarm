# Merge Verdict — 3surface Wins

**Date:** 2026-07-31
**Author:** Lumina
**Decision:** Merge hydrodynamic-swarm (main) → hydrodynamic-swarm-3surface

## Why 3surface

1. **Feature completeness:** 7 new source files (`jacobian.rs`, `hooks.rs`, `hud.rs`, `logit_physics.rs`, `repl_tui.rs`, `algo_scale.rs`, `picks.rs`), 18 modified files. Main repo is the predecessor.
2. **Config quality:** 3surface has rich, tuned config (physics, hooks, generation, memory, micro_dream). Main has basic config.
3. **Binary works:** 44.9MB release binary, loads Gemma4 31B, builds Diderot field (262k points × 5376 dims), runs physics-steered generation.
4. **Physics systems:** Residual physics, logit physics, forward hooks (post_mlp), endocrine, shared ocean, field wake — all initialized and operational.
5. **Jacobain module:** Shep's spec has a home. Jason's "semantic clustering" hypothesis has a measurement engine.

## Smoke Test Results

- ✅ Model loads (Gemma4 31B, 60 layers, CUDA)
- ✅ Diderot field builds (262144 points, σ=457.015)
- ✅ Physics engine initializes (all subsystems green)
- ✅ Generation runs (output in live.txt)
- ⚠️ JSONL logging bug: config written, generation output missing from JSONL (live.txt has it)
- ⚠️ Config path mismatch between runs: first run used `/media/ruffianl/ghost_team/models/...`, second used `data/google/...` (relative path)

## Open Issues

1. **JSONL generation output missing** — config entry present, but token-level results not written. Need to fix the JSONL writer in `main.rs` or `logger.rs`.
2. **Config path inconsistency** — some runs use absolute paths, some use relative. Standardize on one.
3. **Force ramp parameters differ between runs** — first run: `force_cap=5.0, goal_force_scale=0.15, ramp=0.20/12`. Second run: `force_cap=1.8, goal_force_scale=0.02, ramp=0.10/12`. Config file says `force_cap=1.8` but CLI override may be happening.
4. **Generation output quality** — need to run controlled tests to compare steered vs unsteered output.

## Next Steps

1. Fix JSONL writer to capture generation output
2. Standardize config paths
3. Run controlled comparison: steered vs baseline
4. Wire in Jacobian lens (Shep's spec)
5. Test niodoo control tags integration

## Vote

**[vote/continue]** — Merge main into 3surface, fix JSONL bug, run controlled tests.
