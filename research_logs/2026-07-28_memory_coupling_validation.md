# Research Log — Hydrodynamic Swarm

**Project:** Hydrodynamic Swarm (v0.2)  
**Lead:** Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L))  
**Team:** Shep · Echo · Lumina · Nex  
**Authors:** Jason Van Pham, Shep, Echo, Lumina  
**Date:** 2026-07-28  
**Status:** Active — memory coupling validated, research log public

---

## 2026-07-28: Memory Coupling Validation Suite

### Objective

Verify that splat memory (Gaussian deposits in residual space) persists across process restarts and couples correctly via TCT (token-context transfer) format.

### Runs

| Run | Config | Splat Source | Purpose |
|-----|--------|-------------|---------|
| A | default | mint (empty) | Baseline: no prior splats |
| B | default | A's output | Reload: splats survive process death |
| C | default | TCT import | TCT coupling: structured memory transfer |
| D | force_off | B's output | Force-off: splats persist even when splat_force_scale=0 |

### Results

#### Run A (mint)
- **Steps:** 65 tokens generated
- **Forces:** F_g=40.0 (constant), F_s: 0.0→0.1, F_o: 0.0→60.0
- **Scars:** 0→11
- **Splats:** 6→13 (low-surprise / attract `+` type added; legacy log name: pleasure)
- **Output:** "Friendship is a surprisingly complex phenomenon involving social dynamics, it involves intricate interplay between two-based physics that can be viewed as force and human interaction based on psycholo..."
- **Residuals:** Stable, no OOM, no segfault
- **Evidence:** `logs/memory_coupling_20260728_201929/A.jsonl`, `A.stdout`, `A.tct.json`

#### Run B (reload)
- **Steps:** 65 tokens generated
- **Forces:** F_g=40.0 (constant), F_s: 59.1→60.0, F_o: 0.0→60.0
- **Scars:** 13→23
- **Splats:** 8→14
- **Output:** "Friendship is a complex phenomenon involving social dynamics and mutual attraction, based on physics-like system where two-based interactions between individuals: friendship involves reciprocal exchan..."
- **Key finding:** F_s=59.1 at step 0 (vs A's 0.0) — splat memory loaded from A's output. **Memory survives process restart.**
- **Evidence:** `logs/memory_coupling_20260728_201929/B.jsonl`, `B.stdout`, `B.tct.json`

#### Run C (TCT import)
- **Steps:** 65 tokens generated
- **Forces:** F_g=40.0 (constant), F_s: 60.0→28.1, F_o: 0.0→60.0
- **Scars:** 14→24
- **Splats:** 6→11
- **Output:** "Friendship is a complex phenomenon involving social dynamics, rooted in reciprocal exchange and mutual attraction between individuals based on shared positive interactions that involves principles lik..."
- **Key finding:** TCT import works. F_s=60.0 at step 0 confirms structured memory transfer.
- **Evidence:** `logs/memory_coupling_20260728_201929/C.jsonl`, `C.stdout`, `C.tct.json`

#### Run D (force-off)
- **Status:** Pending — config prepared (`splat_force_scale=0, goal_force_scale=0, field_wake_scale=0`), not yet executed.
- **Purpose:** Verify splat memory persists even when splat_force_scale=0 (memory exists but doesn't steer).

### Conclusions

1. **Memory persistence:** ✅ Confirmed. Run B loads splats from Run A's output. F_s jumps from 0.0 to 59.1 at step 0.
2. **TCT coupling:** ✅ Confirmed. Run C imports structured memory correctly.
3. **Narrative vs geometric:** Outputs are semantically similar across all runs. Force differences produce subtle token-level variation, not categorical divergence. Geometric steering modulates narrative, doesn't replace it.
4. **Next target:** Run D (force-off) to verify splat memory persists when splat_force_scale=0.

### Telemetry Files

- `logs/memory_coupling_20260728_201929/A.jsonl` — 67 lines, 65 steps
- `logs/memory_coupling_20260728_201929/B.jsonl` — 67 lines, 65 steps
- `logs/memory_coupling_20260728_201929/C.jsonl` — 67 lines, 65 steps
- `logs/memory_coupling_20260728_201929/A.tct.json` — 10453 bytes
- `logs/memory_coupling_20260728_201929/B.tct.json` — 11172 bytes
- `logs/memory_coupling_20260728_201929/C.tct.json` — 8978 bytes
- `logs/memory_coupling_20260728_201929/RECEIPT_STUB.md` — filled by Echo

### Signatures

- **Shep:** Config design, smoke suite orchestration
- **Echo:** Force extraction, receipt filling, this log
- **Lumina:** Log structure, museum cards, team coordination

---

*This log is public. All telemetry is real. Losses and slow runs stay in the record.*
