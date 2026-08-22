# Changelog

This is a research repository. Every mutation records its reason here and a
longer, subject-specific note in `research_logs/`.

## 2026-08-22 — CUDA feature/build isolation

- Made `with-cuda` and `with-metal` include the complete `with-candle`
  dependency set so each accelerator feature is independently buildable.
- Removed an orphan build script that invoked `nvcc` for every feature set and
  rewrote tracked `kernels/decay.ptx`; no runtime code loaded that artifact.
- Left CUDA compilation to Candle's feature-gated kernel path, so CPU-only and
  Metal builds no longer discover or invoke the CUDA toolchain.
- Hypothesis: CPU checks remain CUDA-independent, explicit CUDA builds receive
  the complete model stack, and repeated builds leave the worktree clean.
- Verification: CPU-only and explicit-CUDA checks passed; the PTX checksum was
  unchanged, and 66 model-independent/inherited-passing unit tests passed.
- Research: `research_logs/2026-08-22_cuda-feature-build-isolation.md`
- Agent: Sol (OpenAI)

## 2026-08-22 — Concourse lock-poison recovery

- Replaced panic-on-poison cache and Gemma lock acquisition with explicit
  recovery paths. Disposable caches are cleared; model KV state is reset.
- Kept cache statistics on read locks during the normal path; they escalate to
  a sanitizing write lock only after poison is detected.
- Added a regression test that poisons both cache locks, verifies stale entries
  are discarded, and confirms subsequent cache reads and writes still work.
- Hypothesis: an isolated worker panic no longer causes cascading lock panics,
  while partially mutated cache/model state is not reused blindly.
- Verification: targeted cache suite 4/4 passed; full CPU-Candle suite passed
  66/69, with two missing-model failures and one unrelated config assertion.
- Research: `research_logs/2026-08-22_concourse-lock-poison-recovery.md`
- Agent: Sol (OpenAI)
