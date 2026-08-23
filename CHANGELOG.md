# Changelog

This is a research repository. Every mutation records its reason here and a
longer, subject-specific note in `research_logs/`.

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
