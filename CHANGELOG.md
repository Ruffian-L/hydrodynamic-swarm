# Changelog

This is a research repository. Every mutation records its reason here and a
longer, subject-specific note in `research_logs/`.

## 2026-08-22 — Preserve checksum-required vendor files

- Scoped exceptions keep nested vendored `target`, `.cargo`, `env`, binary,
  editor, and `AGENTS` files from inheriting broad workspace ignore rules.
- Restored 17 files listed by vendored `.cargo-checksum.json` manifests but
  absent from a fresh GitHub checkout.
- Added a standard-library integrity checker for file presence and SHA-256.
- Hypothesis: a fresh clone will pass Cargo source checksum validation instead
  of failing before compilation.
- Research: `research_logs/2026-08-22_vendor-checksum-ignore-collisions.md`
- Agent: Sol (OpenAI)
