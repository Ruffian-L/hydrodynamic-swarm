# Concourse lock-poison recovery

> Date: 2026-08-22
> Agent: Sol (OpenAI)
> Repo: Ruffian-L/hydrodynamic-swarm

## Context

GitHub has no open Issues, but its blocked pull-request backlog contains many
duplicate automated fixes for panic-on-poison lock acquisition. PRs 725–727
replace `LockResult::unwrap` with `PoisonError::into_inner` across the Concourse
cache and Gemma wrappers.

## What changed

- `CacheManager` now owns the poison policy for each `RwLock`. On recovery it
  clears the disposable cache and clears the poison flag before continuing.
- Statistics retain concurrent read-lock behavior when healthy and acquire a
  write guard only when poisoned state must be cleared.
- EmbeddingGemma and InstructGemma recover their `Mutex` guards only after
  clearing mutable KV state. Loaded model weights remain unchanged.
- A cache regression test intentionally poisons both locks and verifies that
  recovery drops pre-panic entries, clears poison, and permits later traffic.

## Hypothesis

An isolated panic while holding a Concourse lock will no longer cascade into
panics on every later access. Resetting the mutable state should preserve this
availability benefit without trusting an operation that may have stopped
mid-mutation.

## Findings

- The first supported-feature compile reached the new test and correctly
  required the generic poison helper's `T` to be both `Send` and `Sync`; the
  test-only bound was tightened before rerunning.
- Repository-wide formatting is already red on unrelated files. Targeted
  `rustfmt --check` passes for `cache.rs` and `instruct_gemma.rs`; the only
  `embed/gemma.rs` diffs are four pre-existing expression wraps outside this
  change.
- A fresh GitHub checkout also omits 17 checksum-listed vendor files because
  root ignore patterns match nested crate paths. Checksum-correct copies were
  extracted from the local Cargo crate cache only to enable verification; no
  vendored source is part of this change.
- Targeted receipt: `cargo test --no-default-features --features with-candle
  concourse::cache::tests -- --nocapture` with `RUSTFLAGS=-C
  target-feature=+fp16` passed 4/4. The poison test emitted its two intentional
  worker panics, recovered both locks, and passed.
- Full receipt under the same feature/target flags: 66/69 passed. The failures
  are pre-existing and outside this change: `test_embed_alpha` and
  `test_swarm_ingest_flux_with_real_models` require absent GGUF files;
  `config::tests::toml_parsing_works` expects viscosity `0.15` from unrelated
  config parsing.
- The build script regenerated `kernels/decay.ptx` with local CUDA 13.3; its
  header was restored to the tracked CUDA 13.0 artifact so this branch carries
  no unrelated kernel churn.
- No model smoke is required because generation behavior and model parameters
  are unchanged.

## Next

1. Reconcile the consolidated change with the duplicate blocked PR cluster.
2. Fix vendored-package ignore collisions in a separate packaging change.
