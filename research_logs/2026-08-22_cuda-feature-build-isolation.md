# CUDA feature and build isolation

Date: 2026-08-22
Agent: Sol (OpenAI)
Branch: `fix/cuda-feature-gating`

## Question

Do Hydro's Cargo accelerator features actually isolate CUDA from the documented
CPU test lane, and can `with-cuda` be selected as a self-contained feature?

## Baseline receipts

On clean commit `0561cd2f6175ad432b30b208eb2d47c80219ffd2`, this documented
CPU command still invoked NVIDIA's compiler:

```text
cargo check --offline --no-default-features --features with-candle --bin hydrodynamic-swarm
warning: hydrodynamic-swarm@0.2.0: Compiled decay.ptx with nvcc
```

It also changed the tracked `kernels/decay.ptx` checksum from
`60b8852166ff0e162ba196a8db49c1c6e247f2d4dedf0b73315a3c3ef778fa88`
to
`6abf123814532366804f98e46d5290b46de6b03a1d6ff14c962c6c16bff14834`.
The content delta was compiler metadata: local CUDA 13.3 / PTX 9.3 replaced
the checked-in CUDA 13.0 / PTX 9.0 header.

The explicit CUDA feature was not self-contained either:

```text
cargo check --offline --no-default-features --features with-cuda --bin hydrodynamic-swarm
error[E0433]: cannot find module or crate `candle_transformers` in this scope
```

The three errors came from `src/gemma.rs`, `src/gemma4.rs`, and
`src/llama.rs`. `with-cuda` activated `candle-core` and `candle-nn` CUDA
subfeatures, but did not activate the rest of `with-candle`.

## Mutation

1. `with-cuda` and `with-metal` now include `with-candle`.
2. The orphan `build.rs` was removed. A repository-wide reference search found
   no Rust or script consumer of `kernels/decay.ptx`; compiling it could not
   affect the active runtime.
3. CUDA kernel compilation is now exclusively Candle's own feature-gated path.
   The checked-in `kernels/decay.cu` and PTX remain as research sketches, but
   ordinary Cargo builds no longer rewrite them.
4. The unused direct `cc` build dependency was removed.

This patch does not add CPU generation. `src/main.rs` still deliberately
requires `Device::new_cuda(0)` for the live generation path; the CPU lane here
is the documented unit-test and compile lane.

## Verification plan

- CPU: build with only `with-candle`; require no `nvcc` message and no PTX
  checksum/worktree change.
- CUDA: build with only `with-cuda`; require the complete model dependency
  graph, successful compile through Candle's CUDA kernels, and a clean
  checked-in `kernels/decay.ptx`.
- Feature graph: inspect Cargo metadata to confirm accelerator features include
  `with-candle`.
- Tests: run the CPU-Candle suite. Model files are not required for the unit
  tests, although known tests that open missing GGUF fixtures may still fail.

## Verification receipts

- `cargo metadata --no-deps` resolves both accelerator features through
  `with-candle`.
- CPU-only `cargo check` completed successfully. It emitted no project `nvcc`
  warning, and `kernels/decay.ptx` retained checksum
  `60b8852166ff0e162ba196a8db49c1c6e247f2d4dedf0b73315a3c3ef778fa88`.
- The final explicit-CUDA check completed successfully with only `with-cuda`
  selected. It reached `candle-transformers` through the corrected feature
  graph, used Candle's CUDA build, and left the checked-in PTX hash unchanged.
- The CPU-Candle unit suite remains at the inherited baseline: 66/69 pass. Two
  failures require absent GGUF model fixtures; `config::tests::toml_parsing_works`
  is an unrelated viscosity assertion. No model generation/eval ran.
- Re-running the inherited passing set with those three baseline failures
  explicitly skipped produced 66/66 passes.

The aarch64 host checks used `RUSTFLAGS='-C target-feature=+fp16'`, matching the
repository's existing workaround for the upstream `gemm-f16` assembler issue.
The CUDA compile check used `CUDA_COMPUTE_CAP=90` because the sandbox cannot
query the host driver with `nvidia-smi`. Both checks were offline and used the
checksum-repaired vendored source tree audited separately from this branch.

## Boundary

No model smoke/eval is part of this change. The model-size scaler receipt does
not exist yet, so this work makes no claim that scaling caused any downstream
force or output trajectory.
