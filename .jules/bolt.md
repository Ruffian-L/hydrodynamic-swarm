## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-24 - Vectorized Batch Gradients
**Learning:** In `src/gpu.rs` `CpuBackend::batch_field_gradient`, mapping `probe_gradient` over positions via `get()`, `unsqueeze(0)`, and `Tensor::cat` causes severe CPU bottlenecking due to N individual allocations and synchronizations.
**Action:** Always prefer vectorized broadcast math (e.g., `unsqueeze(1)` and `broadcast_sub/mul`) over looping `unsqueeze` and `cat` for O(1) device dispatches.

## 2026-03-24 - Tensor Operations Optimization
**Learning:** In `candle_core` implementations like `src/memory.rs` in `hydrodynamic-swarm`, calling `.to_scalar()` or performing tensor allocations (`.affine()`, `+`) inside per-item iteration loops causes severe CPU-GPU synchronization bottlenecks.
**Action:** A proven optimization is to implement early-exit checks (e.g., `if scale.abs() < 1e-7 { continue; }`) to skip these expensive tensor operations when force contributions are negligible.

## 2026-03-24 - Unnecessary Allocations in Cache Lookups
**Learning:** In Rust hash maps (like `src/concourse/cache.rs`), using `.entry(key.to_string())` for simple `.get()` lookups forces a heap allocation for the `String` key on every single cache hit, even when the key already exists or is missing.
**Action:** Always prefer `.get(key)` combined with conditional `.remove(key)` when avoiding allocations during cache lookups. The borrow checker (NLL) prevents conditionally mutating the map if a reference is returned from the same block. Perform a second `.get(key)` inside the else block to safely return the reference.
