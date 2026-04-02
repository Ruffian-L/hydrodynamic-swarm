## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-24 - Vectorized Batch Gradients
**Learning:** In `src/gpu.rs` `CpuBackend::batch_field_gradient`, mapping `probe_gradient` over positions via `get()`, `unsqueeze(0)`, and `Tensor::cat` causes severe CPU bottlenecking due to N individual allocations and synchronizations.
**Action:** Always prefer vectorized broadcast math (e.g., `unsqueeze(1)` and `broadcast_sub/mul`) over looping `unsqueeze` and `cat` for O(1) device dispatches.

## 2026-05-20 - Redundant heap allocations in high-frequency Governor loops
**Learning:** In high-frequency event loops like `ActiveCell::add_edge` in `src/concourse/governor.rs`, pushing cloned structs (like `FluxTuple`) into collections at the start of a function introduces redundant string heap allocations. The struct's fields can instead be accessed by reference during intermediate checks.
**Action:** Always defer ownership transfers (like `push(tuple)`) to the end of a function whenever possible. This avoids whole-struct clones and allows underlying fields to be safely accessed by reference during intermediate conditions, significantly reducing allocations in the hot path.
