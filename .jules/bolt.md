## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-24 - Vectorized Batch Gradients
**Learning:** In `src/gpu.rs` `CpuBackend::batch_field_gradient`, mapping `probe_gradient` over positions via `get()`, `unsqueeze(0)`, and `Tensor::cat` causes severe CPU bottlenecking due to N individual allocations and synchronizations.
**Action:** Always prefer vectorized broadcast math (e.g., `unsqueeze(1)` and `broadcast_sub/mul`) over looping `unsqueeze` and `cat` for O(1) device dispatches.

## 2026-03-30 - Reducing Redundant Heap Allocations in High-Frequency Processing Loops
**Learning:** In high-frequency event loops like `PrimeGovernor::add_edge`, `.clone()` on struct wrappers like `FluxTuple` (which contain `String` fields) triggers unnecessary heap allocations. Unsafe lookup implementations (`get_mut().unwrap()`) also pose a DoS panic risk if map entries are missing.
**Action:** Extract necessary fields (like `tuple.edge.clone()`) prior to transferring ownership of the struct `tuple` into collections via `push`. Replace `.unwrap()` on container lookups with safe alternatives like `.entry().or_insert()` to maintain high throughput safely without unnecessary deep cloning.
