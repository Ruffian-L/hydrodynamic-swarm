## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-24 - Vectorized Batch Gradients
**Learning:** In `src/gpu.rs` `CpuBackend::batch_field_gradient`, mapping `probe_gradient` over positions via `get()`, `unsqueeze(0)`, and `Tensor::cat` causes severe CPU bottlenecking due to N individual allocations and synchronizations.
**Action:** Always prefer vectorized broadcast math (e.g., `unsqueeze(1)` and `broadcast_sub/mul`) over looping `unsqueeze` and `cat` for O(1) device dispatches.

## 2026-03-24 - Avoiding clone in cache operations
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where keys were unnecessarily cloned when passed into HashMap operations or during map cleaning. In `LruCache::put` the key is cloned into both `entries` map and `access_order`, but we can pass ownership to one and clone for the other. During `cleanup` we cloned keys just to remove them.
**Action:** Minimize key cloning by reorganizing logic, for instance using `retain` for cleanups to avoid creating intermediate vecs of cloned keys.

## 2026-03-24 - Avoiding clone in ActiveCell::add_edge
**Learning:** In highly concurrent event-driven architectures like `src/concourse/governor.rs`, receiving elements over channels (`FluxTuple`s) and then calling `.clone()` multiple times (like inserting the `String` into `HashMap` nodes and the tuple into `Vec`) adds unnecessary heap allocations. By transferring ownership using `remove` or splitting tuples and only cloning what's necessary (or modifying `add_edge` to accept `FluxTuple` and destructuring it) we can avoid overhead.
**Action:** Avoid `.clone()` in high-frequency event processing loops like `ActiveCell::add_edge` by transferring ownership where possible.
