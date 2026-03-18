## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-25 - Top-K Extraction with select_nth_unstable_by
**Learning:** Found O(N log N) `sort_by` being used for Top-K extraction (e.g., pruning splats). `select_nth_unstable_by` is O(N) and should be used instead when full sorting is not required or when sorting only the top K elements.
**Action:** Use `select_nth_unstable_by` for partial sorting of large arrays instead of full `sort_by` when extracting top-K items.
