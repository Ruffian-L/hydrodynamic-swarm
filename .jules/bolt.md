## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Tensor Batch Retrieval Optimization
**Learning:** In `candle_core`, looping with `Tensor::get().unsqueeze()` and `Tensor::cat()` for batch row retrieval causes significant performance degradation due to intermediate allocations and host-device syncs.
**Action:** Always prefer `Tensor::index_select` with a 1D index tensor (converted to `u32` via `Tensor::from_vec(indices, (len,), device)`) for optimal batch row retrieval.

## 2026-03-31 - Missing database index on frequently queried fields in TacoDb
**Learning:** Found a missing database index in `src/logger.rs` where `TacoDb::query_steps` queried by `session_id` and `entry_type` without an index, resulting in full table scans.
**Action:** Always create compound indexes (e.g., `CREATE INDEX IF NOT EXISTS idx_taco_session_type ON taco_entries(session_id, entry_type)`) for SQLite tables when they are frequently queried by multiple fields.
