## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-28 - Batched RwLock reads to reduce contention
**Learning:** Found redundant lock contention where multiple RwLock::read().await calls on active_cell were made sequentially within execute_splat.
**Action:** Batch state reads into a single .read().await lock acquisition and capture the required fields to minimize lock contention in async contexts.
