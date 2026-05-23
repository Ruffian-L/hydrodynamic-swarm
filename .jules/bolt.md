## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-23 - Fast path for sequential LRU Cache hits
**Learning:** Found an opportunity in `src/concourse/cache.rs` where accessing an item in the LRU cache requires an O(N) scan using `iter().position()` to update the item's position to the back of the queue.
**Action:** Always check if the key being accessed is already at the back of the `VecDeque` via `back()` first to optimize sequential accesses to the same key, which turns it from O(N) into O(1) before attempting the full O(N) scan.
