## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-20 - O(N) penalty in VecDeque based LRU Cache
**Learning:** The `LruCache::get` function utilizes an O(N) `.iter().position()` scan to locate and update a key's order in its internal `VecDeque` track structure upon *every* read hit. Sequential reads of the same key caused unnecessary performance bottlenecks.
**Action:** When implementing or optimizing LRU cache `get` methods that use a `VecDeque` for tracking access order, always include a fast-path check `self.access_order.back().map(|k| k.as_str()) == Some(key)` to avoid the O(N) linear scan for sequential accesses to the same key.
