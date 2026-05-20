## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-20 - Avoid O(N) scan during repeated LRU cache accesses
**Learning:** The LRU cache `get` method (`LruCache::get` in `src/concourse/cache.rs`) performs an O(N) linear scan of the `access_order` `VecDeque` using `self.access_order.iter().position(|k| k == key)` on every valid cache hit, even when the key accessed is already the most recently accessed key (at the back of the queue). This redundant scan can cause performance bottlenecks for workloads with high temporal locality on the same keys.
**Action:** Always optimize sequential or clustered access patterns in `VecDeque`-backed LRU cache access order updates by first checking if the requested key is already at the back (`self.access_order.back()`) before performing an expensive O(N) linear scan and queue manipulation.
