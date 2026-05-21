## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2024-05-21 - O(N) penalty on sequential cache hits
**Learning:** The LRU cache implementation (`LruCache::get`) unnecessarily searched and mutated the `access_order` (`VecDeque`) even when the requested key was already the most recently accessed (at the back of the queue). This turned back-to-back cache hits for the same item into an O(N) scan.
**Action:** Always check the `back()` of the queue (`self.access_order.back()`) and skip the O(N) `.iter().position()` scan and update if the item is already the most recently accessed.
