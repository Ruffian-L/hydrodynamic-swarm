## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Unnecessary O(n) operations on LRU cache hits
**Learning:** Found that `LruCache::get` performed an O(n) scan on the `access_order` (`VecDeque`) for every single cache hit to find and move the key to the back. When a cache key was requested multiple times sequentially (the most common cache hit pattern), it still performed the scan and allocation.
**Action:** Always check `self.access_order.back()` before doing an O(n) `.iter().position()` scan to handle sequential access efficiently and avoid `key.to_string()` allocations for the most recent item.
