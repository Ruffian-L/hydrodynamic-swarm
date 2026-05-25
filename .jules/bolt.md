## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-25 - O(N) scan overhead during sequential access in LRU cache
**Learning:** Found a performance bottleneck in `src/concourse/cache.rs` where accessing the same cache key multiple times sequentially in the `LruCache` triggers an O(N) `.iter().position()` scan through the `access_order` double-ended queue.
**Action:** When tracking access order with a `VecDeque`, always check if the requested key is already the most recently used (e.g., `self.access_order.back()`) before performing linear scans, to avoid redundant reordering overhead on repeated access.
