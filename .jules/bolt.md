## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-03-16 - O(N) scan overhead in LRU access order tracking
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where the LRU cache's `access_order` tracking performed an O(N) `.iter().position()` scan on every `get` and `put` operation, even for repeated sequential accesses to the same key.
**Action:** Always check the back of the queue first (`self.access_order.back()`) to short-circuit the scan for repeated accesses, creating an O(1) fast path.
