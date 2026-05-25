## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-25 - O(N) scan during repeated LRU access
**Learning:** Found a performance bottleneck in src/concourse/cache.rs where accessing the same cache key sequentially in the LRU cache triggers an O(N) scan of the VecDeque via iter().position(). Since VecDeque doesn't provide O(1) lookups by value, this becomes expensive for repeated hits.
**Action:** Always check if the requested key is already at the back of the access order queue (self.access_order.back()) before performing an O(N) scan, providing an O(1) fast path for sequential accesses.
