## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-27 - O(N) penalties in VecDeque for sequential LRU Cache accesses
**Learning:** In LRU cache implementations that manage access order with a `VecDeque`, sequential `get` accesses for the same key incur an unnecessary O(N) scan overhead (`iter().position()`) just to realize the key is already the most recently used element.
**Action:** Short-circuit access updates by explicitly checking if the requested key is already at the back of the queue (`self.access_order.back()`) before initiating the O(N) position scan.
