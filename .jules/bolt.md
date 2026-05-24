## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-24 - LRU Cache Sequential Access Optimization
**Learning:** In the LRU cache implementation using a VecDeque for access order, sequential reads of the same key trigger a redundant O(N) scan to find its position.
**Action:** Always check if the requested key is already at the back of the VecDeque using `self.access_order.back()` before performing an O(N) scan for its position.
