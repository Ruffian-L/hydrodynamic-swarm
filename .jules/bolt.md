## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-22 - LRU Cache Sequential Access Fast Path
**Learning:** In the hydrodynamic swarm's LRU cache implementation (`LruCache`), sequential cache hits to the exact same key repeatedly incurred an O(N) penalty due to an `.iter().position()` scan over the underlying `VecDeque` used to track access order. This scan is completely redundant if the element is already at the back of the queue (which happens immediately after its first access).
**Action:** When updating access order tracking structures for cache hits, always check the tail of the tracking queue (`self.access_order.back()`) before initiating an O(N) scan.
