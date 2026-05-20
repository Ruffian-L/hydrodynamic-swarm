## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2024-05-20 - LRU Cache Sequential Access Bottleneck
**Learning:** O(N) access_order.iter().position() lookups in VecDeque-backed LRU caches cause severe slowdowns for sequential accesses to the same key.
**Action:** Always check the back of the access_order queue before performing an O(N) scan.
