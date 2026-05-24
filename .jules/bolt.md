## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2024-05-19 - Optimizing sequential cache hits
**Learning:** In LRU caches that use a `VecDeque` for access order tracking, sequential accesses to the same key can result in redundant O(N) `.iter().position()` scans to remove the key before pushing it to the back again. This is particularly noticeable for caches with many entries.
**Action:** Always check if the requested key is already at the back (`self.access_order.back()`) before performing the O(N) scan.
