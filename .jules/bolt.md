## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-23 - Optimize sequential LRU cache accesses
**Learning:** LRU cache `get` operations on `VecDeque`-based access orders incur O(N) penalties during sequential accesses to the same keys due to full iteration scans (`.iter().position()`).
**Action:** Always check if the requested key is already at the back (`self.access_order.back()`) before performing an O(N) scan, providing a fast path for hot sequential accesses.
