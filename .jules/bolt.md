## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-21 - LRU Cache Sequential Access Optimization
**Learning:** In LRU caches backed by a `VecDeque` for access order, repeatedly accessing the same key sequentially triggers an O(N) `.iter().position()` scan if we blindly try to update the order. This is a common pattern when a cache hit is immediately followed by another use of the same data.
**Action:** When optimizing LRU `get` methods, always add a fast path `if self.access_order.back().map(|k| k.as_str()) != Some(key)` to check if the requested key is already the most recently accessed item. This avoids the O(N) scan entirely for sequential accesses to the same key.
