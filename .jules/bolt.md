## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-26 - O(N) LRU Cache sequential access hit
**Learning:** `VecDeque::iter().position()` in the LRU `get` method causes an unnecessary O(N) scan even for repeated accesses to the same key, which are extremely common when nodes sequentially access their cached states.
**Action:** Always check `self.access_order.back()` before doing O(N) scans for sequential cache hits.
