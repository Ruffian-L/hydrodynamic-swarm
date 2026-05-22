## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-22 - LRU Cache sequential access optimization
**Learning:** Sequential accesses to the same key in LRU cache result in O(N) scans due to `.iter().position()` searching through the entire history queue for every access, unnecessarily shifting memory.
**Action:** Always check the back of the queue (most recently used) before performing an O(N) position scan on subsequent accesses, reducing duplicate operations.
