## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-16 - Avoid redundant entry clones in HashMap lookups
**Learning:** Using `.entry(key.clone()).or_insert(0)` inside tight loops causes unnecessary heap allocations and cloning on every lookup even for existing keys.
**Action:** Replaced `.entry()` with `.get_mut()` / `.insert()` to avoid cloning keys on cache hits.
