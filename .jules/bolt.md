## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-03-16 - Prevent unnecessary string cloning in hash map iteration
**Learning:** Found a performance bottleneck in `src/concourse/function/mod.rs` where calling `.entry(edge.edge.clone()).or_insert(0)` inside a high-frequency loop forces a heap allocation for the string key on every single iteration, even for existing entries.
**Action:** Replace `.entry(key.clone())` with a `.get_mut(key)` lookup combined with a separate `insert(key.clone())` fallback to ensure heap allocation only occurs when a new key is actually being added to the map.
