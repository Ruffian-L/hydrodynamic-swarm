## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-03-15 - Unnecessary entry() allocations in tight loops
**Learning:** In Rust hash maps, using `.entry(key.clone()).or_insert(...)` inside a tight loop forces an unnecessary heap allocation or clone on every single cache hit.
**Action:** For performance optimization, avoid `.entry()` in tight loops and prefer allocation-free `.get_mut(key)` lookups, handling inserts separately (`if let Some(val) = map.get_mut(key) { ... } else { map.insert(key.clone(), val); }`).
