## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-04 - Unnecessary heap allocations in `entry()` calls for HashMaps
**Learning:** In Rust hash maps, using `.entry(key.clone()).or_insert(0) += 1` inside loops forces an unnecessary heap allocation or clone on every single iteration (even for cache hits). For `get_edge_counts` in `src/concourse/function/mod.rs`, this was cloning the `RelationalEdge` enum on every iteration.
**Action:** For performance optimization, avoid `.entry()` in tight loops and prefer allocation-free `.get_mut(key)` lookups, handling inserts separately (`if let Some(count) = map.get_mut(key) { ... } else { map.insert(key.clone(), val); }`).
