## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-04 - Unnecessary heap allocations in edge counting with `HashMap::entry`
**Learning:** In `src/concourse/function/mod.rs` (`get_edge_counts`), using `counts.entry(edge.edge.clone()).or_insert(0) += 1` forced an unnecessary clone and heap allocation of `RelationalEdge` on every cache hit during the loop. This degraded performance during viscosity calculation.
**Action:** Replaced `.entry()` usage with allocation-free `.get_mut()` lookups inside tight loops, handling insertions separately only on cache misses (`if let Some(count) = counts.get_mut(&edge.edge) { ... } else { counts.insert(edge.edge.clone(), 1); }`).
