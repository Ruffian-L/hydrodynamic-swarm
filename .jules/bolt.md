## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-05 - Avoid .entry() clone allocations in tight loops
**Learning:** Found an anti-pattern in `src/concourse/function/mod.rs` where `counts.entry(edge.edge.clone()).or_insert(0) += 1` forced an unnecessary heap allocation or clone on every cache hit.
**Action:** Replaced with allocation-free `.get_mut()` lookups, handling inserts separately (`if let Some(count) = counts.get_mut(&edge.edge) { ... } else { counts.insert(edge.edge.clone(), 1); }`).
