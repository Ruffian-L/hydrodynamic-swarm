## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-24 - Implement Copy for tiny enums
**Learning:** `RelationalEdge` and `NodeClass` were only deriving `Clone`, leading to unnecessary `.clone()` calls and allocations in hot paths like calculating edge counts inside locks (`get_edge_counts_vec`) or inserting into frequency maps. Small, trivially copyable enums should always derive `Copy`.
**Action:** Always derive `Copy` for lightweight enums to avoid `.clone()` overhead, especially when they are frequently used as hash map keys or iterated over in performance-critical loops.
