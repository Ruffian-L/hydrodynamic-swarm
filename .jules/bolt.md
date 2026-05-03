## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.
## 2026-05-03 - Unnecessary clones in hash map entry operations
**Learning:** Found an anti-pattern in the edge counting loop (`src/concourse/function/mod.rs`) where `.entry(edge.edge.clone()).or_insert(0)` forced a heap allocation or expensive clone on every single cache hit, instead of just the first insertion.
**Action:** Always prefer `.get_mut()` for hot loop lookups on types that are expensive to clone, only falling back to `.insert()` with a clone if the key is actually missing.
