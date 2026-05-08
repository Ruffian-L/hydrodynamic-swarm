## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-08 - Idiomatic Copy Trait Optimization for Enums in Hot Paths
**Learning:** Adding the `Copy` trait to lightweight, fieldless enums (like `RelationalEdge` and `NodeClass`) is a zero-cost abstraction that allows using idiomatic Rust APIs (like `HashMap::entry`) without incurring `.clone()` overhead in tight loops. Attempting to bypass `entry` with verbose `get_mut`/`insert` blocks for simple types is an anti-pattern that sacrifices code readability for no performance gain.
**Action:** Always verify if a type can trivially derive `Copy` before implementing complex access patterns to avoid allocations. If it can, derive `Copy` and retain standard idiomatic APIs.
