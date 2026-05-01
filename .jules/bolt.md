## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-24 - Avoiding String allocations in access order queue updates
**Learning:** In LRU cache updates (`src/concourse/cache.rs`), removing an element and then appending `key.to_string()` triggers an unnecessary heap allocation on every cache hit. Extracting the element via `remove().unwrap()` and re-inserting it avoids this overhead.
**Action:** Always reuse existing strings when updating access queues or similar data structures instead of allocating new strings.
