## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2026-03-15 - Inefficient N-gram penalty checking via excessive allocation
**Learning:** Found an anti-pattern in the generation loop (`src/main.rs`) where N-gram repetition checking dynamically allocated a new vector for every vocabulary token, leading to O(V) allocations per step.
**Action:** Optimize N-gram repetition checking by directly scanning the previously generated tokens array for matches against the current prefix, completely eliminating all heap allocations during the penalty phase.
