## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-27 - Concurrency optimization with `try_join_all`
**Learning:** Found sequential asynchronous loops awaiting operations (e.g., `.await?`) one-by-one inside `src/concourse/async_patterns.rs`. This results in blocking for the duration of each loop step.
**Action:** When working with independent async tasks inside loops, push the futures into a `Vec` and resolve them simultaneously using `futures::future::try_join_all`. This significantly reduces total waiting time from the sum of all tasks to the maximum duration of a single task.
