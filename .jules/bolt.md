## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2024-05-18 - Concurrent Async Polling in the Swarm Scheduler
**Learning:** In the Swarm's `WorkStealingScheduler`, methods like `get_stats()` and `shutdown()` were iterating over `self.worker_pools` and sequentially `await`ing operations on each pool (`pool.get_stats().await?` and `pool.shutdown().await?`). This causes the total duration to be the sum of all individual durations rather than parallelizing the asynchronous tasks.
**Action:** When aggregating results or initiating operations across multiple independent asynchronous actors (like worker pools), push the futures into a `Vec` and use `futures::future::try_join_all` (or `join_all`) to execute them concurrently, bounded only by the longest individual execution time.
