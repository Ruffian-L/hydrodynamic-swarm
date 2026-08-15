## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.
## 2026-03-15 - Lock contention and deadlock risk from multiple simultaneous RwLock guards
**Learning:** Found anti-patterns in asynchronous Tokio tasks (`src/concourse/swarm.rs` and `src/concourse/function/mod.rs`) where multiple `RwLock` guards (both read and write) were acquired simultaneously or unnecessarily without block-scoping or conditional checks, increasing lock contention, reducing concurrency, and raising the risk of deadlocks.
**Action:** Fetch necessary data from the first resource into local variables and explicitly drop its guard (e.g., via block scoping) before acquiring the next lock. Use conditional checks to only acquire locks when strictly necessary.
