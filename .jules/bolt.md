## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.
## 2026-08-18 - Redundant RwLock reads and lock contention in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs` and `src/concourse/swarm.rs`) where `RwLock` locks on the same resources (`active_cell` and `cognitive_state`) were held longer than necessary or redundantly re-acquired. In `governor.rs`, dropping a write lock to re-acquire a read lock for Lyapunov stability checks introduced unnecessary context switching. In `swarm.rs`, acquiring multiple read locks concurrently and holding them across calculations increased lock contention.
**Action:** Compute required read conditions while still holding the initial write lock guard to guarantee atomicity. For multiple read locks, fetch necessary data into local variables and explicitly drop guards early before acquiring the next lock.
