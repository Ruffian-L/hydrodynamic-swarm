## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.
## 2023-10-27 - [Swarm Metrics Lock Contention]
**Learning:** In asynchronous Rust code, acquiring multiple `RwLock` guards simultaneously (e.g., locking `active_cell` and then `cognitive_state` while still holding the guard for `active_cell`) increases lock contention, reduces concurrency, and heightens the risk of deadlocks. The `get_metrics` method in `SwarmMatrix` acquired two read locks simultaneously to fetch fields.
**Action:** To optimize performance and safety, fetch necessary data from the first resource into local variables and explicitly drop its guard (e.g., via block scoping) before acquiring the next lock.

## 2023-10-27 - [Governor Lock Contention]
**Learning:** The `poll_viscosity` method in `PrimeGovernor` acquired a write lock on `cognitive_state` and dropped it, only to immediately re-acquire a read lock on the same resource a few lines down to check Lyapunov stability.
**Action:** Compute required read conditions (like Lyapunov stability) while still holding the initial write lock guard. This eliminates a redundant lock acquisition and context-switching overhead in the hot path.
