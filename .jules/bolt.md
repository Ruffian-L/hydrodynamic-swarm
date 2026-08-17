## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2026-03-15 - Redundant lock re-acquisitions in async Rust sequence
**Learning:** Found an anti-pattern in `PrimeGovernor::poll_viscosity()` where an `RwLock::write().await` guard on `self.cognitive_state` was dropped and immediately re-acquired as `RwLock::read().await` to check `is_lyapunov_stable()`. This introduces an unnecessary lock contention point and a potential race condition.
**Action:** Always capture necessary post-update state properties (e.g., calling `is_lyapunov_stable()`) while still holding the original `write().await` guard and return the result to a local variable. This guarantees atomicity and reduces context-switching overhead.
