## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2024-08-14 - Optimize Sequential Lock Acquisitions
**Learning:** In asynchronous Rust code (Tokio), dropping an `RwLock::write().await` guard to immediately re-acquire an `RwLock::read().await` lock on the same resource introduces unnecessary context switching and lock contention. The same applies for sequential reads and writes on different locks like `ttl_cache` and `lru_cache`, where dropping the read guard before acquiring the write guard avoids deadlocks and reduces contention.
**Action:** Compute required read conditions (like `is_lyapunov_stable`) while still holding the initial write lock guard. For operations involving multiple locks (like promoting from `ttl_cache` to `lru_cache`), explicitly read required values, drop the read guard using `drop()`, and then acquire the write guard.
