## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2024-11-20 - Unnecessary Lock Contention Overhead in Governor State
**Learning:** In asynchronous Rust code using Tokio, repeatedly releasing a `write().await` guard to immediately re-acquire a `read().await` lock on the same shared resource (e.g., `CognitiveState` in `governor.rs`) introduces unnecessary lock contention, delays, and context-switching overhead in a hot loop (viscosity polling).
**Action:** When a read operation follows immediately after a write update on the same resource, compute the required read condition (like `is_lyapunov_stable`) while still holding the initial write guard to eliminate the redundant lock acquisition entirely.
