## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2023-10-24 - Async RwLock Read/Write Early Dropping
**Learning:** Sequential `.read().await` or `.write().await` locks inside the same function can block other asynchronous tasks from resolving their states efficiently. Attempting to drop locks and immediately acquire them again on the same resource causes excessive context switching.
**Action:** Always fetch the necessary inner data into scoped local variables and drop the guard explicitly before doing any downstream computation or subsequent asynchronous locks. Compute read condition checks using the pre-existing write lock when valid.
