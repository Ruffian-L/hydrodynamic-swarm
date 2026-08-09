## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.
## 2026-08-09 - Levenshtein Distance Memory Optimization
**Learning:** The Levenshtein distance algorithm calculating semantic distance originally allocated an O(N*M) 2D vector for dynamic programming, which causes excessive memory consumption and allocation overhead on large strings. Since each step of the DP only relies on the current and immediate previous row, maintaining the full matrix is unnecessary.
**Action:** Optimize Levenshtein distance calculations by maintaining only two 1D vectors (`prev_row` and `curr_row`), reducing memory complexity from O(N*M) to O(M). This minimizes memory overhead, especially when analyzing long semantic hashes.
