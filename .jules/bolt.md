## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-05-27 - Reduce lock contention and mathematical overhead in Governor Loop
**Learning:** In highly concurrent areas like the Governor loop, repeatedly acquiring `read().await` locks on the same resource (like `active_cell`) across different steps adds severe contention and await overhead. Furthermore, replacing generalized arbitrary base exponents `E.powf(x)` with natural exponentiation `(x).exp()` offers non-trivial performance improvements mathematically.
**Action:** Always batch related state reads into a single `.read().await` lock to avoid repeated async context switches and contention. Always prefer `.exp()` over `E.powf()` for base-e calculations.
