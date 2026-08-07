## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.
## 2026-03-15 - Efficient base-e exponentiation in physics simulation
**Learning:** Found an anti-pattern in the physics simulation (`src/concourse/physics.rs`) where `std::f64::consts::E.powf(x)` was used instead of the more efficient `x.exp()`. The `powf` function performs a generalized arbitrary-base power calculation, which is significantly slower than the optimized base-e exponential function.
**Action:** Always use `x.exp()` for calculating base-e exponentials in performance-critical code paths like physics simulations.
