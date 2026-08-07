## 2026-03-15 - Redundant heap allocations during LRU promotion in TTL Caching
**Learning:** Found an anti-pattern in the caching layer (`src/concourse/cache.rs`) where TTL entries promoted to LRU cache were unnecessarily re-serializing data via `bincode::serialize` that was already serialized in `entry.value`.
**Action:** Always reuse the existing serialized byte vectors (`entry.value.clone()`) and transfer ownership of strings (`cache_key`) directly when performing caching promotions across layers to avoid triggering new heap allocations.

## 2026-03-15 - Redundant RwLock reads in async concurrency sequences
**Learning:** Found an anti-pattern in asynchronous Tokio tasks (`src/concourse/governor.rs`) where `RwLock::read().await` was repeatedly acquired on the same resource within a single concurrent sequence, causing severe lock contention and context-switching overhead.
**Action:** Batch state reads into a single `.read().await` lock acquisition and capture the required fields to optimize asynchronous performance.

## 2024-08-07 - Inefficient base-e exponential calculation
**Learning:** Found an anti-pattern in the math layer (`src/concourse/physics.rs`) where calculating base-e exponentials using `std::f64::consts::E.powf(x)` is less efficient than using the native `(x).exp()` method because the former performs a generalized arbitrary-base power calculation.
**Action:** In Rust, calculating base-e exponentials using `(x).exp()` is more efficient than using `std::f64::consts::E.powf(x)`. Always use `.exp()` for base-e calculations.
