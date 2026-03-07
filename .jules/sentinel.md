## 2025-01-22 - Predictable Cache Keys Using DefaultHasher

**Vulnerability:** In `src/concourse/cache.rs`, `std::collections::hash_map::DefaultHasher` is used to generate cache keys (in `generate_cache_key` and `generate_edge_key`). Rust's `DefaultHasher` is intentionally designed to be non-deterministic across executions to prevent HashDoS. When used to generate string keys for a cache (especially one that might be serialized or persisted in the future, as `TtlCache` often implies), the keys will change every time the process restarts, breaking the cache. Moreover, a 64-bit hash has a non-negligible collision probability when caching millions of embeddings, leading to potential Cache Poisoning (returning the wrong embedding for a text).

**Learning:** `DefaultHasher` is strictly for in-memory, transient hash tables (`HashMap`/`HashSet`). It must never be used to derive stable, unique identifiers or cache keys from content.

**Prevention:** Use a stable cryptographic hash function like SHA-256 (via the `sha2` crate) or BLAKE3 to generate deterministic, collision-resistant string keys from content for caching purposes.
