## 2024-05-15 - Cache LRU Promotion Optimization
**Learning:** In Rust LRU caches, serializing data twice (once on insertion, once on promotion) causes redundant heap allocations and CPU cycles.
**Action:** Re-use the existing serialized value during LRU cache promotion instead of deserializing and serializing again, reducing latency and memory overhead. Also, remove redundant `.clone()` calls by consuming variables that are no longer needed (like `cache_key`).
