# 2026-08-27 — Cache Model SHA256 for Fast Loader Boot

We did: Added an mtime-and-size based cache in `sha256_file` (saving to `data/.sha256_cache/`) to avoid rehashing the 20GB+ GGUF files every time the `hydrodynamic-swarm` boots up.

We think: The scaler receipt builder was strictly hashing the model binary synchronously on the main thread for provenance. On large models (like 27B) or slow storage, this added 1-2 minutes of blocked I/O every single boot. By hashing the path, size, and mtime, we can safely cache the heavy `sha256sum` step and bypass the wait.

Next: The chat REPL and swarm loops should boot instantly once the field is live, preserving the user's momentum.
