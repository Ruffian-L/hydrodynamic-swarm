## 2025-03-08 - Path Hijacking in crucible.rs
**Vulnerability:** The crucible test runner script executed `Command::new("target/release/hydrodynamic-swarm")` using a relative path. If executed from another directory while a malicious binary was present in a matching relative path, it could execute the wrong file.
**Learning:** `cargo run` sets `CARGO_MANIFEST_DIR` reliably, but raw binaries or sub-commands executed directly do not natively guarantee safe CWDs for inner spawned processes.
**Prevention:** Use `env!("CARGO_MANIFEST_DIR")` to construct absolute paths when spawning internal cargo binaries.
