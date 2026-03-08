## 2024-05-18 - [Path Traversal in CLI Arguments]
**Vulnerability:** The `tokens` CLI argument in `src/bin/crucible.rs` was directly interpolated into a file path (`logs/crucible_{}t.txt`) without validation, allowing path traversal (e.g., passing `../`).
**Learning:** CLI arguments used in file paths must be strictly sanitized. In Rust, checking `.chars().all(|c| c.is_ascii_alphanumeric())` is an effective way to prevent path hijacking.
**Prevention:** Always sanitize or validate CLI inputs before using them in filesystem operations.