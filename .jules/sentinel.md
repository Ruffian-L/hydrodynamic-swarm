## 2024-11-20 - Prevent Path Traversal in Log Output Paths
**Vulnerability:** Path traversal vulnerability in `src/bin/crucible.rs`. The `tokens` CLI argument was directly interpolated into the log file path string `format!("logs/crucible_{}t.txt", tokens)` without any validation.
**Learning:** Even internal testing scripts or command-line utilities can be vulnerable to path traversal if user-provided CLI arguments are used in file paths without proper sanitization.
**Prevention:** Always validate numeric CLI arguments intended for file paths using strict type checks, such as `tokens.chars().all(|c| c.is_ascii_digit())`, rather than just using `unwrap_or("default")`.
