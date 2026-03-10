## 2026-03-10 - Strict Input Validation for File Paths
**Vulnerability:** Path Traversal / Command Injection via unvalidated CLI arguments used in file paths.
**Learning:** `std::env::args()` reads raw strings. When these strings are directly interpolated into file paths (e.g., `format!("logs/crucible_{}t.txt", tokens)`) without validation, it allows directory traversal attacks (e.g., `../../../tmp`). Iterating over `.chars().all(|c| c.is_ascii_digit())` is insufficient because it evaluates to `true` for empty strings.
**Prevention:** Always use strict type parsing (e.g., `parse::<u32>()`) for numeric CLI arguments before they are used in sensitive operations like file creation or command execution.
