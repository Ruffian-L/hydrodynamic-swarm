## 2024-05-28 - Strict Numeric Sanitization for File Paths
**Vulnerability:** Path traversal vulnerability in `src/bin/crucible.rs` allowed arbitrary strings in the `tokens` CLI argument to dictate the generated log file path.
**Learning:** Generic alphanumeric checks or relying on user intent is insufficient. Numeric arguments used in file paths must be strictly sanitized.
**Prevention:** Validate that string arguments representing integers strictly contain ASCII digits (`.chars().all(|c| c.is_ascii_digit())`) before utilizing them in file paths.
