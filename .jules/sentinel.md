## 2024-05-18 - Fix path traversal in Crucible log file path
**Vulnerability:** Path traversal in `src/bin/crucible.rs` where the `tokens` CLI argument is directly formatted into the log file path (`logs/crucible_{}t.txt`) without validation.
**Learning:** Always validate CLI arguments, especially when they are used to construct file paths. Simple alphanumeric checks might not be enough if only digits are expected.
**Prevention:** Strictly sanitize CLI arguments according to their expected type. For example, ensure the `tokens` argument only contains ASCII digits (`.chars().all(|c| c.is_ascii_digit())`).
