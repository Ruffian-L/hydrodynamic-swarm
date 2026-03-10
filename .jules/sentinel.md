## 2024-05-24 - Prevent Path Traversal in File Output
**Vulnerability:** The `src/bin/crucible.rs` binary accepted an unsanitized `tokens` argument from the CLI and directly used it to construct a file output path (`format!("logs/crucible_{}t.txt", tokens)`), allowing a path traversal attack via inputs like `../`.
**Learning:** Unsanitized CLI inputs used in file system operations represent an immediate path traversal risk, even if they are expected to be numeric arguments.
**Prevention:** Always strictly validate and sanitize inputs used in file paths against expected types (e.g., requiring `.chars().all(|c| c.is_ascii_digit())` for numeric values) to prevent manipulation.
