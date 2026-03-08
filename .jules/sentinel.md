## 2024-05-01 - [Path Traversal in Crucible CLI Tokens Argument]
**Vulnerability:** A path traversal vulnerability existed in `src/bin/crucible.rs` where the unsanitized `tokens` CLI argument was directly interpolated into the log file path (`logs/crucible_{}t.txt`), allowing arbitrary file writes (e.g. using `../../../foo` as the argument).
**Learning:** CLI arguments that are incorporated into file paths must always be sanitized or validated, even in testing or evaluation tools, to prevent unintended file system modifications or access.
**Prevention:** Always validate that such inputs contain only expected characters (e.g., enforcing `.chars().all(|c| c.is_ascii_alphanumeric())` for tokens) before using them in `std::fs::File::create` or format strings for paths.
