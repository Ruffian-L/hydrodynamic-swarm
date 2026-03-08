## 2024-05-24 - Path Traversal in CLI Args
**Vulnerability:** The `tokens` CLI argument in `crucible.rs` was used directly in a file path without validation, allowing path traversal (e.g., writing logs outside the `logs/` directory).
**Learning:** Even internal CLI tools or evaluation scripts need strict input validation if user-provided arguments are used to construct file paths, to prevent arbitrary file writes.
**Prevention:** Always sanitize CLI arguments used in file paths (e.g., using `.chars().all(|c| c.is_ascii_alphanumeric())`) before using them in file operations.
