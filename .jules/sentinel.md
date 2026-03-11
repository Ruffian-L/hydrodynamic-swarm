## 2024-05-24 - Unsanitized User Input in Log Paths
**Vulnerability:** In `src/bin/crucible.rs`, the `tokens` CLI argument is directly incorporated into the log file path (`logs/crucible_{}t.txt`) without validation. This allows a path traversal vulnerability where a user could pass a string like `../foo` to create files outside the intended directory.
**Learning:** Even internal testing or CLI tools need input validation when that input dictates file system operations.
**Prevention:** Validate that the `tokens` argument contains only numeric characters.
