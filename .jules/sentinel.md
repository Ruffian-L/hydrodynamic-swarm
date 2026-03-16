## 2024-05-16 - Path Traversal Fix in Crucible CLI
**Vulnerability:** The `tokens` CLI argument was injected directly into the `log_path` string formatting without any validation, enabling path traversal if the user passed something like `../secrets`.
**Learning:** Even when checking for numeric input, using `.chars().all(|c| c.is_ascii_digit())` evaluates to true for an empty string `""`. This must always be paired with `!input.is_empty()`.
**Prevention:** Always validate and sanitize external CLI inputs before appending them to file paths or system commands. Ensure strict type enforcement.
