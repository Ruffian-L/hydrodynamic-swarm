## 2026-03-12 - Path Traversal via CLI Arguments
**Vulnerability:** The `crucible` utility uses the unsanitized `tokens` CLI argument to construct a file path (`logs/crucible_{tokens}t.txt`), allowing an attacker to overwrite arbitrary files using path traversal sequences like `../`.
**Learning:** CLI arguments used in file paths must be strictly validated according to their expected type (e.g., positive integers) rather than just alphanumeric checks.
**Prevention:** Always validate numeric CLI arguments using `.chars().all(|c| c.is_ascii_digit())` before using them in file paths.
