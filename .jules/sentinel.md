## 2024-05-24 - Path Traversal in File Output
**Vulnerability:** User-controlled CLI argument `tokens` was directly interpolated into a log file path without validation, allowing directory traversal or writing to arbitrary locations.
**Learning:** When validating numeric CLI arguments used in file paths, strictly enforcing `.chars().all(|c| c.is_ascii_digit())` and `!tokens.is_empty()` prevents path manipulation.
**Prevention:** Always validate and sanitize user inputs that dictate file system access, avoiding string concatenation for path construction.
