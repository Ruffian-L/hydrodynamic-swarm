## 2025-03-11 - [Path Traversal in CLI Logs]
**Vulnerability:** Path traversal vulnerability via unsanitized CLI arguments (`tokens`) used directly in log file paths (`logs/crucible_{}t.txt`). Generic alphanumeric checks are insufficient.
**Learning:** Even internal testing or utility scripts can be vulnerable if user input is passed into file paths without strict type validation.
**Prevention:** Always validate numeric CLI arguments strictly according to their expected type (e.g., `.chars().all(|c| c.is_ascii_digit())`) before using them in file paths or other sensitive operations.
