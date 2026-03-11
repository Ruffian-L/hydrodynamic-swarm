## 2024-05-24 - Path Traversal Vulnerability in Crucible

**Vulnerability:** Path traversal via unsanitized `tokens` CLI argument in `src/bin/crucible.rs`. The script used the argument directly to formulate the output path string `logs/crucible_{tokens}t.txt` without checking its format.
**Learning:** CLI arguments passed to file path formatting strings can lead to path traversal even if they appear as intended specific formats (e.g. number of tokens) if left unchecked. Simple alphanumeric validation isn't enough to prevent directory traversal in log destinations.
**Prevention:** Strictly validate numeric CLI arguments used in file paths by checking their type using functions like `tokens.chars().all(|c| c.is_ascii_digit())` to prevent directory traversal and superficial security checks.
