## 2024-05-20 - Path Traversal Vulnerability in Logging

**Vulnerability:** Path traversal vulnerability in `src/bin/crucible.rs` where the `tokens` CLI argument is directly used to format the output log file path without validation.
**Learning:** Even internal tooling or benchmarking scripts must validate user inputs, as they often deal with the file system. In this case, passing `../` or similar characters could result in overwriting or creating files outside of the expected `logs/` directory.
**Prevention:** Strictly validate numeric CLI arguments (e.g. checking `.chars().all(|c| c.is_ascii_digit())`) before using them in file paths instead of doing generic alphanumeric checks or no checks at all.
