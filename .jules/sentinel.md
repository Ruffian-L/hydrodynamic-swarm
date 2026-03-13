## 2024-05-24 - Path Traversal in Crucible CLI
**Vulnerability:** Unsanitized CLI argument `tokens` in `src/bin/crucible.rs` is used directly to construct an output file path (`logs/crucible_{}t.txt`). This allows a path traversal attack where a user could provide a value like `../some_other_path` to write to arbitrary file locations.
**Learning:** Even internal CLI utilities must strictly sanitize input used in file system operations. Rust's string formatting does not implicitly prevent directory traversal sequences.
**Prevention:** Strictly validate numeric arguments by ensuring they only contain digits (e.g., `.chars().all(|c| c.is_ascii_digit())`) or by parsing them into integer types before using them in file paths.
