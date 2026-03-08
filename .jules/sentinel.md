## 2026-03-08 - [Path Traversal in Crucible Logs]
**Vulnerability:** The CLI tokens argument is interpolated directly into the `logs/crucible_{}t.txt` file path without validation, allowing directory traversal.
**Learning:** Unsanitized CLI inputs used in file paths can lead to arbitrary file creation/overwrites outside intended directories.
**Prevention:** Apply rigorous alphanumeric validation (e.g., `.chars().all(|c| c.is_ascii_alphanumeric())`) to any untrusted string used to construct filesystem paths.
