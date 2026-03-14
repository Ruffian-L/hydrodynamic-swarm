## 2026-03-14 - Prevent Path Traversal in crucible.rs
**Vulnerability:** The `tokens` CLI argument in `src/bin/crucible.rs` was used directly in a file path without validation, allowing path traversal (e.g., `../`).
**Learning:** Even internal tooling arguments used in file paths can lead to path traversal vulnerabilities if unsanitized. Generic string format concatenation into paths is risky.
**Prevention:** Strictly validate numeric arguments to ensure they match their expected type (e.g., using `.chars().all(|c| c.is_ascii_digit())`) before using them in file paths.
