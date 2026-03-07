## 2024-05-15 - [Initial Sentinel Setup]
**Vulnerability:** N/A
**Learning:** Initial Sentinel file created.
**Prevention:** N/A

## 2024-05-15 - [Path Traversal in CLI Argument]
**Vulnerability:** Path Traversal via unvalidated CLI arguments (e.g., in `src/bin/crucible.rs`)
**Learning:** Raw command line arguments (like `tokens` count) injected directly into file path formats (e.g. `format!("logs/crucible_{}t.txt", tokens)`) allow users to traverse directories using `../` inputs.
**Prevention:** Validate and sanitize all user input before using it in file system operations. For simple metrics like token counts or identifiers, ensure the input is purely alphanumeric using `chars().all(|c| c.is_ascii_alphanumeric())`.
