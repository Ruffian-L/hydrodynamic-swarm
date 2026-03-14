## 2024-05-23 - Prevent Path Traversal in CLI Arguments
**Vulnerability:** Unsanitized CLI arguments used to construct file paths in `src/bin/crucible.rs` could lead to path traversal vulnerabilities.
**Learning:** Even internal CLI tools must sanitize inputs used in file system operations to prevent unintended file writes or reads. Superficial checks are insufficient; strictly validate against the expected type (e.g., digits only for numeric arguments).
**Prevention:** Always validate and sanitize user input before using it to construct file paths. Require strict character sets or type parsing (e.g., `.chars().all(|c| c.is_ascii_digit())` for integers) instead of relying on generic string manipulation.
