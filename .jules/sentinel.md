## 2024-05-18 - Path Traversal in File Output
**Vulnerability:** Unsanitized CLI arguments used directly in file paths (`logs/crucible_{tokens}t.txt`) in `src/bin/crucible.rs`.
**Learning:** Even internal CLI tools or scripts must validate their inputs, especially if those inputs format strings that become file paths, as they can lead to path traversal attacks.
**Prevention:** Always validate numeric inputs to ensure they contain only digits before using them in file path construction.
