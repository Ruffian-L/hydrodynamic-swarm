## 2024-05-24 - Path Traversal Vulnerability in crucible.rs
**Vulnerability:** Unsanitized CLI argument (`tokens`) directly concatenated into file path.
**Learning:** `format!("logs/crucible_{}t.txt", tokens)` allows directory traversal if `tokens` contains `../`. Always check `is_empty()` when using `.chars().all()` because `.all()` returns true for empty strings.
**Prevention:** Validate that numeric inputs used in paths strictly contain only ASCII digits.
