# Sentinel Journal
## 2024-05-24 - Path Traversal Vulnerability in `crucible.rs`
**Vulnerability:** The `crucible` utility uses the unsanitized `tokens` CLI argument to construct a file path for logging (`format!("logs/crucible_{}t.txt", tokens)`). If a user provides an argument like `../target`, this leads to writing files in unexpected locations, resulting in a path traversal vulnerability.
**Learning:** Even internal CLI utilities must validate arguments if they are used to build file paths. A malicious actor with access to the `crucible` script could exploit this behavior.
**Prevention:** Strictly validate that arguments used for constructing filenames or paths adhere to their expected format. For numeric counts like `tokens`, verify `chars().all(|c| c.is_ascii_digit())` to enforce an integer before interpolation.
