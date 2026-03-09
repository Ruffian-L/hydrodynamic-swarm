## 2024-03-09 - Path Traversal Vulnerability in Crucible

**Vulnerability:** The `crucible` binary accepted a raw command line argument (`tokens`) and used it directly in `format!("logs/crucible_{}t.txt", tokens)`, passing the result to `fs::File::create()`. This allowed path traversal to create files outside the `logs/` directory.

**Learning:** Unsanitized user inputs from the command line can lead to serious filesystem vulnerabilities when dynamically interpolating paths, even in seemingly benign utility scripts. General string inputs meant for numeric usage must be actively constrained.

**Prevention:** Ensure numeric CLI arguments used in file paths (e.g., the `tokens` argument in `crucible.rs`) are strictly sanitized according to their expected type. For instance, requiring `.chars().all(|c| c.is_ascii_digit())` rather than generic alphanumeric checks to prevent path traversal vulnerabilities.
