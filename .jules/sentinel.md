## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.

## 2024-03-19 - Path Hijacking Vulnerability in System Processes
**Vulnerability:** Spawning system processes using hardcoded commands like `Command::new("cargo")` is vulnerable to path hijacking if the `PATH` environment variable is manipulated.
**Learning:** System commands should use runtime environment variables provided by the parent process when available to securely resolve standard tools.
**Prevention:** Use `std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())` to resolve the `cargo` executable safely at runtime instead of relying solely on `$PATH`.
