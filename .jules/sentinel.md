## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.

## 2024-05-25 - Path Hijacking via Unqualified Command Execution
**Vulnerability:** Spawning system processes using unqualified command names like `cargo` or `nvcc` via `std::process::Command::new` allows an attacker to perform path hijacking. If the `PATH` environment variable is manipulated, it could execute a malicious payload instead of the intended tool.
**Learning:** Hardcoding standard tools as bare strings in `Command::new` relies implicitly on the environment's `PATH` variable, which may be untrusted in certain contexts.
**Prevention:** Always use runtime environment variables provided by the parent process (e.g., `std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())`) to resolve the paths of standard tools securely.
