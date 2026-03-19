## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.

## 2024-05-24 - Path Hijacking in System Processes
**Vulnerability:** The code spawns a system process using `Command::new("cargo")`. This relies on the `$PATH` environment variable and is susceptible to path hijacking attacks.
**Learning:** Hardcoding standard tools as commands (like "cargo") when spawning processes can cause security issues if the `$PATH` is maliciously modified or improperly sanitized.
**Prevention:** Use runtime environment variables provided by the parent process (e.g., `std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())`) when executing standard tools via `std::process::Command` to ensure correct and secure resolution. DO NOT use compile-time environment variables like `env!("CARGO")` for standard tool executions in distributed code, as it hardcodes the build machine's absolute path and causes portability regressions.
