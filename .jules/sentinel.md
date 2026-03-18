## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.

## 2024-05-25 - Command Path Hijacking
**Vulnerability:** Invoking system tools like `cargo` directly via `Command::new("cargo")` relies on the `PATH` environment variable, exposing the application to path hijacking if an attacker alters `PATH`.
**Learning:** Standard development tools provide their absolute paths via environment variables to child processes. Relying on `PATH` is less secure than using these provided paths.
**Prevention:** Always use runtime environment variables (e.g., `std::env::var("CARGO")`) to securely resolve standard tool executables instead of relying on `PATH` resolution.
