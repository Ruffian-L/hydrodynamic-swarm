## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-05-24 - SystemTime unwrap panic vulnerability
**Vulnerability:** The codebase incorrectly calculates the duration since `UNIX_EPOCH` using `.unwrap()` on `SystemTime`. This can cause panics in systems with incorrect time settings or clock drift, potentially leading to a Denial of Service (DoS).
**Learning:** `SystemTime::now().duration_since(UNIX_EPOCH)` can return a `SystemTimeError` if the current time is set before the UNIX epoch. Default unwrapping is an unsafe assumption.
**Prevention:** Always use `.unwrap_or_default()` instead of `.unwrap()` to handle `SystemTimeError` gracefully by defaulting to a zero duration.
