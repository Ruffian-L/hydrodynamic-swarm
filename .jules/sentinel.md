## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.

## 2024-05-24 - Fix clock drift panic DoS vulnerability
**Vulnerability:** The use of `.unwrap()` on `.duration_since(std::time::UNIX_EPOCH)` in `src/main.rs` can cause the application to panic and crash on systems with clock drift or incorrectly set system times (i.e. before January 1, 1970). This exposes a denial of service (DoS) vulnerability.
**Learning:** Panicking on time retrieval errors can lead to application crashes, especially on systems with unreliable clocks.
**Prevention:** Change `.unwrap()` to `.unwrap_or_default()` to handle `SystemTimeError` gracefully without panicking.
