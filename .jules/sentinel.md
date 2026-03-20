## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-03-24 - Fix potential unwrap panic on system time conversion
**Vulnerability:** Application panic due to clock drift or system time set before Unix Epoch.
**Learning:** `SystemTime::now().duration_since(UNIX_EPOCH)` can fail and return an error if the system time is set incorrectly (e.g. before Jan 1, 1970). Using `.unwrap()` on this result can cause a hard crash, creating a potential Denial of Service (DoS) vector.
**Prevention:** Always use `.unwrap_or_default()` or explicitly handle the `SystemTimeError` instead of unwrapping, allowing the application to degrade gracefully (defaulting to a 0 duration) rather than crashing.
