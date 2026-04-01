## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-05-24 - SystemTime Clock Drift Panic
**Vulnerability:** SystemTime::now().duration_since(UNIX_EPOCH).unwrap() will panic and crash the application if the system clock drifts or is misconfigured to a time before January 1, 1970. This creates a reliability and potential Denial of Service (DoS) issue.
**Learning:** Never assume the system clock is perfectly synced or monotonically increasing relative to the Unix Epoch when calculating timestamps, especially in logging or utility code that runs frequently.
**Prevention:** Use `.duration_since(UNIX_EPOCH).unwrap_or_default()` instead of `.unwrap()` to gracefully handle `SystemTimeError` by returning a zero duration, preventing application crashes.
## 2024-04-01 - Container Lookup Denial of Service (DoS) Panic
**Vulnerability:** In src/concourse/governor.rs, the add_edge method uses unwrap() directly on a container lookup (edge_counts.get_mut(&tuple.edge).unwrap()). If an uninitialized or unexpected edge type is processed, this will cause the application to panic and crash.
**Learning:** Using unwrap() on dictionary or map lookups in high-frequency event loops or state updates creates a critical DoS vector because a single malformed or unhandled state input can bring down the entire system daemon.
**Prevention:** Always use safe accessors like if let Some(val) or pattern matching when reading or mutating dynamically accessed state containers, ensuring the application handles missing keys gracefully without panicking.
