## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-05-24 - SystemTime Clock Drift Panic
**Vulnerability:** SystemTime::now().duration_since(UNIX_EPOCH).unwrap() will panic and crash the application if the system clock drifts or is misconfigured to a time before January 1, 1970. This creates a reliability and potential Denial of Service (DoS) issue.
**Learning:** Never assume the system clock is perfectly synced or monotonically increasing relative to the Unix Epoch when calculating timestamps, especially in logging or utility code that runs frequently.
**Prevention:** Use `.duration_since(UNIX_EPOCH).unwrap_or_default()` instead of `.unwrap()` to gracefully handle `SystemTimeError` by returning a zero duration, preventing application crashes.
## 2024-05-24 - Unsafe Map Lookup Panic DoS
**Vulnerability:** In `ActiveCell::add_edge`, `self.edge_counts.get_mut(&tuple.edge).unwrap()` is used. If an unknown edge type is processed, it will panic and crash the application, leading to a Denial of Service.
**Learning:** Using `unwrap()` on container lookups in event-driven systems is highly dangerous. While `.entry().or_insert()` prevents panics, it can introduce an Out-Of-Memory (OOM) DoS vulnerability by allowing unbounded map growth if an attacker spams unknown keys.
**Prevention:** Use pattern matching like `if let Some(count) = map.get_mut(key)` to safely handle lookups and ignore unknown/malicious keys without panicking or allocating unbounded memory.
