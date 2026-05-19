## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-05-24 - SystemTime Clock Drift Panic
**Vulnerability:** SystemTime::now().duration_since(UNIX_EPOCH).unwrap() will panic and crash the application if the system clock drifts or is misconfigured to a time before January 1, 1970. This creates a reliability and potential Denial of Service (DoS) issue.
**Learning:** Never assume the system clock is perfectly synced or monotonically increasing relative to the Unix Epoch when calculating timestamps, especially in logging or utility code that runs frequently.
**Prevention:** Use `.duration_since(UNIX_EPOCH).unwrap_or_default()` instead of `.unwrap()` to gracefully handle `SystemTimeError` by returning a zero duration, preventing application crashes.
## 2024-05-24 - Missing Timeout on External API Call
**Vulnerability:** The HTTP client in `GrokOracle` was initialized without a timeout, making it vulnerable to infinite hangs if the external API is unresponsive, leading to resource exhaustion (DoS).
**Learning:** Always configure reasonable timeouts for network requests, especially to third-party APIs, to ensure the system remains responsive and does not leak resources or hang indefinitely.
**Prevention:** Use `.timeout(std::time::Duration::from_secs(X))` when building `reqwest::Client` configurations.
## 2024-05-24 - TOCTOU and Predictable File Overwrite in Logger
**Vulnerability:** The session logger in `logger.rs` constructed log filenames using the current time and a label, then blindly opened them with `File::create()`. If an attacker can predict the filename (which is easy as it is based on the clock), they can place a symlink at that path pointing to a sensitive file (e.g., `/etc/passwd`). `File::create()` follows symlinks, so the application would overwrite the target file with its own logs.
**Learning:** `File::create()` truncates and opens files, following symlinks if they exist. In directories where multiple users or processes might write, this creates a classic Time-of-Check to Time-of-Use (TOCTOU) vulnerability.
**Prevention:** Always use `std::fs::OpenOptions::new().write(true).create_new(true).open(&path)` when creating new files, especially in shared or predictable locations. The `.create_new(true)` flag ensures the operation fails atomically if the file (or a symlink) already exists.

## 2024-05-24 - Relative Path Hijacking in Subprocess Execution
**Vulnerability:** The internal `crucible.rs` tool executed the main binary using a relative path (`target/release/hydrodynamic-swarm`). If a user or script executes the crucible binary from a different working directory, it will fail. More critically, an attacker could create a malicious binary at `target/release/hydrodynamic-swarm` in an arbitrary directory and trick a developer into running crucible from there, leading to arbitrary code execution under the developer's privileges.
**Learning:** Hardcoded relative paths for executing binaries make tools fragile and susceptible to path hijacking attacks when the Current Working Directory (CWD) is manipulated.
**Prevention:** When a Rust build script or internal tooling needs to reference binaries within the same cargo workspace, always anchor the path to the workspace root using the `env!("CARGO_MANIFEST_DIR")` compile-time environment variable to construct an absolute path.
