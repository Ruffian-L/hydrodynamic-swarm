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

## 2024-05-23 - Path Hijacking Vulnerability via Relative Binary Execution
**Vulnerability:** The `crucible` development tool previously used a relative path (`target/release/hydrodynamic-swarm`) to locate and execute the project's internal binary via `std::process::Command::new(binary)`.
**Learning:** If `crucible` is executed from an unexpected working directory or if the `PATH` logic within `Command::new` allows unexpected resolution, an attacker could plant a malicious executable at `target/release/hydrodynamic-swarm` relative to the current directory, leading to arbitrary code execution when the tool is run. This was similar to the previously fixed `cargo` path hijacking vulnerability.
**Prevention:** Always use `env!("CARGO_MANIFEST_DIR")` to construct absolute paths to project-internal binaries (e.g., `concat!(env!("CARGO_MANIFEST_DIR"), "/target/release/hydrodynamic-swarm")`) at compile time. This ensures path resolution is independent of the current working directory.
