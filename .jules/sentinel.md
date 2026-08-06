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
## 2026-08-05 - Command Hijacking via Relative Path
**Vulnerability:** The `crucible` binary executes the project-internal binary using a relative path (`target/release/hydrodynamic-swarm`). If a user runs `crucible` from a different directory where a malicious `target/release/hydrodynamic-swarm` exists, the malicious binary will be executed instead of the intended one.
**Learning:** Relying on relative paths for executing binaries makes the application vulnerable to command hijacking, as the resolved path depends entirely on the unpredictable current working directory.
**Prevention:** Construct absolute paths to project-internal binaries at compile time using `env!("CARGO_MANIFEST_DIR")` (e.g., `concat!(env!("CARGO_MANIFEST_DIR"), "/target/release/binary")`) to ensure path resolution is safe and independent of the execution context.
## 2024-05-18 - Lock Poisoning DoS Vulnerability
**Vulnerability:** Found multiple instances where thread panics could cause permanent application denial of service (DoS). Specifically, calling `.unwrap()` on `RwLock::read()`, `RwLock::write()`, and `Mutex::lock()` acquisitions throughout `src/concourse/cache.rs`, `src/concourse/embed/gemma.rs`, and `src/concourse/function/instruct_gemma.rs`.
**Learning:** If a thread panics while holding one of these locks, the lock becomes poisoned. Subsequent threads calling `.unwrap()` on the poisoned lock will also panic, cascading the failure and permanently bringing down the application state.
**Prevention:** Instead of using `.unwrap()` on lock acquisitions, safely recover the lock guard by handling the poison error with `.unwrap_or_else(|e| e.into_inner())` or matching on the result. This ignores the poisoned state and allows the application to recover and continue operating, which is crucial for high-availability systems.
