## 2024-05-24 - Path Traversal in Log Filename
**Vulnerability:** The CLI tool `crucible.rs` takes a user-provided `tokens` argument and interpolates it directly into a file path (`logs/crucible_{}t.txt`) which is then passed to `File::create`. This allows an attacker to perform path traversal attacks using `../` to create or overwrite arbitrary files on the filesystem.
**Learning:** Even internal or development-focused utilities need input validation when user input is used in file system operations. Relying on the assumption that an argument meant to represent a number will only contain numbers is unsafe.
**Prevention:** Always explicitly validate and sanitize input before using it in file paths. For numeric arguments, enforcing strict checks like `!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())` ensures the value cannot contain path separators or traversal sequences.
## 2024-05-24 - SystemTime Clock Drift Panic
**Vulnerability:** SystemTime::now().duration_since(UNIX_EPOCH).unwrap() will panic and crash the application if the system clock drifts or is misconfigured to a time before January 1, 1970. This creates a reliability and potential Denial of Service (DoS) issue.
**Learning:** Never assume the system clock is perfectly synced or monotonically increasing relative to the Unix Epoch when calculating timestamps, especially in logging or utility code that runs frequently.
**Prevention:** Use `.duration_since(UNIX_EPOCH).unwrap_or_default()` instead of `.unwrap()` to gracefully handle `SystemTimeError` by returning a zero duration, preventing application crashes.
## 2024-05-24 - Missing Timeout on External API Call
**Vulnerability:** The `GrokOracle` in `src/grok_oracle.rs` uses `reqwest::Client` without configuring a timeout. If the `https://api.x.ai/v1/chat/completions` endpoint hangs or becomes unresponsive, the application using `GrokOracle` will also hang indefinitely, potentially leading to resource exhaustion (Denial of Service).
**Learning:** Always explicitly configure a reasonable timeout when creating HTTP clients to external services to ensure the application can recover or fail gracefully when the service is unavailable or slow.
**Prevention:** Configure the `reqwest::Client` builder with a timeout: `Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap()`.
