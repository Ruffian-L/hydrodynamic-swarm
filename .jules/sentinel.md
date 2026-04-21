## 2024-05-24 - Missing network client timeout configuration
**Vulnerability:** The reqwest::Client in grok_oracle.rs is initialized without a timeout configuration.
**Learning:** Network clients without timeouts can hang indefinitely if the API is unresponsive, causing resource exhaustion, DoS, and application hang.
**Prevention:** Always configure reasonable timeouts when creating network clients, like .timeout(Duration::from_secs(60)).
