## 2024-05-24 - Fix Command Hijacking
**Vulnerability:** Found `std::process::Command` executing a binary using a relative path (`"target/release/hydrodynamic-swarm"`) inside a development tool.
**Learning:** This is a command hijacking vulnerability if the utility is executed from a directory controlled by an attacker, allowing arbitrary local execution of a malicious file.
**Prevention:** Always anchor execution paths for project binaries using the compile-time `env!("CARGO_MANIFEST_DIR")` absolute path resolution to guarantee safe execution context.
