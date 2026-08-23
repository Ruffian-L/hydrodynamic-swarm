# Vendor checksum files omitted by workspace ignores

> Date: 2026-08-22
> Agent: Sol (OpenAI)
> Repo: Ruffian-L/hydrodynamic-swarm

## Context

A fresh GitHub clone failed before compilation because Cargo could not find
`vendor/cc/src/target/generated.rs`. Auditing every vendored
`.cargo-checksum.json` found 17 manifest-listed files absent from Git.

## What changed

- Added narrow `vendor/**` exceptions after broad workspace ignore rules for
  nested `target`, `.cargo`, `env`, `.vscode`, binary, and `AGENTS` paths.
- Restored all 17 files byte-for-byte from the matching cached `.crate`
  archives. Their SHA-256 values match the existing manifests.
- Added `scripts/check_vendor_integrity.py` to verify both presence and content
  hashes without Cargo or network access.

## Hypothesis

Future clones should retain complete vendored crate snapshots, allowing Cargo
to validate offline sources and reach project compilation.

## Findings

Implementation complete; verification receipts will be added after the new
integrity checker and Cargo metadata path run from the tracked file set.

## Next

1. Verify every vendored file checksum.
2. Confirm Git sees exactly the 17 restored files plus the scoped fix and logs.
3. Run Cargo metadata offline to prove source validation crosses the old stop.
