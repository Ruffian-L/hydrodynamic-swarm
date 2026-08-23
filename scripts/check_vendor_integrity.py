#!/usr/bin/env python3
"""Verify that every vendored Cargo checksum entry exists and matches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    checked = 0

    for manifest in sorted(VENDOR.rglob(".cargo-checksum.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        crate_dir = manifest.parent
        for relative, expected in sorted(data.get("files", {}).items()):
            checked += 1
            source = crate_dir / relative
            label = source.relative_to(ROOT)
            if not source.is_file():
                failures.append(f"missing: {label}")
                continue
            actual = sha256(source)
            if actual != expected:
                failures.append(
                    f"checksum mismatch: {label} (expected {expected}, got {actual})"
                )

    if failures:
        print("Vendor integrity check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Vendor integrity check passed: {checked} files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
