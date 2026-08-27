#!/usr/bin/env python3
"""Follow Path B IFEval jsonl mouths into mouth.txt. Idempotent."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
MOUTH = ROOT / "mouth.txt"
ARMS = ("tags", "vanilla")


def recs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for arm in ARMS:
        p = ROOT / arm / f"ifeval.{arm}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append((arm, rec))
    return out


def render(rows: list[tuple[str, dict]]) -> str:
    parts = ["mouth follow started\n"]
    for arm, rec in rows:
        gen = rec.get("generation") or rec.get("text") or ""
        tags = rec.get("tags") or []
        iid = rec.get("id")
        elapsed = rec.get("elapsed_s")
        extra = f" {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        parts.append("=" * 72)
        parts.append(f"arm={arm} id={iid} tags={tags} chars={len(gen)}{extra}")
        parts.append(gen.rstrip())
        parts.append("")
    return "\n".join(parts) + "\n"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    last_n = -1
    while True:
        rows = recs()
        if len(rows) != last_n:
            MOUTH.write_text(render(rows), encoding="utf-8")
            last_n = len(rows)
        if (ROOT / "TABLE.md").exists() and last_n >= 20:
            MOUTH.write_text(render(recs()), encoding="utf-8")
            return 0
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
