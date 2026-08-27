#!/usr/bin/env bash
# Poll the hydro mouth. uutils `tail -f` dies when inotify is full.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILE="${1:-$ROOT/logs/smoke_convo_latest.txt}"
python3 -u - "$FILE" <<'PY'
import sys, time
from pathlib import Path
p = Path(sys.argv[1])
seen = 0
while True:
    if p.exists() or p.is_symlink():
        try:
            data = p.read_bytes()
        except OSError:
            time.sleep(0.2)
            continue
        if len(data) > seen:
            sys.stdout.buffer.write(data[seen:])
            sys.stdout.buffer.flush()
            seen = len(data)
        elif len(data) < seen:
            sys.stdout.buffer.write(b"\n----- file truncated / new stamp -----\n")
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            seen = len(data)
    time.sleep(0.2)
PY
