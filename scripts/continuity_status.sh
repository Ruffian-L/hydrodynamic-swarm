#!/usr/bin/env bash
# One-screen continuity status (store + bridges + latest cards). No model run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== CONTINUITY STATUS ==="
if [[ -f data/splat_memory.tct.json ]]; then
  python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path("data/splat_memory.tct.json").read_text())
print(f"TCT n_records={d.get('n_records')} bridges={d.get('n_prefill_bridge')} dim={d.get('model_dim')} fp={hex(d.get('model_fp') or 0)}")
PY
else
  echo "no data/splat_memory.tct.json"
fi
echo
echo "=== BRIDGES (weight / temporal) ==="
python3 scripts/list_bridges.py
echo
echo "=== LATEST MULTI-BRIDGE CARD ==="
ls -t logs/continuity_multibridge_*/CONTINUITY_CARD.md 2>/dev/null | head -1 | while read -r f; do
  echo "file: $f"
  cat "$f"
done || echo "(none yet — run scripts/continuity_multibridge.sh)"
echo
echo "=== LATEST REVISIT CARD ==="
ls -t logs/continuity_revisit_*/CONTINUITY_CARD.md 2>/dev/null | head -1 | while read -r f; do
  echo "file: $f"
  head -30 "$f"
done || echo "(none)"
