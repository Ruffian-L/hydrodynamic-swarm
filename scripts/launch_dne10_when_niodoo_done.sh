#!/usr/bin/env bash
# Wait for Niodoo DNE-10 TABLE, then run Hydro matching 10. stdout: DONE/FAILED only.
set -euo pipefail
NROOT=/home/ruffianl/Hub/Projects/niodoo/niodoo-live/runs/2026-08-22_pathb_ifeval_dne10_niodoo
HROOT=/home/ruffianl/Hub/Projects/hydro/hydrodynamic-swarm-3surface/runs/2026-08-22_pathb_ifeval_dne10_hydro
LIVE=/home/ruffianl/Hub/Projects/niodoo/niodoo-live
HYDRO=/home/ruffianl/Hub/Projects/hydro/hydrodynamic-swarm-3surface
niodoo_alive() {
  pgrep -f 'run_pathb_ifeval_tqa.py --engine niodoo' >/dev/null \
    || pgrep -f '/niodoo --model-path' >/dev/null
}
niodoo_done() {
  [[ -f "$NROOT/TABLE.md" ]] && grep -q 'win(CI' "$NROOT/TABLE.md"
}
while ! niodoo_done; do
  if ! niodoo_alive; then
    if niodoo_done; then break; fi
    echo FAILED
    exit 1
  fi
  sleep 20
done
mkdir -p "$HROOT"
python3 -u "$HYDRO/scripts/follow_pathb_mouth.py" "$HROOT" >>"$HROOT/mouth.follow.log" 2>&1 &
cd "$LIVE"
python3 -u scripts/run_pathb_ifeval_tqa.py \
  --engine hydro \
  --all-arms \
  --tasks ifeval \
  --limit 10 \
  --root "$HROOT" \
  >>"$HROOT/campaign.log" 2>&1
if [[ -f "$HROOT/TABLE.md" ]] && grep -q 'win(CI' "$HROOT/TABLE.md"; then
  echo DONE
  exit 0
fi
echo FAILED
exit 1
