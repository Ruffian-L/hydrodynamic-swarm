#!/usr/bin/env bash
# Concise console / log scrapers for Shep · Echo · Lumina
# Floor: projects/hydrodynamic-swarm only
#
# Usage:
#   ./scripts/console_watch.sh              # menu / help
#   ./scripts/console_watch.sh live         # tail human console (live.txt)
#   ./scripts/console_watch.sh hits         # follow only important console hits
#   ./scripts/console_watch.sh forces       # scrape latest.jsonl force table (once)
#   ./scripts/console_watch.sh forces -f    # re-scrape every 2s
#   ./scripts/console_watch.sh g3           # pain / pleasure / budget hits
#   ./scripts/console_watch.sh snap         # one-screen status (no follow)
#   ./scripts/console_watch.sh coupling     # tail newest memory_coupling_*/ stdout
#   ./scripts/console_watch.sh weather      # point at latest.termsplat.jsonl
#   ./scripts/console_watch.sh proc         # is hydro running?
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGS="$ROOT/logs"
LIVE="$LOGS/live.txt"
JSONL="$LOGS/latest.jsonl"
TERMSPLAT="$LOGS/latest.termsplat.jsonl"

# console lines that actually matter (G2/G3 + continuity)
HIT_RE='PAIN BUDGET|PLEASURE ANSWER|ENDOCRINE|scar geometry|Prefill-bridge|BASIN LIVE|Saved .*splats|TCT-splat-lite|memory_loaded|scars_start|FORCE|force_off|ERROR|panic|BLOCKED|PASS|FAIL|\[PAIN|SPLATS|OPERATIONAL'

usage() {
  cat <<'EOF'
console_watch — tail / scrape hydro console without re-auditing G1

  live       tail -F logs/live.txt
  hits       follow only signal lines (pain, scars, endocrine, save, errors)
  forces     table from logs/latest.jsonl (step / Fs / Fg / scars)
  forces -f  refresh every 2s
  g3         pain/pleasure/budget scrapes (live + latest)
  snap       one-shot: tree, proc, vendor, latest paths, last hits
  coupling   follow newest logs/memory_coupling_*/*.stdout
  weather    echo path + line count of latest.termsplat.jsonl
  proc       pgrep hydrodynamic-swarm

Always from: /media/ruffianl/ghost_team/projects/hydrodynamic-swarm
EOF
}

need_live() {
  [[ -f "$LIVE" ]] || { echo "no $LIVE yet — start a run first"; return 1; }
}

cmd_live() {
  need_live
  echo "[tail] $LIVE  (Ctrl-C stop)"
  tail -n 40 -F "$LIVE"
}

cmd_hits() {
  need_live
  echo "[hits] $LIVE  pattern: pain|scars|endocrine|save|error"
  # show last matching, then follow
  grep -E "$HIT_RE" "$LIVE" 2>/dev/null | tail -30 || true
  echo "----- follow -----"
  tail -n 0 -F "$LIVE" | grep --line-buffered -E "$HIT_RE"
}

cmd_forces() {
  local follow=0
  [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]] && follow=1
  scrape() {
    if [[ ! -e "$JSONL" ]]; then
      echo "no $JSONL"
      return 0
    fi
    # resolve symlink for display
    local real
    real="$(readlink -f "$JSONL" 2>/dev/null || echo "$JSONL")"
    echo "=== forces  $(date -u +%H:%M:%S)Z  ← $real ==="
    python3 - "$JSONL" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = []
cfg = None
for line in p.open():
    try:
        d = json.loads(line)
    except Exception:
        continue
    et = d.get("entry_type")
    if et == "config":
        cfg = d.get("config") or d
    if et == "step" and isinstance(d.get("step"), dict):
        s = d["step"]
        rows.append(s)
if cfg:
    for k in ("nearest_scar_dist", "scar_potential_at_prefill", "scars_at_start",
              "splat_force_scale", "goal_force_scale"):
        if k in cfg:
            print(f"config.{k}={cfg[k]}")
print(f"{'step':>4}  {'Fs':>8}  {'Fg':>8}  {'Fa/goal':>8}  {'δ':>8}  scars  tok")
print("-" * 64)
for s in rows[:8]:
    print(f"{s.get('step',0):>4}  {s.get('splat_force_mag',0):>8.2f}  "
          f"{s.get('grad_force_mag',0):>8.2f}  {s.get('goal_force_mag',0):>8.2f}  "
          f"{s.get('steering_delta',0):>8.2f}  {s.get('scars_active','?'):>5}  "
          f"{(s.get('token_text') or '')[:12]!r}")
if len(rows) > 16:
    print("  ...")
for s in rows[-8:] if len(rows) > 8 else []:
    print(f"{s.get('step',0):>4}  {s.get('splat_force_mag',0):>8.2f}  "
          f"{s.get('grad_force_mag',0):>8.2f}  {s.get('goal_force_mag',0):>8.2f}  "
          f"{s.get('steering_delta',0):>8.2f}  {s.get('scars_active','?'):>5}  "
          f"{(s.get('token_text') or '')[:12]!r}")
print(f"n_steps={len(rows)}")
if rows:
    fs0 = rows[0].get("splat_force_mag", 0) or 0
    fsn = rows[-1].get("splat_force_mag", 0) or 0
    print(f"early_Fs={fs0:.3f}  late_Fs={fsn:.3f}")
PY
  }
  if [[ "$follow" -eq 1 ]]; then
    while true; do
      clear 2>/dev/null || true
      scrape
      sleep 2
    done
  else
    scrape
  fi
}

cmd_g3() {
  echo "=== G3 signal scrape ==="
  if [[ -f "$LIVE" ]]; then
    echo "-- live.txt (last 40 hits) --"
    grep -E 'PAIN BUDGET|PLEASURE ANSWER|pain|PLEASURE|decay|consolidat|\[PAIN' "$LIVE" \
      2>/dev/null | tail -40 || echo "(no pain/pleasure lines yet)"
  fi
  echo
  echo "-- counts in live --"
  if [[ -f "$LIVE" ]]; then
    printf 'PAIN BUDGET:     '; grep -c 'PAIN BUDGET' "$LIVE" 2>/dev/null || echo 0
    printf 'PLEASURE ANSWER: '; grep -c 'PLEASURE ANSWER' "$LIVE" 2>/dev/null || echo 0
    printf 'ENDOCRINE:       '; grep -c 'ENDOCRINE' "$LIVE" 2>/dev/null || echo 0
  fi
  echo
  echo "-- follow new hits (Ctrl-C) --"
  need_live
  tail -n 0 -F "$LIVE" | grep --line-buffered -E 'PAIN BUDGET|PLEASURE ANSWER|ENDOCRINE|ERROR|panic'
}

cmd_snap() {
  echo "=== SNAP $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "pwd:  $ROOT"
  echo "git:  $(git log -1 --oneline 2>/dev/null || echo '?')"
  echo "vendor: $(du -sh vendor 2>/dev/null | awk '{print $1}' || echo MISSING)  offline: $(test -f .cargo/config.toml && echo yes || echo no)"
  echo
  echo "--- processes ---"
  pgrep -af 'hydrodynamic-swarm|memory_coupling|g3_ablation' 2>/dev/null | grep -v 'pgrep\|console_watch' || echo "(none)"
  echo
  echo "--- latest artifacts ---"
  for f in live.txt latest.jsonl latest.termsplat.jsonl readable.txt; do
    if [[ -e "$LOGS/$f" ]]; then
      local_sz=$(du -h "$LOGS/$f" 2>/dev/null | awk '{print $1}')
      local_tgt=$(readlink -f "$LOGS/$f" 2>/dev/null || echo "$LOGS/$f")
      echo "  $f  $local_sz  → $local_tgt"
    else
      echo "  $f  MISSING"
    fi
  done
  echo
  echo "--- newest memory_coupling dir ---"
  ls -dt "$LOGS"/memory_coupling_* 2>/dev/null | head -3 | while read -r d; do
    echo "  $d"
    ls "$d" 2>/dev/null | tr '\n' ' '; echo
    test -f "$d/RECEIPT.md" && echo "    RECEIPT.md YES" || echo "    RECEIPT.md no (stub? $(test -f "$d/RECEIPT_STUB.md" && echo stub || echo none))"
  done
  echo
  echo "--- last console hits ---"
  if [[ -f "$LIVE" ]]; then
    grep -E "$HIT_RE" "$LIVE" 2>/dev/null | tail -12 || true
  fi
  echo
  echo "--- early/late Fs (latest.jsonl) ---"
  cmd_forces 2>/dev/null | tail -15 || true
}

cmd_coupling() {
  local d
  d="$(ls -dt "$LOGS"/memory_coupling_* 2>/dev/null | head -1 || true)"
  if [[ -z "$d" ]]; then
    echo "no memory_coupling_* dirs"
    exit 1
  fi
  echo "[coupling] $d"
  ls -la "$d" | head -20
  echo "----- tail stdout (A/B/C/D as present) -----"
  # follow all stdout files that exist; create empty waiters not needed
  shopt -s nullglob
  local files=("$d"/*.stdout)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "no *.stdout yet — waiting for dir updates..."
    tail -F "$d"/A.stdout 2>/dev/null || tail -n 5 -F "$LIVE"
  else
    tail -n 20 -F "${files[@]}"
  fi
}

cmd_weather() {
  if [[ -e "$TERMSPLAT" ]]; then
    real="$(readlink -f "$TERMSPLAT")"
    n=$(wc -l < "$real" 2>/dev/null || echo 0)
    echo "termsplat: $TERMSPLAT → $real  lines=$n"
    echo "paint: termsplat pipe $TERMSPLAT   # if termsplat bin on PATH"
  else
    echo "no $TERMSPLAT yet"
  fi
}

cmd_proc() {
  pgrep -af hydrodynamic-swarm 2>/dev/null | grep -v pgrep || echo "(no hydrodynamic-swarm)"
  pgrep -af 'memory_coupling|g3_ablation' 2>/dev/null | grep -v pgrep || true
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    ""|-h|--help|help) usage ;;
    live)       cmd_live "$@" ;;
    hits)       cmd_hits "$@" ;;
    forces)     cmd_forces "$@" ;;
    g3)         cmd_g3 "$@" ;;
    snap|status) cmd_snap "$@" ;;
    coupling)   cmd_coupling "$@" ;;
    weather)    cmd_weather "$@" ;;
    proc)       cmd_proc "$@" ;;
    *) echo "unknown: $cmd"; usage; exit 2 ;;
  esac
}

main "$@"
