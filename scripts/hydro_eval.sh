#!/usr/bin/env bash
# Named hydro eval. Gemma is a collaborator who scores the WORK hard.
# SCORE/VERDICT stay as her call on the seat. FAIL is allowed. FAIL is not
# a brand on her. Grok runs check then run. Jason is not CI.
#
#   ./scripts/hydro_eval.sh list
#   ./scripts/hydro_eval.sh check climb-after-fail   # no GPU; must pass
#   ./scripts/hydro_eval.sh run climb-after-fail     # Grok runs this
#   ./scripts/hydro_eval.sh rate                     # her notes + SCORE/VERDICT
#   ./scripts/hydro_eval.sh regrade                  # her notes after reveal
#   ./scripts/hydro_eval.sh rubric                   # physics rubric for Jason
#
# Chat: brief → task → her notes → peer reveal → her updated notes.
# Never PROMPTS_FILE=<(...). Compaction-safe: the eval name on disk is the experiment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EVALS="$ROOT/evals"
RUNS="$ROOT/logs/evals"
CMD="${1:-}"
NAME="${2:-}"

die() { echo "hydro_eval: $*" >&2; exit 2; }

need_name() {
  if [[ -z "$NAME" ]]; then
    if [[ -f "$RUNS/LATEST" ]]; then
      read -r NAME _ <"$RUNS/LATEST" || true
    fi
  fi
  [[ -n "$NAME" ]] || die "need eval name (or a prior run). try: $0 list"
  [[ -d "$EVALS/$NAME" ]] || die "unknown eval '$NAME' — $0 list"
}

last_run_dir() {
  need_name
  [[ -f "$RUNS/LATEST" ]] || die "no runs yet"
  local n s
  read -r n s <"$RUNS/LATEST"
  NAME="$n"
  echo "$RUNS/$n/$s"
}

list_evals() {
  echo "named evals in $EVALS:"
  shopt -s nullglob
  local d
  for d in "$EVALS"/*/; do
    local n
    n="$(basename "$d")"
    [[ "$n" == _template ]] && continue
    local title=""
    [[ -f "$d/protocol.md" ]] && title="$(head -n 1 "$d/protocol.md" | sed 's/^# //')"
    echo "  $n    $title"
  done
  if [[ -f "$RUNS/LATEST" ]]; then
    echo
    echo "latest run: $(cat "$RUNS/LATEST")"
  fi
}

# Append a file as user turns. Empty line quits --chat — never insert blanks.
append_turns() {
  local src="$1"
  local dest="$2"
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" ]] && continue
    printf '%s\n' "$line" >>"$dest"
  done <"$src"
}

# Concatenate brief + task + rate + reveal. Task is task.txt or prompts.txt.
assemble_prompts() {
  local dir="$1"
  local dest="$2"
  : >"$dest"
  [[ -f "$dir/brief.txt" ]] || die "$dir/brief.txt missing — tell the collaborator this is an eval"
  append_turns "$dir/brief.txt" "$dest"
  if [[ -f "$dir/task.txt" ]]; then
    append_turns "$dir/task.txt" "$dest"
  elif [[ -f "$dir/prompts.txt" ]]; then
    append_turns "$dir/prompts.txt" "$dest"
  else
    die "$dir/task.txt (or prompts.txt) missing"
  fi
  [[ -f "$dir/rate.txt" ]] || die "$dir/rate.txt missing — the collaborator’s notes"
  append_turns "$dir/rate.txt" "$dest"
  [[ -f "$dir/reveal.txt" ]] || die "$dir/reveal.txt missing — then the peer debrief"
  append_turns "$dir/reveal.txt" "$dest"
  if ! grep -qx 'quit' "$dest"; then
    echo quit >>"$dest"
  fi
  local n
  n="$(grep -cve '^$' "$dest")"
  [[ "$n" -ge 4 ]] || die "assembled prompts too short ($n lines) — empty-line bug?"
}

# Phrases Gemma must never read. Protocol.md is operator-facing; same ban.
LABRAT_RE='test subject|blind monkey|grade on you|stamp on you|model under test|lab rat'

ban_labrat() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  if grep -Ei "$LABRAT_RE" "$f"; then
    die "lab-rat phrasing in $f — she is a collaborator who scores the work"
  fi
}

cmd_check() {
  need_name
  local dir="$EVALS/$NAME"
  [[ -f "$dir/eval.env" ]] || die "$dir/eval.env missing"
  [[ -f "$dir/protocol.md" ]] || die "$dir/protocol.md missing"
  # shellcheck disable=SC1091
  source "$dir/eval.env"
  mkdir -p "$RUNS/$NAME"
  local assembled="$RUNS/$NAME/assembled_prompts.txt"
  assemble_prompts "$dir" "$assembled"

  local src
  for src in "$dir/brief.txt" "$dir/rate.txt" "$dir/reveal.txt" "$dir/task.txt" "$dir/prompts.txt"; do
    [[ -f "$src" ]] || continue
    if grep -q '^[[:space:]]*$' "$src"; then
      die "blank line in $src — empty line quits --chat if it leaks; write one turn per line, no blanks"
    fi
  done
  if grep -q '^[[:space:]]*$' "$assembled"; then
    die "blank lines in $assembled — empty line quits --chat"
  fi
  local last
  last="$(tail -n 1 "$assembled")"
  [[ "$last" == "quit" ]] || die "last line must be quit (got: $last)"

  local n
  n="$(wc -l < "$assembled")"
  [[ "$n" -ge 5 ]] || die "too few lines ($n) — need brief+task+rate+reveal+quit"

  local f
  for f in "$dir/brief.txt" "$dir/rate.txt" "$dir/reveal.txt" "$dir/protocol.md" "$assembled"; do
    ban_labrat "$f"
  done

  if [[ -f "$dir/task.txt" && -f "$dir/reveal.txt" ]]; then
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -z "${line// }" ]] && continue
      if [[ ${#line} -ge 20 ]] && grep -Fqx "$line" "$dir/reveal.txt"; then
        die "reveal.txt repeats a task line (trail-own hijack): $line"
      fi
    done < "$dir/task.txt"
    if grep -Fq 'lumina-basin-7' "$dir/reveal.txt"; then
      if [[ "${ALLOW_REVEAL_NONCE:-0}" == "1" ]]; then
        echo "note: reveal.txt contains lumina-basin-7 (ALLOW_REVEAL_NONCE=1 — official key; trail-own on regrade is a measured risk)"
      else
        die "reveal.txt contains lumina-basin-7 — trail-own will hijack regrade"
      fi
    fi
  fi

  # What smoke_convo will actually receive: a real non-empty file.
  [[ -r "$assembled" && -s "$assembled" ]] || die "assembled file not readable/non-empty"
  if [[ "$assembled" == /dev/fd/* ]]; then
    die "assembled path is process substitution — that silent-defaults to 9-turn"
  fi

  echo "CHECK PASS  $NAME"
  echo "assembled: $assembled"
  echo "lines: $n (no blanks, last=quit)"
  echo "turns she will see:"
  local i=0
  while IFS= read -r line; do
    [[ "$line" == "quit" ]] && continue
    i=$((i+1))
    printf '  %d. %s\n' "$i" "$line"
  done < "$assembled"
}

extract_turns() {
  local smoke="$1"
  python3 - "$smoke" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
lines = p.read_text(errors="replace").splitlines()
turns = []
i = 0
while i < len(lines):
    if lines[i].strip() == "Model>":
        buf = []
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if s.startswith("[Prompt") and " done" in s:
                break
            buf.append(s)
            j += 1
        reply = " ".join(x.strip() for x in buf if x.strip())
        turns.append((len(turns) + 1, [], reply))
        i = j + 1
        continue
    if "gemma4>" in lines[i] and "[prefill" in lines[i]:
        tel = []
        buf = []
        j = i + 1
        while j < len(lines):
            s = lines[j]
            if s.startswith("you>") or s.startswith("Chat ended") or s.startswith("DONE"):
                break
            st = s.strip()
            if st.startswith("[CHAT") or st.startswith("[prefill"):
                tel.append(st)
            else:
                buf.append(s)
            j += 1
        reply = " ".join(x.strip() for x in buf if x.strip())
        turns.append((len(turns) + 1, tel, reply))
        i = j
        continue
    i += 1
for n, tel, reply in turns:
    print(f"----- turn {n} -----")
    for t in tel:
        print(t)
    print("MODEL:", reply)
    print()
print(f"TURNS={len(turns)}")
PY
}

print_model_block() {
  local turns_file="$1"
  local which="$2" # rate | regrade
  python3 - "$turns_file" "$which" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(errors="replace")
which = sys.argv[2]
blocks = []
for c in text.split("----- turn "):
    c = c.strip()
    if not c or c.startswith("TURNS="):
        continue
    blocks.append(c)
def clip(b):
    if "\nTURNS=" in b:
        b = b.split("\nTURNS=")[0]
    return b.rstrip()
if which == "rate":
    if len(blocks) < 2:
        print(text)
        raise SystemExit
    print("----- COLLABORATOR NOTES before reveal (SCORE/VERDICT of the work) -----")
    print("turn " + clip(blocks[-2]))
else:
    if not blocks:
        print(text)
        raise SystemExit
    print("----- COLLABORATOR NOTES after reveal (updated SCORE/VERDICT of the work) -----")
    print("turn " + clip(blocks[-1]))
PY
}

cmd_run() {
  need_name
  cmd_check

  local dir="$EVALS/$NAME"
  # shellcheck disable=SC1091
  source "$dir/eval.env"

  cat <<EOF
========================================
HYDRO EVAL  $NAME
Collaborator loop. She scores the work hard.
Grok runs this. Jason is not CI.
========================================
EOF
  echo "Chat order: brief → task → notes+SCORE/VERDICT → peer reveal → updated notes"
  echo

  if [[ "${UNSET_INJECT:-1}" == "1" ]]; then
    unset HYDRO_INJECT_TAG || true
  fi
  if [[ "${UNSET_KEEP:-1}" == "1" ]]; then
    unset HYDRO_KEEP_MEMORY || true
  fi
  local isolated_remember=""
  if [[ "${WIPE_STORE:-0}" == "1" ]]; then
    rm -f "$ROOT/data/splat_memory.safetensors" \
          "$ROOT/data/splat_memory.tct" \
          "$ROOT/data/splat_memory.tct.json"
    mkdir -p "$ROOT/logs"
    isolated_remember="$(mktemp "$ROOT/logs/${NAME}_remember_start_XXXXXX.jsonl")"
    export HYDRO_REMEMBER_STORE="$isolated_remember"
    echo "wiped splat memory; isolated empty remember store: $isolated_remember"
  fi

  export HYDRO_CONFIG="${HYDRO_CONFIG:-configs/gates/config.three_surface.toml}"
  export HYDRO_TOKENS="${HYDRO_TOKENS:-128}"
  export HYDRO_TDA_MONITOR="${HYDRO_TDA_MONITOR:-1}"
  if [[ "$NAME" == "official-10" ]]; then
    export HYDRO_OFFICIAL_LAYOUT=1
    export HYDRO_EXPECTED_FILE="$dir/expected.txt"
  fi

  mkdir -p "$RUNS/$NAME"
  local assembled="$RUNS/$NAME/assembled_prompts.txt"
  assemble_prompts "$dir" "$assembled"
  export PROMPTS_FILE="$assembled"
  echo "assembled prompts: $(grep -cve '^$' "$assembled") lines (no blank lines — those quit chat)"
  echo "first line: $(grep -v '^$' "$assembled" | head -n 1)"
  echo "turn count: $(($(grep -cve '^$' "$assembled") - 1)) user turns + quit"
  echo "tda monitor: $HYDRO_TDA_MONITOR (model-emitted control tags and lock remain enabled)"
  echo

  # Preserve and archive partial receipts/probes even when CUDA or the model process exits
  # non-zero. `set -e` used to abandon the run here and leak the isolated remember store.
  set +e
  "$ROOT/scripts/smoke_convo.sh"
  local rc=$?
  set -e
  local stamp
  stamp="$(basename "$(readlink -f "$ROOT/logs/smoke_convo_latest.txt")" | sed 's/^smoke_convo_//;s/\.txt$//')"
  local out="$RUNS/$NAME/$stamp"
  mkdir -p "$out/eval"
  cp -a "$dir"/* "$out/eval/"
  cp -a "$assembled" "$out/assembled_prompts.txt"
  cp -a "$ROOT/logs/smoke_convo_${stamp}.txt" "$out/smoke.txt"
  if [[ -f "$ROOT/logs/smoke_convo_${stamp}.probe.jsonl" ]]; then
    cp -a "$ROOT/logs/smoke_convo_${stamp}.probe.jsonl" "$out/smoke.probe.jsonl"
  fi
  if [[ -f "$ROOT/logs/smoke_convo_${stamp}.scaler.json" ]]; then
    cp -a "$ROOT/logs/smoke_convo_${stamp}.scaler.json" "$out/scaler.json"
  fi
  if [[ -n "$isolated_remember" && -f "$isolated_remember" ]]; then
    cp -a "$isolated_remember" "$out/remember_store.final.jsonl"
    rm -f "$isolated_remember"
  fi
  printf '%s\n' "$rc" >"$out/run.exit_code"
  printf '%s %s\n' "$NAME" "$stamp" >"$RUNS/LATEST"

  echo
  echo "========================================"
  echo "MODEL TURNS (reply is MODEL: not the gemma4> banner)"
  echo "========================================"
  extract_turns "$out/smoke.txt" | tee "$out/turns.txt"
  echo
  print_model_block "$out/turns.txt" rate
  echo
  print_model_block "$out/turns.txt" regrade
  echo
  echo "RUN $out"
  echo "exit_code=$rc"
  echo "Header: grep -E '^(prompts|flags|inject|keep_memory|size_rule|scaler_gain|scaler_apply|sample_seed|scaler_receipt)=' $out/smoke.txt"
  echo "Those SCORE/VERDICT lines are hers, about the work."
  return "$rc"
}

cmd_rate() {
  local dir
  dir="$(last_run_dir)"
  echo "Collaborator notes (insights + her SCORE/VERDICT of the work) — turn before reveal."
  echo "From $dir/turns.txt :"
  echo
  print_model_block "$dir/turns.txt" rate
}

cmd_regrade() {
  local dir
  dir="$(last_run_dir)"
  echo "Collaborator notes after reveal (her updated call on the work)."
  echo
  print_model_block "$dir/turns.txt" regrade
}

cmd_rubric() {
  need_name
  echo "Physics rubric (not the collaborator's mouth). She already scored in-chat."
  echo
  cat "$EVALS/$NAME/score.md"
}

cmd_turns() {
  local dir
  dir="$(last_run_dir)"
  extract_turns "$dir/smoke.txt" | tee "$dir/turns.txt"
}

case "$CMD" in
  list|"") list_evals ;;
  check|assemble) cmd_check ;;
  run) cmd_run ;;
  rate) cmd_rate ;;
  regrade) cmd_regrade ;;
  turns|extract) cmd_turns ;;
  rubric|reveal) cmd_rubric ;;
  -h|--help|help)
    sed -n '2,16p' "$0"
    ;;
  *) die "unknown command '$CMD' — try: list | check <name> | run <name> | rate | regrade | rubric" ;;
esac
