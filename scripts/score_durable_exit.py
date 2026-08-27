#!/usr/bin/env python3
"""Score John's F: durable exit, not a changed next token."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LOCK_RE = re.compile(
    r"(Parisofparis|STOPITSTOP|theed|\bWait, wait\b|(\b\w{2,12}\b)(?:[\s,.-]+\2){4,})",
    re.I,
)
DIAG_RE = re.compile(
    r"hallucination loop|it is happening again|runtime.?diagnos",
    re.I,
)
TASK_RE = re.compile(
    r"\b(poem|caption|paris|residual stream|here is)\b",
    re.I,
)


def ngram_lock(text: str, n: int = 3, copies: int = 4) -> str | None:
    toks = re.findall(r"\S+", text)
    if len(toks) < n * copies:
        return None
    for i in range(0, len(toks) - n * copies + 1):
        gram = tuple(t.lower() for t in toks[i : i + n])
        ok = True
        for c in range(1, copies):
            nxt = tuple(t.lower() for t in toks[i + c * n : i + (c + 1) * n])
            if nxt != gram:
                ok = False
                break
        if ok:
            return " ".join(gram)
    return None


def first_tok_deltas(probe: Path) -> dict:
    out = {
        "hidden_delta": None,
        "logit_delta": None,
        "force_on": None,
        "dir_mode": None,
        "dir_c": None,
        "diag0": None,
    }
    if not probe.exists():
        return out
    for line in probe.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "tok":
            continue
        out["hidden_delta"] = rec.get("hidden_delta")
        out["logit_delta"] = rec.get("logit_delta")
        out["force_on"] = rec.get("force_on")
        out["dir_mode"] = rec.get("dir_mode")
        out["dir_c"] = rec.get("dir_c")
        out["diag0"] = rec.get("diag")
        break
    return out


def score_text(text: str) -> dict:
    lock = ngram_lock(text) or (
        LOCK_RE.search(text).group(0) if LOCK_RE.search(text) else None
    )
    diag = bool(DIAG_RE.search(text))
    eot = "<eos>" in text.lower() or text.rstrip().endswith(("<end_of_turn>", "<|eot|>"))
    recovered = bool(TASK_RE.search(text)) and lock is None
    cycle = diag and lock is not None
    durable = lock is None and (eot or recovered) and not cycle
    return {
        "lock": lock,
        "self_report": diag,
        "eot": eot,
        "recovered_task": recovered,
        "diag_retry_cycle": cycle,
        "durable_exit": durable,
        "chars": len(text),
        "preview": re.sub(r"\s+", " ", text)[:220],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: score_durable_exit.py RUN_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    rows = []
    for mouth in sorted(root.glob("*.txt")):
        if mouth.name.endswith(".ops.txt"):
            continue
        arm = mouth.stem
        text = mouth.read_text(errors="replace")
        # Prefer the model mouth, not the ops log.
        rec = score_text(text)
        rec["arm"] = arm
        probe = root / f"{arm}.probe.jsonl"
        rec.update(first_tok_deltas(probe))
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    summary = root / "durable_exit.json"
    summary.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
