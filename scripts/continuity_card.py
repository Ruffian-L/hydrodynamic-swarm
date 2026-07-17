#!/usr/bin/env python3
"""One-line continuity card from hydro session jsonl + optional tct sidecar.

Warm heuristic (bridge revisit):
  nearest_min small OR pot_max high OR early |F_s| coupling.
Cold: far nearest, pot~0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict):
            rows.append(o)
    return rows


def fnum(x, default=float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def dig_forces(rows: list[dict]) -> dict:
    """Collect nearest / pot / F_s style fields from nested hydro jsonl."""
    nearest = []
    pots = []
    fs = []
    n_bridges = None
    for r in rows:
        # common shapes: step records, nested force windows
        def walk(o, path=""):
            nonlocal n_bridges
            if isinstance(o, dict):
                for k, v in o.items():
                    lk = k.lower()
                    if lk in (
                        "nearest_scar_dist",
                        "nearest_scar_l2",
                        "scar_nearest_dist",
                        "nearest_l2",
                        "mean_scar_dist",
                    ):
                        nearest.append(fnum(v))
                    if lk in (
                        "scar_potential",
                        "scar_potential_at_prefill",
                        "potential",
                        "query_potential",
                        "splat_potential",
                    ):
                        pots.append(fnum(v))
                    if lk in (
                        "f_s",
                        "fs",
                        "scar_force",
                        "force_scar",
                        "splat_force_norm",
                        "splat_force_mag",
                    ):
                        fs.append(abs(fnum(v, 0.0)))
                    if lk in ("n_prefill_bridges", "prefill_bridges") and isinstance(
                        v, (int, float)
                    ):
                        n_bridges = int(v)
                    if isinstance(v, (dict, list)):
                        walk(v, path + "." + k)
            elif isinstance(o, list):
                for i, x in enumerate(o[:200]):
                    walk(x, f"{path}[{i}]")

        walk(r)
    nearest_ok = [n for n in nearest if math.isfinite(n)]
    pots_ok = [p for p in pots if math.isfinite(p)]
    fs_ok = [x for x in fs if math.isfinite(x)]
    return {
        "nearest_min": min(nearest_ok) if nearest_ok else float("inf"),
        "nearest_n": len(nearest_ok),
        "pot_max": max(pots_ok) if pots_ok else 0.0,
        "pot_n": len(pots_ok),
        "fs_max": max(fs_ok) if fs_ok else 0.0,
        "fs_n": len(fs_ok),
        "n_bridges": n_bridges,
        "n_records": len(rows),
    }


def classify(d: dict) -> str:
    """Continuity status from prefill geometry (not raw F_s — trail scars always push).

    WARM: sitting near a scar center (bridge revisit) or high prefill potential.
    LUKE / COLD: farther first-visit basins. F_s alone does not warm the card.
    """
    nm = d["nearest_min"]
    pot = d["pot_max"]
    if d["nearest_n"] == 0 and d["pot_n"] == 0 and d["fs_n"] == 0:
        return "NO_METRIC"
    # pot = prefill scar potential (best multi-bridge separator).
    # nearest often ≈ offset*σ (0.35*90=31.5) when sitting on a soft-offset bridge.
    if pot > 0.15 or (math.isfinite(nm) and nm < 80.0 and pot > 0.05):
        return "WARM"
    if math.isfinite(nm) and nm < 80.0:
        return "WARM"  # on-offset-ring geometry even if pot soft
    if math.isfinite(nm) and nm < 200.0:
        return "NEAR"
    if math.isfinite(nm) and nm < 600.0:
        return "LUKE"
    return "COLD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="?", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tct-json", type=Path, default=Path("data/splat_memory.tct.json"))
    args = ap.parse_args()

    d = dig_forces(load_jsonl(args.jsonl)) if args.jsonl else {
        "nearest_min": float("inf"),
        "nearest_n": 0,
        "pot_max": 0.0,
        "pot_n": 0,
        "fs_max": 0.0,
        "fs_n": 0,
        "n_bridges": None,
        "n_records": 0,
    }
    status = classify(d)

    bridges = None
    fps = []
    bridge_gains: dict[str, float] = {}
    if args.tct_json and args.tct_json.exists():
        try:
            doc = json.loads(args.tct_json.read_text(encoding="utf-8"))
            bridges = doc.get("n_prefill_bridge")
            fps = doc.get("bridge_prompt_fps") or []
            for r in doc.get("records") or []:
                if not isinstance(r, dict):
                    continue
                if not (r.get("is_prefill_bridge") or r.get("trigger_kind") == 5):
                    continue
                fp = str(r.get("prompt_fp") or "")
                if fp:
                    bridge_gains[fp] = float(r.get("gain") or 0.0)
        except Exception:
            pass
    if d["n_bridges"] is None and bridges is not None:
        d["n_bridges"] = bridges

    gain_max = max(bridge_gains.values()) if bridge_gains else 0.0
    # weight summary: strongest bridge gain (earned mass proxy)
    gain_s = f"{gain_max:.3f}" if bridge_gains else "n/a"

    nm = d["nearest_min"]
    nm_s = f"{nm:.4g}" if math.isfinite(nm) else "inf"
    label = (args.label or (str(args.jsonl) if args.jsonl else "")).replace("\n", " ")[:80]
    card = (
        f"CONT  {status}  nearest_min={nm_s}  pot_max={d['pot_max']:.4g}  "
        f"gain_max={gain_s}  fs_max={d['fs_max']:.4g}  bridges={d['n_bridges']}  "
        f"samples[n/p/f]={d['nearest_n']}/{d['pot_n']}/{d['fs_n']}  "
        f"fps={fps}  {label}"
    )
    print(card)
    payload = {
        "status": status,
        "label": label,
        "nearest_min": nm if math.isfinite(nm) else None,
        "pot_max": d["pot_max"],
        "fs_max": d["fs_max"],
        "gain_max": gain_max if bridge_gains else None,
        "bridge_gains": bridge_gains,
        "n_prefill_bridges": d["n_bridges"],
        "bridge_prompt_fps": fps,
        "metric_counts": {
            "nearest": d["nearest_n"],
            "potential": d["pot_n"],
            "fs": d["fs_n"],
            "jsonl_records": d["n_records"],
        },
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if status != "NO_METRIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
