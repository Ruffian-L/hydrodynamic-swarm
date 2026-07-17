#!/usr/bin/env python3
"""Extract early/late force windows from a hydro session JSONL (Echo / RECEIPT).

Real log shape:
  {"entry_type":"step","step":{"step":0,"splat_force_mag":...,"goal_force_mag":...,
   "grad_force_mag":...,"steering_delta":...}}

Usage:
  python3 scripts/extract_force_windows.py logs/foo.jsonl
  python3 scripts/extract_force_windows.py logs/memory_coupling_*/*.jsonl --tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_steps(path: Path) -> list[dict]:
    steps: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict):
                continue
            if o.get("entry_type") == "step" and isinstance(o.get("step"), dict):
                steps.append(o["step"])
            elif "splat_force_mag" in o or "steering_delta" in o:
                steps.append(o)
    steps.sort(key=lambda s: int(s.get("step", 0)))
    return steps


def load_config(path: Path) -> dict:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and o.get("entry_type") == "config":
                return o.get("config") or {}
    return {}


def extract(path: Path) -> dict:
    steps = load_steps(path)
    cfg = load_config(path)
    fs = [float(s["splat_force_mag"]) for s in steps if "splat_force_mag" in s]
    fa = [float(s["goal_force_mag"]) for s in steps if "goal_force_mag" in s]
    fg = [float(s["grad_force_mag"]) for s in steps if "grad_force_mag" in s]
    delta = [float(s["steering_delta"]) for s in steps if "steering_delta" in s]
    n = len(steps)

    def window(arr: list[float], a: int, b: int) -> float:
        if not arr:
            return 0.0
        sl = arr[a:b]
        return mean(sl) if sl else 0.0

    early_end = min(10, max(n, 1))
    late_start = max(0, n - 15)
    early_fs = window(fs, 0, early_end)
    nearest = cfg.get("nearest_scar_dist")
    pot = cfg.get("scar_potential_at_prefill")
    sigma = cfg.get("nearest_scar_sigma")
    try:
        nearest_f = float(nearest) if nearest is not None else None
    except (TypeError, ValueError):
        nearest_f = None
    try:
        pot_f = float(pot) if pot is not None else None
    except (TypeError, ValueError):
        pot_f = None
    try:
        sigma_f = float(sigma) if sigma is not None else None
    except (TypeError, ValueError):
        sigma_f = None
    # Basin live: pot high + start within ~0.5σ of a scar (covers soft off-center bridges).
    # Force at exact center can still be ~0 (gradient of Gaussian).
    near_ok = False
    if nearest_f is not None and nearest_f >= 0:
        if nearest_f < 1.0:
            near_ok = True
        elif sigma_f is not None and sigma_f > 1e-6 and (nearest_f / sigma_f) <= 0.5:
            near_ok = True
    basin_live = pot_f is not None and pot_f > 0.1 and near_ok
    couples_force = early_fs > 0.05
    return {
        "file": str(path),
        "n_steps": n,
        "scars_at_start": cfg.get("scars_at_start"),
        "n_prefill_bridges": cfg.get("n_prefill_bridges"),
        "memory_loaded": cfg.get("memory_loaded"),
        "nearest_L2": nearest_f,
        "scar_potential": pot_f,
        "force_ramp_tokens": cfg.get("force_ramp_tokens"),
        "early_Fs": round(early_fs, 4),
        "late_Fs": round(window(fs, late_start, n), 4),
        "step0_Fs": round(fs[0], 6) if fs else None,
        "early_Fa": round(window(fa, 0, early_end), 4),
        "late_Fa": round(window(fa, late_start, n), 4),
        "early_Fg": round(window(fg, 0, early_end), 4),
        "early_delta": round(window(delta, 0, early_end), 4),
        "late_delta": round(window(delta, late_start, n), 4),
        "max_Fs": round(max(fs), 4) if fs else 0.0,
        "has_Fs_series": bool(fs),
        "basin_live": basin_live,
        "memory_couples_hint": couples_force or basin_live,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--tsv", action="store_true")
    args = ap.parse_args()
    rows = []
    for p in args.paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        rows.append(extract(p))
    if not rows:
        sys.exit(1)
    if args.tsv:
        keys = [
            "file",
            "n_steps",
            "scars_at_start",
            "n_prefill_bridges",
            "nearest_L2",
            "scar_potential",
            "step0_Fs",
            "early_Fs",
            "late_Fs",
            "early_Fa",
            "max_Fs",
            "basin_live",
            "memory_couples_hint",
        ]
        print("\t".join(keys))
        for r in rows:
            print("\t".join(str(r.get(k, "")) for k in keys))
    else:
        for r in rows:
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
