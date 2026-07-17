#!/usr/bin/env python3
"""List prefill-bridges: fingerprint, gain (weight), age, prompt.

Temporal / weight columns are for continuity ops — not feelings claims.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    reg = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bridge_prompts.json")
    tct = Path(sys.argv[2] if len(sys.argv) > 2 else "data/splat_memory.tct.json")
    labels: dict[str, str] = {}
    if reg.exists():
        root = json.loads(reg.read_text(encoding="utf-8"))
        for k, v in (root.get("prompts") or {}).items():
            if isinstance(v, dict):
                labels[k] = str(v.get("prompt") or v.get("text") or "")
            else:
                labels[k] = str(v)

    records = []
    if tct.exists():
        side = json.loads(tct.read_text(encoding="utf-8"))
        if not labels and side.get("bridge_prompt_labels"):
            labels = {str(k): str(v) for k, v in side["bridge_prompt_labels"].items()}
        for r in side.get("records") or []:
            if not isinstance(r, dict):
                continue
            if not (r.get("is_prefill_bridge") or r.get("trigger_kind") == 5):
                continue
            fp = str(r.get("prompt_fp") or "")
            if not fp.startswith("0x") and fp.isdigit():
                fp = hex(int(fp))
            text = r.get("prompt_text") or labels.get(fp) or labels.get(fp.lower()) or ""
            records.append(
                {
                    "fp": fp or "?",
                    "gain": float(r.get("gain") or 0.0),
                    "sigma": float(r.get("sigma") or 0.0),
                    "created_at_ms": int(r.get("created_at_ms") or 0),
                    "center_l2": float(r.get("center_l2") or 0.0),
                    "prompt": str(text),
                }
            )

    if not records:
        # fallback: fps only from sidecar
        if tct.exists():
            side = json.loads(tct.read_text(encoding="utf-8"))
            for fp in side.get("bridge_prompt_fps") or []:
                records.append(
                    {
                        "fp": fp,
                        "gain": float("nan"),
                        "sigma": float("nan"),
                        "created_at_ms": 0,
                        "center_l2": float("nan"),
                        "prompt": labels.get(fp, "(unknown)"),
                    }
                )

    if not records:
        print("no bridges found (run a gen with prefill_bridge_scar first)")
        return

    # newest first (temporal ordering visible)
    records.sort(key=lambda r: r["created_at_ms"], reverse=True)
    now_ms = int(time.time() * 1000)
    print(
        f"{'fp':12}  {'gain':>7}  {'σ':>6}  {'age':>8}  {'|μ|':>8}  prompt"
    )
    print("-" * 96)
    for r in records:
        age = ""
        if r["created_at_ms"] > 0:
            # created_at_ms in store may be session-scale units; show raw if huge future
            dt = now_ms - r["created_at_ms"]
            if abs(dt) > 1000 * 86400 * 365 * 50:
                age = f"t={r['created_at_ms']}"
            else:
                sec = max(0, dt) / 1000.0
                if sec < 3600:
                    age = f"{sec/60:.0f}m"
                elif sec < 86400:
                    age = f"{sec/3600:.1f}h"
                else:
                    age = f"{sec/86400:.1f}d"
        gain_s = f"{r['gain']:.3f}" if r["gain"] == r["gain"] else "?"
        sig_s = f"{r['sigma']:.0f}" if r["sigma"] == r["sigma"] else "?"
        l2_s = f"{r['center_l2']:.1f}" if r["center_l2"] == r["center_l2"] else "?"
        prompt = (r["prompt"] or "(unknown)")[:48]
        print(f"{r['fp']:12}  {gain_s:>7}  {sig_s:>6}  {age:>8}  {l2_s:>8}  {prompt}")


if __name__ == "__main__":
    main()
