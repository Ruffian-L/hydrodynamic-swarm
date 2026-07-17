#!/usr/bin/env python3
"""No-GPU continuity tooling selftest (public CI / Tier 1 friendly)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(dir: Path) -> Path:
    """Minimal jsonl + tct sidecar shapes used by continuity_card."""
    jl = dir / "session.jsonl"
    # config-shaped record (as hydro logs)
    rec = {
        "config": {
            "nearest_scar_dist": 31.5,
            "nearest_scar_sigma": 90.0,
            "scar_potential_at_prefill": 0.67,
            "n_prefill_bridges": 2,
            "scars_at_start": 12,
            "mean_scar_dist": 180.0,
        },
        "step": {"splat_force_mag": 1.2},
    }
    jl.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    tct = dir / "splat_memory.tct.json"
    tct.write_text(
        json.dumps(
            {
                "n_records": 3,
                "n_prefill_bridge": 2,
                "model_dim": 2560,
                "model_fp": 0,
                "bridge_prompt_fps": ["0x8b262d40", "0x18934cbe"],
                "records": [
                    {
                        "is_prefill_bridge": True,
                        "prompt_fp": "0x8b262d40",
                        "gain": 0.75,
                        "sigma": 90.0,
                        "prompt_text": "Explain the Physics of Friendship in one short paragraph.",
                        "created_at_ms": 1,
                        "center_l2": 190.0,
                    },
                    {
                        "is_prefill_bridge": True,
                        "prompt_fp": "0x18934cbe",
                        "gain": 0.3,
                        "sigma": 90.0,
                        "prompt_text": "Write three short tips for debugging a CUDA kernel.",
                        "created_at_ms": 2,
                        "center_l2": 185.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return jl


def main() -> int:
    card_py = ROOT / "scripts" / "continuity_card.py"
    list_py = ROOT / "scripts" / "list_bridges.py"
    assert card_py.exists(), card_py
    assert list_py.exists(), list_py

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        jl = write_fixture(d)
        tct = d / "splat_memory.tct.json"
        out = subprocess.check_output(
            [sys.executable, str(card_py), str(jl), "--tct-json", str(tct), "--label", "selftest"],
            text=True,
        ).strip()
        print(out)
        if "CONT  WARM" not in out:
            print("FAIL: expected WARM for pot=0.67 nearest=31.5", file=sys.stderr)
            return 1
        if "gain_max=0.750" not in out and "gain_max=0.75" not in out:
            print("FAIL: expected gain_max from sidecar", file=sys.stderr)
            return 1

        # cold-ish: far nearest, low pot
        cold = {
            "config": {
                "nearest_scar_dist": 900.0,
                "scar_potential_at_prefill": 0.01,
                "n_prefill_bridges": 0,
            }
        }
        jl2 = d / "cold.jsonl"
        jl2.write_text(json.dumps(cold) + "\n", encoding="utf-8")
        out2 = subprocess.check_output(
            [sys.executable, str(card_py), str(jl2), "--tct-json", str(tct), "--label", "cold"],
            text=True,
        ).strip()
        print(out2)
        if "COLD" not in out2 and "LUKE" not in out2:
            print("FAIL: expected COLD/LUKE for far basin", file=sys.stderr)
            return 1

        # list_bridges against fixture registry-less (records only)
        # script defaults to data/ paths — pass tct path as argv2
        reg = d / "bridge_prompts.json"
        reg.write_text(json.dumps({"prompts": {}}), encoding="utf-8")
        out3 = subprocess.check_output(
            [sys.executable, str(list_py), str(reg), str(tct)],
            text=True,
        )
        print(out3)
        if "0x8b262d40" not in out3 or "0.750" not in out3:
            print("FAIL: list_bridges missing friendship bridge", file=sys.stderr)
            return 1

    print("continuity_selftest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
