#!/usr/bin/env python3
"""
Suggest hydrodynamic-swarm physics knobs from model size + type.

Source: Documents/Algo_WIPjuly.md (golden 3B √-law) + docs/MODEL_SIZE_PHYSICS_SCALING.md

  python3 scripts/scale_physics_for_model.py --params 4 --type instruct
  python3 scripts/scale_physics_for_model.py --params 1 --type standard
  python3 scripts/scale_physics_for_model.py --params 27 --type instruct --toml
  python3 scripts/scale_physics_for_model.py --params 4 --type instruct --algo-only
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


TYPE_MULT = {
    "standard": 1.0,
    "instruct": 0.9,
    "chat": 1.1,
    "thinking": 0.4,  # CoT / house-of-cards — whisper force
    "coding": 0.27,  # syntax wall — minimal jiggle
}

# Golden mid-zone swarm knobs ≈ 3B *standard* (Algo σ=0.15, θ=2.0 spirit).
# Map force_intensity = √(params/3) × type_mult onto these.
REF_3B_STANDARD = {
    "force_cap": 3.0,
    "splat_force_max": 28.0,
    "field_wake_max": 25.0,
    "goal_force_max": 40.0,
    "goal_force_scale": 0.12,
    "field_wake_scale": 0.18,
}

# Hard ceilings (learning-lane B on ~27B instruct — do not exceed on bigger models).
CEILING = {
    "force_cap": 3.5,
    "splat_force_max": 28.0,
    "field_wake_max": 25.0,
    "goal_force_max": 40.0,
    "goal_force_scale": 0.15,
    "field_wake_scale": 0.22,
}

# Soft floors so tiny models still get *some* steering (not dead zone).
FLOOR = {
    "force_cap": 1.2,
    "splat_force_max": 10.0,
    "field_wake_max": 8.0,
    "goal_force_max": 15.0,
    "goal_force_scale": 0.05,
    "field_wake_scale": 0.08,
}


@dataclass
class AlgoProcess:
    """Original Algo_WIP process intensities (σ/θ/β/repulsion)."""

    sigma: float
    theta: float
    beta: float
    loop_repulsion: float
    scale: float
    intensity: float
    type_mult: float


def algo_process(params_b: float, model_type: str) -> AlgoProcess:
    """
    Algo_WIPjuly core:
      scale = sqrt(params_B / 3.0)
      force intensity ∝ scale × type_mult
      σ,θ,repulsion get type_mult; β gets scale only
      hard clamps = traversable stability zone
    """
    golden_params = 3.0
    scale = math.sqrt(params_b / golden_params)
    mult = TYPE_MULT.get(model_type, 1.0)
    intensity = scale * mult

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    return AlgoProcess(
        sigma=round(clamp(0.15 * intensity, 0.04, 0.20), 3),
        theta=round(clamp(2.0 * intensity, 0.5, 3.0), 2),
        beta=round(clamp(100.0 * scale, 40.0, 150.0), 1),
        loop_repulsion=round(clamp(2.0 * intensity, 0.3, 3.0), 2),
        scale=round(scale, 3),
        intensity=round(intensity, 3),
        type_mult=mult,
    )


def swarm_knobs(params_b: float, model_type: str) -> dict:
    """
    Map Algo force_intensity onto hydrodynamic-swarm config.toml fields.

    Primary anchor: **3B standard golden mid-zone** (not 27B).
      knob ≈ REF_3B_STANDARD[k] * intensity
    Then clamp to FLOOR…CEILING.

    Small models (intensity < 1): softer caps — the 1B gibberish path.
    Large models: hit CEILING (same spirit as Algo σ≤0.20, θ≤3).
    """
    p = algo_process(params_b, model_type)
    # intensity for 3B standard = 1.0
    rel = p.intensity  # already vs golden 3B standard

    def sc(key: str) -> float:
        v = REF_3B_STANDARD[key] * rel
        v = max(FLOOR[key], min(CEILING[key], v))
        return round(v, 3)

    # Ramp: smaller → longer/gentler start (respect J-space prefill)
    if params_b <= 2:
        ramp_tokens, ramp_start = 18, 0.10
    elif params_b <= 5:
        ramp_tokens, ramp_start = 15, 0.15
    elif params_b <= 12:
        ramp_tokens, ramp_start = 12, 0.18
    else:
        ramp_tokens, ramp_start = 12, 0.20

    # Logit surface tip: off on small; light on mid/large
    if params_b < 8:
        logit_a = 0.0
    elif params_b < 16:
        logit_a = 0.05
    else:
        logit_a = 0.0  # default off; A/B at 0.15 only after residual is healthy

    temp = 0.85 if params_b <= 3 else 0.80

    # --- Splat *geometry* (NOT the Algo process σ) ---
    # Force √-law does not set scar width. 27B-era hand knobs used splat_sigma=40
    # on a larger residual/emb manifold. Prefer field auto-σ and hidden dim when known.
    # Priors (4B hidden=2560 field_σ≈7.6 → σ≈12; 27B hidden≈5k field_σ≈11 → σ≈40-ish).
    if params_b <= 5:
        # Residual walk δ~100 on 4B: field emb σ~7.6 is too narrow for scars (S3 cold).
        # 27B-era σ=40 late-climbs. Mid ~22 + soft mass from splat_lane_4b.
        splat_sigma, min_dist, delta_thr = 22.0, 16.0, 90.0
        splat_force_scale = 0.14
        splat_fmax = 16.0
        online_iv = 8
    elif params_b <= 14:
        splat_sigma, min_dist, delta_thr = 20.0, 16.0, 85.0
        splat_force_scale = 0.18
        splat_fmax = sc("splat_force_max")
        online_iv = 6
    else:
        splat_sigma, min_dist, delta_thr = 40.0, 30.0, 70.0
        splat_force_scale = 0.25
        splat_fmax = sc("splat_force_max")
        online_iv = 6

    return {
        "force_cap": sc("force_cap"),
        "splat_force_max": splat_fmax,
        "splat_force_scale": splat_force_scale,
        "splat_sigma": splat_sigma,
        "min_splat_dist": min_dist,
        "splat_delta_threshold": delta_thr,
        "online_splat_interval": online_iv,
        "field_wake_max": sc("field_wake_max"),
        "goal_force_max": sc("goal_force_max"),
        "goal_force_scale": sc("goal_force_scale"),
        "field_wake_scale": sc("field_wake_scale"),
        "force_ramp_tokens": ramp_tokens,
        "force_ramp_start": ramp_start,
        "field_logit_alpha": logit_a,
        "targeted_splat_only": True,
        "temperature": temp,
        "_scale_sqrt": p.scale,
        "_intensity": p.intensity,
        "_type_mult": p.type_mult,
        "_algo_sigma": p.sigma,
        "_algo_theta": p.theta,
        "_algo_beta": p.beta,
        "_algo_repulsion": p.loop_repulsion,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Scale physics knobs for model size (Algo_WIP √-law → swarm)"
    )
    ap.add_argument("--params", type=float, required=True, help="Model size in billions")
    ap.add_argument(
        "--type",
        default="instruct",
        choices=list(TYPE_MULT.keys()),
        help="Model type (thinking/coding need much gentler force)",
    )
    ap.add_argument(
        "--toml",
        action="store_true",
        help="Print only [physics] fragment for pasting",
    )
    ap.add_argument(
        "--algo-only",
        action="store_true",
        help="Print only Algo_WIP σ/θ/β/repulsion (legacy process params)",
    )
    args = ap.parse_args()

    kn = swarm_knobs(args.params, args.type)
    ap_ = algo_process(args.params, args.type)

    if args.algo_only:
        print(f"# Algo_WIP process params  ~{args.params}B  type={args.type}")
        print(f"scale (√p/3)     = {ap_.scale}")
        print(f"type_mult        = {ap_.type_mult}")
        print(f"force_intensity  = {ap_.intensity}")
        print(f"sigma            = {ap_.sigma}   # jiggle  [0.04, 0.20]")
        print(f"theta            = {ap_.theta}   # drift   [0.5, 3.0]")
        print(f"beta             = {ap_.beta}    # inv-T   [40, 150]")
        print(f"loop_repulsion   = {ap_.loop_repulsion}")
        print(f"stable_zone      = {0.04 <= ap_.sigma <= 0.20 and 0.5 <= ap_.theta <= 3.0}")
        return

    if not args.toml:
        print(f"# Model ~{args.params}B  type={args.type}")
        print(f"# √(params/3B) scale = {ap_.scale}")
        print(f"# type_mult          = {ap_.type_mult}")
        print(f"# force intensity    = {ap_.intensity}  (= scale × type_mult)")
        print(
            f"# Algo_WIP process   : σ={ap_.sigma} θ={ap_.theta} "
            f"β={ap_.beta} rep={ap_.loop_repulsion}"
        )
        print(f"# Anchor: 3B standard mid-zone × intensity, clamp FLOOR…CEILING")
        print(f"# See docs/MODEL_SIZE_PHYSICS_SCALING.md  |  Algo_WIPjuly.md")
        print()
        print("# Suggested hydrodynamic-swarm knobs (starting point — smoke 40–50 tok):")
        print()

    print("[physics]")
    for k in (
        "force_cap",
        "splat_force_scale",
        "splat_force_max",
        "splat_sigma",
        "min_splat_dist",
        "splat_delta_threshold",
        "online_splat_interval",
        "field_wake_max",
        "goal_force_max",
        "goal_force_scale",
        "field_wake_scale",
        "force_ramp_tokens",
        "force_ramp_start",
        "field_logit_alpha",
        "targeted_splat_only",
    ):
        v = kn[k]
        if isinstance(v, bool):
            print(f"{k} = {'true' if v else 'false'}")
        else:
            print(f"{k} = {v}")

    if not args.toml:
        print()
        print("[generation]")
        print(f"temperature = {kn['temperature']}")
        print()
        print("# Keep: targeted_splat_only=true, force ramp on, short smokes first.")
        print("# Do NOT paste 27B caps onto 1B–4B — that is the gibberish path.")
        print("# splat_sigma / min_splat_dist are GEOMETRY (field σ, hidden) — not Algo process σ.")
        print("# Hierarchy uses with_scale_ref(δ, splat_delta_threshold) — not absolute 20/30.")
        print("# Model type often matters more than size (thinking 0.4×, coding 0.27×).")


if __name__ == "__main__":
    main()
