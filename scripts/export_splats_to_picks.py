#!/usr/bin/env python3
"""
Thin export: 3surface splats (safetensors) → SplatRag pick JSON.

Maps:
  splat.mu (2560D) → pick.text (placeholder; actual embed via prefill at import time)
  splat.alpha → pick.gain (steering direction/magnitude)
  splat.scale → pick.domain (Fine/Medium/Coarse heuristic)
  splat.is_anchor → pick.basin_id (anchor splats get basin labels)
  splat.created_at → pick.ingested_at (unix timestamp)

Acceptance criteria for the bridge:
  1. Pick file parses as valid MemoryPickSet (source_dim=64, injection="text")
  2. Import at force_cap=0.5 produces measurable steering shift
  3. Negative alpha → negative gain → repel behavior
"""

import json
import sys
import os
import hashlib
import time
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try safetensors
try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def splat_to_pick(splat_id: str, splat: dict, idx: int) -> dict:
    """Convert a single splat tensor dict to a Pick entry."""
    mu = splat.get("mu")
    alpha = splat.get("alpha")
    sigma = splat.get("sigma")
    lambda_val = splat.get("lambda")
    scale = splat.get("scale")
    is_anchor = splat.get("is_anchor")
    created_at = splat.get("created_at")
    friction = splat.get("friction")
    flux = splat.get("flux")
    current_dim = splat.get("current_dim")

    # Generate a deterministic UUID from splat_id + idx
    uuid_hash = hashlib.md5(f"{splat_id}:{idx}".encode()).hexdigest()
    memory_id = f"{uuid_hash[:8]}-{uuid_hash[8:12]}-{uuid_hash[12:16]}-{uuid_hash[16:20]}-{uuid_hash[20:32]}"

    # Extract alpha (steering gain)
    alpha_val = float(alpha) if alpha is not None else 0.0

    # Determine domain from scale
    if scale is not None:
        scale_val = float(scale)
        if scale_val < 0.5:
            domain = "Fine"
        elif scale_val < 2.0:
            domain = "Medium"
        else:
            domain = "Coarse"
    else:
        domain = "Medium"

    # Basin ID for anchors
    basin_id = None
    basin_label = None
    if is_anchor is not None and float(is_anchor) > 0.5:
        basin_id = f"basin-{splat_id[:8]}"
        basin_label = f"anchor-{splat_id[:8]}"

    # Create timestamp
    if created_at is not None:
        ingested_at = int(float(created_at))
    else:
        ingested_at = int(time.time())

    # Text placeholder - actual embedding happens at import via prefill
    # We store the splat metadata as a description
    text = f"[splat:{splat_id}] alpha={alpha_val:.4f} dim={mu.shape[-1] if mu is not None else 'unknown'}"

    return {
        "memory_id": memory_id,
        "text": text,
        "text_truncated": False,
        "injection": "text",
        "score": abs(alpha_val),
        "cosine": 0.0,
        "semantics_64": [],  # Will be populated by SplatRag embedder
        "gain": alpha_val,
        "suggested_gain": alpha_val * 0.8,  # Slightly conservative fallback
        "budget_share": 0.1,
        "mass": 1.0 if alpha_val >= 0 else -1.0,  # Negative alpha → negative mass → repel
        "basin_id": basin_id,
        "basin_label": basin_label,
        "domain": domain,
        "source": f"3surface:{splat_id}",
        "ingested_at": ingested_at,
        # Extra metadata for bridge verification
        "_splat_mu_l2": float(torch.linalg.vector_norm(mu)) if mu is not None else 0.0,
        "_splat_sigma": float(sigma) if sigma is not None else 0.0,
        "_splat_lambda": float(lambda_val) if lambda_val is not None else 0.0,
    }


def export_splats(safetensors_path: str, output_path: str):
    """Export all splats from a safetensors file to pick JSON."""
    if not HAS_SAFETENSORS:
        print("ERROR: safetensors not installed. Install with: pip install safetensors")
        sys.exit(1)

    print(f"Loading splats from: {safetensors_path}")
    splat_data = load_file(safetensors_path)

    # Extract individual splats
    # The safetensors file contains tensors named like:
    # mu, alpha, sigma, lambda, scale, is_anchor, created_at, friction, flux, current_dim
    # Each is shape [N, ...] where N = number of splats

    mu = splat_data.get("mu")
    alpha = splat_data.get("alpha")
    sigma = splat_data.get("sigma")
    lambda_val = splat_data.get("lambda")
    scale = splat_data.get("scale")
    is_anchor = splat_data.get("is_anchor")
    created_at = splat_data.get("created_at")
    friction = splat_data.get("friction")
    flux = splat_data.get("flux")
    current_dim = splat_data.get("current_dim")

    if mu is None:
        print("ERROR: No 'mu' tensor found in safetensors file")
        sys.exit(1)

    n_splats = mu.shape[0]
    print(f"Found {n_splats} splats")

    picks = []
    for i in range(n_splats):
        splat = {
            "mu": mu[i],
            "alpha": alpha[i] if alpha is not None else None,
            "sigma": sigma[i] if sigma is not None else None,
            "lambda": lambda_val[i] if lambda_val is not None else None,
            "scale": scale[i] if scale is not None else None,
            "is_anchor": is_anchor[i] if is_anchor is not None else None,
            "created_at": created_at[i] if created_at is not None else None,
            "friction": friction[i] if friction is not None else None,
            "flux": flux[i] if flux is not None else None,
            "current_dim": current_dim[i] if current_dim is not None else None,
        }
        pick = splat_to_pick(f"A", splat, i)
        picks.append(pick)

    # Build MemoryPickSet
    pickset = {
        "version": 1,
        "prompt": "3surface splat memory import",
        "source_embedder": "Gemma-4B-residual",
        "source_dim": 2560,  # Actual residual dim
        "confidence": 0.5,
        "separation": 0.1,
        "total_suggested_gain": sum(p["suggested_gain"] for p in picks),
        "picks": picks,
    }

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(pickset, f, indent=2)

    print(f"Exported {len(picks)} picks to: {output_path}")

    # Summary stats
    alphas = [p["gain"] for p in picks]
    print(f"Alpha range: [{min(alphas):.4f}, {max(alphas):.4f}]")
    print(f"Positive alpha: {sum(1 for a in alphas if a > 0)}")
    print(f"Negative alpha: {sum(1 for a in alphas if a < 0)}")
    print(f"Zero alpha: {sum(1 for a in alphas if a == 0)}")

    return pickset


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: export_splats_to_picks.py <safetensors_path> <output_pick.json>")
        sys.exit(1)

    safetensors_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(safetensors_path):
        print(f"ERROR: File not found: {safetensors_path}")
        sys.exit(1)

    export_splats(safetensors_path, output_path)
