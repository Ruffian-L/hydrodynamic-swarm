#!/usr/bin/env python3
"""
Thin Bridge: 3surface safetensors → SplatRag MemoryRecords + HotGeometry.

Usage:
    python3 thin_bridge.py <input.safetensors> <output_dir>

Converts 3surface splats (mu + alpha + sigma + lambda) into:
1. SplatRag cold_store.jsonl (MemoryRecords with text from mu projection)
2. hot_geometry.json (HotSplats with positions derived from mu)

The bridge does NOT require a model — it uses the existing HotGeometry
hash-embedding for cold store, and optionally projects mu to 3D for
position if a projection matrix exists.
"""

import sys
import json
import time
import uuid
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Optional

# Try to import safetensors
try:
    import safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not installed, trying numpy fallback")

# SplatRag imports
sys.path.insert(0, "/home/ruffianl/projects/splatrag/src")
from splatrag.cold_store import MemoryRecord, ColdStore
from splatrag.hot_geometry import HotGeometry, HotSplat

# Niod bridge for 4096→64 projection (if available)
try:
    sys.path.insert(0, "/home/ruffianl/hydrodynamic-swarm-3surface/src/bridge")
    from niod_bridge import load_proj
    HAS_NIOD = True
except ImportError:
    HAS_NIOD = False
    print("WARNING: niod_bridge not available, using hash-embedding only")


@dataclass
class BridgeSplat:
    """Intermediate representation bridging 3surface → SplatRag."""
    id: str
    mu: np.ndarray          # 2560D or 4096D latent vector
    alpha: float             # gain/valence
    sigma: float             # uncertainty
    lambda_: float           # decay rate
    scale: float             # spatial scale
    created_at: float        # timestamp
    is_anchor: bool          # anchor splat flag
    flux: float              # flux magnitude
    friction: float          # friction coefficient
    current_dim: int         # dimensionality of mu

    @classmethod
    def from_safetensors(cls, idx: int, tensors: dict) -> "BridgeSplat":
        return cls(
            id=f"splat_{idx:04d}",
            mu=tensors["mu"][idx],
            alpha=float(tensors["alpha"][idx]),
            sigma=float(tensors["sigma"][idx]),
            lambda_=float(tensors["lambda"][idx]),
            scale=float(tensors["scale"][idx]),
            created_at=float(tensors["created_at"][idx]),
            is_anchor=bool(tensors["is_anchor"][idx]),
            flux=float(tensors["flux"][idx]),
            friction=float(tensors["friction"][idx]),
            current_dim=int(tensors["current_dim"][idx]),
        )

    def to_memory_record(self, text: str = "") -> MemoryRecord:
        """Convert to SplatRag MemoryRecord."""
        return MemoryRecord.make(
            text=text or f"[splat:{self.id}] alpha={self.alpha:.3f} sigma={self.sigma:.1f}",
            domain="other",
            source="3surface_bridge",
            meta={
                "splat_id": self.id,
                "mu_dim": int(self.mu.shape[0]),
                "mu_norm": float(np.linalg.norm(self.mu)),
                "alpha": self.alpha,
                "sigma": self.sigma,
                "lambda_": self.lambda_,
                "scale": self.scale,
                "is_anchor": self.is_anchor,
                "flux": self.flux,
                "friction": self.friction,
            },
        )

    def to_hot_splat(self, position: list[float]) -> HotSplat:
        """Convert to SplatRag HotSplat."""
        return HotSplat(
            cold_id=self.id,
            domain="other",
            position=position,
            mass=abs(self.alpha) + 1.0,  # mass from gain magnitude
            valence=self.alpha,
            opacity=min(1.0, self.sigma / 100.0),  # opacity from uncertainty
        )


def project_mu_to_3d(mu: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """Project high-D mu to 3D using a learned projection matrix."""
    if proj_matrix.shape[0] < 3:
        # Fallback: use first 3 components
        return mu[:3]
    return proj_matrix[:3] @ mu


def bridge_safetensors_to_splatrag(
    input_path: str,
    output_dir: str,
) -> dict:
    """
    Main bridge function.
    
    Returns summary dict with counts and stats.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load safetensors
    if not HAS_SAFETENSORS:
        print(f"ERROR: safetensors required, got {input_path}")
        return {"error": "safetensors not installed"}

    with safetensors.safe_open(input_path, framework="np", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    n_splats = tensors["mu"].shape[0]
    print(f"Loaded {n_splats} splats from {input_path.name}")

    # Create bridge splats
    bridge_splats = []
    for i in range(n_splats):
        bs = BridgeSplat.from_safetensors(i, tensors)
        bridge_splats.append(bs)

    # Project mu to 3D if possible
    proj_matrix = None
    if HAS_NIOD:
        try:
            proj_fwd, _ = load_proj()
            # proj_fwd is 4096→64, we need 64→3 or direct 4096→3
            # For now, use first 3 components of mu as position
            print(f"Using first 3 components of mu as position (no 3D proj available)")
        except Exception as e:
            print(f"Projection load failed: {e}")

    # Create MemoryRecords
    records = []
    for bs in bridge_splats:
        # Generate text from mu stats
        mu_norm = float(np.linalg.norm(bs.mu))
        mu_mean = float(np.mean(bs.mu))
        mu_std = float(np.std(bs.mu))
        text = (
            f"[splat:{bs.id}] mu_norm={mu_norm:.1f} mu_mean={mu_mean:.2f} "
            f"mu_std={mu_std:.2f} alpha={bs.alpha:.3f} sigma={bs.sigma:.1f}"
        )
        record = bs.to_memory_record(text=text)
        records.append(record)

    # Save cold store
    cold_store_path = output_dir / "cold_store.jsonl"
    cold_store = ColdStore(cold_store_path)
    cold_store.append_many(records)
    print(f"Saved {len(records)} MemoryRecords to {cold_store_path}")

    # Create HotGeometry
    hot_geom = HotGeometry(output_dir / "hot_geometry.json")
    hot_splats = []
    for bs in bridge_splats:
        # Use first 3 components of mu as position
        pos = bs.mu[:3].tolist()
        hs = bs.to_hot_splat(position=pos)
        hot_splats.append(hs)

    # Manually set splats since from_records expects records
    hot_geom.splats = hot_splats
    hot_geom.save()
    print(f"Saved {len(hot_splats)} HotSplats to {output_dir / 'hot_geometry.json'}")

    # Summary
    summary = {
        "input_file": str(input_path),
        "n_splats": n_splats,
        "n_records": len(records),
        "n_hot_splats": len(hot_splats),
        "output_dir": str(output_dir),
        "mu_dim": int(tensors["mu"].shape[1]),
        "alpha_range": [float(tensors["alpha"].min()), float(tensors["alpha"].max())],
        "sigma_range": [float(tensors["sigma"].min()), float(tensors["sigma"].max())],
        "is_anchor_count": int(tensors["is_anchor"].sum()),
        "bridge_time": time.time(),
    }

    # Save summary
    with open(output_dir / "bridge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBridge complete. Summary saved to {output_dir / 'bridge_summary.json'}")
    return summary


def main():
    if len(sys.argv) < 3:
        print("Usage: thin_bridge.py <input.safetensors> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    summary = bridge_safetensors_to_splatrag(input_path, output_dir)

    if "error" in summary:
        sys.exit(1)

    print(f"\nBridge Summary:")
    print(f"  Splats: {summary['n_splats']}")
    print(f"  Mu dim: {summary['mu_dim']}")
    print(f"  Alpha range: {summary['alpha_range']}")
    print(f"  Sigma range: {summary['sigma_range']}")
    print(f"  Anchor count: {summary['is_anchor_count']}")


if __name__ == "__main__":
    main()
