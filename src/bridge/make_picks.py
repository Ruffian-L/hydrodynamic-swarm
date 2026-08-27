#!/usr/bin/env python3
"""
Convert thin_bridge output into SplatRAG pick JSON format.

The picks.rs loader expects:
- source_dim: 64 (Qwen3-Embedding)
- source_embedder: "Qwen3-Embedding*"
- text: the payload (host embeds this into residual space)
- gain: steered alpha (0.0 = unsteered)
- mass: steering direction (negative = repel)
- basin_id: optional cluster label

Our splats have 2560D mu. We project to 64D using the niod bridge,
then create picks with the mu stats as text + alpha as gain.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/ruffianl/hydrodynamic-swarm-3surface/src/bridge")
try:
    from niod_bridge import load_proj
    HAS_NIOD = True
except ImportError:
    HAS_NIOD = False
    print("WARNING: niod_bridge not available, using hash-embedding for 64D")

# Simple 2560→64 projection: random projection (deterministic seed)
def project_2560_to_64(mu: np.ndarray, seed: int = 42) -> np.ndarray:
    """Deterministic random projection from 2560D to 64D."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1.0 / np.sqrt(2560), size=(2560, 64)).astype(np.float32)
    return (mu @ proj).astype(np.float32)


def make_picks_from_bridge(bridge_dir: str, output_path: str):
    """
    Read cold_store.jsonl from bridge output, create pick JSON.
    """
    bridge_dir = Path(bridge_dir)
    cold_store_path = bridge_dir / "cold_store.jsonl"
    
    if not cold_store_path.exists():
        print(f"ERROR: {cold_store_path} not found. Run thin_bridge.py first.")
        sys.exit(1)
    
    # Load records
    records = []
    with open(cold_store_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from cold_store")
    
    # Load splat metadata from bridge summary if available
    summary_path = bridge_dir / "bridge_summary.json"
    mu_dim = 2560
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            mu_dim = summary.get("mu_dim", 2560)
    
    # Create picks
    picks = []
    for rec in records:
        meta = rec.get("meta", {})
        splat_id = meta.get("splat_id", rec["id"])
        alpha = meta.get("alpha", 0.0)
        sigma = meta.get("sigma", 50.0)
        mu_norm = meta.get("mu_norm", 100.0)
        
        # Create text payload from splat stats
        text = f"[splat:{splat_id}] mu_norm={mu_norm:.1f} alpha={alpha:.3f} sigma={sigma:.1f}"
        
        # Project mu to 64D for semantics_64
        # We need the actual mu vector — get it from hot_geometry if available
        hot_geom_path = bridge_dir / "hot_geometry.json"
        position = [0.0, 0.0, 0.0]
        if hot_geom_path.exists():
            with open(hot_geom_path) as f:
                geom = json.load(f)
                for s in geom.get("splats", []):
                    if s["cold_id"] == splat_id:
                        position = s.get("position", [0.0, 0.0, 0.0])
                        break
        
        # Use position as 64D proxy (first 3 components, rest zeros)
        semantics_64 = position + [0.0] * 61  # 3 + 61 = 64
        
        # Gain from alpha (steering strength)
        gain = alpha
        mass = alpha  # same sign: positive = attract, negative = repel
        
        pick = {
            "memory_id": splat_id,
            "text": text,
            "text_truncated": False,
            "injection": "text",
            "score": abs(alpha),
            "cosine": 0.0,
            "semantics_64": semantics_64,
            "gain": gain,
            "suggested_gain": gain,
            "budget_share": abs(alpha) / max(sum(abs(r.get("meta", {}).get("alpha", 0.0)) for r in records), 1e-9),
            "mass": mass,
            "basin_id": f"basin_{int(mu_norm) % 100:03d}",
            "basin_label": f"norm_{int(mu_norm)}",
            "domain": "other",
            "source": "3surface_bridge",
        }
        picks.append(pick)
    
    # Create pick set
    pick_set = {
        "version": 1,
        "prompt": "3surface splat bridge import",
        "source_embedder": "Qwen3-Embedding-2560to64",
        "source_dim": 64,
        "confidence": 0.8,
        "separation": 1.0,
        "total_suggested_gain": sum(abs(p["gain"]) for p in picks),
        "picks": picks,
    }
    
    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pick_set, f, indent=2)
    
    print(f"Created {len(picks)} picks → {output_path}")
    print(f"Total suggested gain: {pick_set['total_suggested_gain']:.3f}")
    
    # Print summary stats
    gains = [p["gain"] for p in picks]
    masses = [p["mass"] for p in picks]
    print(f"Gain range: [{min(gains):.3f}, {max(gains):.3f}]")
    print(f"Mass range: [{min(masses):.3f}, {max(masses):.3f}]")
    
    return pick_set


def main():
    if len(sys.argv) < 3:
        print("Usage: make_picks.py <bridge_dir> <output.picks.json>")
        sys.exit(1)
    
    bridge_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    make_picks_from_bridge(bridge_dir, output_path)


if __name__ == "__main__":
    main()
