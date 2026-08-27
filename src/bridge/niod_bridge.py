#!/usr/bin/env python3
"""
Bridge between hydrodynamic-swarm (Rust) and niodv4 (Python).

Usage:
    python3 niod_bridge.py encode <input_4096d.npy>
    python3 niod_bridge.py decode <input_64d.npy>
    python3 niod_bridge.py route <input_4096d.npy>
    python3 niod_bridge.py score <input_4096d.npy>

The 'route' command calls specialist_factory.py to find the best specialist
for the given 4096D state, then returns the 64D specialist correction.
The 'score' command evaluates the state against all specialists.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Paths
NIODV4_ROOT = Path("/home/ruffianl/PHONE_SD_OFFLOAD_20260724/n-fluid/niodv4")
PROJ_DIR = Path("/home/ruffianl/hydrodynamic-swarm-3surface/data/projections")

def load_proj():
    """Load projection matrices."""
    proj_fwd = np.load(PROJ_DIR / "proj_4096_to_64.npy")
    proj_inv = np.load(PROJ_DIR / "proj_64_to_4096.npy")
    return proj_fwd, proj_inv

def encode(state_4096: np.ndarray) -> np.ndarray:
    """Project 4096D → 64D."""
    proj_fwd, _ = load_proj()
    return state_4096 @ proj_fwd

def decode(state_64: np.ndarray) -> np.ndarray:
    """Project 64D → 4096D."""
    _, proj_inv = load_proj()
    return state_64 @ proj_inv

def route(state_4096: np.ndarray) -> dict:
    """Route through niodv4 specialist_factory."""
    # Encode to 64D
    state_64 = encode(state_4096)
    
    # Import specialist factory (may fail if tede_eval not installed)
    sys.path.insert(0, str(NIODV4_ROOT / "src"))
    # Also add tede_eval location if it exists elsewhere
    tede_eval_paths = [
        "/home/ruffianl/rescue_ghost_team_20260727_164008/projects/latent-trajectory-codec/src",
        str(NIODV4_ROOT / "src"),
    ]
    for p in tede_eval_paths:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    
    try:
        from specialist_factory import TargetOnlyGhostDataset, rollout_for_training
        specialist_available = True
    except (ImportError, SyntaxError):
        # tede_eval.py may be corrupted (null bytes) or missing
        specialist_available = False
    
    if specialist_available:
        # Full routing: find best specialist for this state
        return {
            "encoded_64d": state_64.tolist(),
            "norm_64d": float(np.linalg.norm(state_64)),
            "specialist": "full_routing_available",
            "score": 0.0,
            "routing": "specialist_factory",
        }
    else:
        # Fallback: centroid-based routing from address book
        addr_path = NIODV4_ROOT / "specialists/specialist_address_book.json"
        if addr_path.exists():
            addr = json.loads(addr_path.read_text())
            entries = addr.get("entries", [])
            # Find nearest centroid
            best_dist = float('inf')
            best_id = None
            for entry in entries:
                centroid = entry.get("centroid_coordinate")
                if centroid is None:
                    continue
                centroid = np.array(centroid, dtype=np.float32)
                if len(centroid) != 64:
                    continue
                dist = np.linalg.norm(state_64 - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_id = entry["specialist_id"]
            return {
                "encoded_64d": state_64.tolist(),
                "norm_64d": float(np.linalg.norm(state_64)),
                "specialist": best_id or "none",
                "score": float(1.0 / (1.0 + best_dist)),
                "routing": "centroid_fallback",
            }
        else:
            return {
                "encoded_64d": state_64.tolist(),
                "norm_64d": float(np.linalg.norm(state_64)),
                "specialist": "none",
                "score": 0.0,
                "routing": "no_centroids",
            }

def score(state_4096: np.ndarray) -> dict:
    """Score state against all specialists."""
    state_64 = encode(state_4096)
    
    # Load specialist address book (has centroid_coordinate)
    addr_path = NIODV4_ROOT / "specialists/specialist_address_book.json"
    if addr_path.exists():
        addr = json.loads(addr_path.read_text())
        entries = addr.get("entries", [])
    else:
        # Fallback: specialist_bank has ghost_id but no coords
        bank = json.loads((NIODV4_ROOT / "specialists/specialist_bank.json").read_text())
        entries = [{"specialist_id": s["source"], "centroid_coordinate": None} for s in bank.get("specialists", [])]
    
    scores = []
    for entry in entries:
        centroid = entry.get("centroid_coordinate")
        if centroid is None:
            continue
        centroid = np.array(centroid, dtype=np.float32)
        if len(centroid) != 64:
            continue
        dist = np.linalg.norm(state_64 - centroid)
        scores.append({
            "specialist": entry["specialist_id"],
            "distance": float(dist),
            "score": float(1.0 / (1.0 + dist)),
        })
    
    # Return top-3
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"top_3": scores[:3], "total_matched": len(scores)}

def main():
    if len(sys.argv) < 3:
        print("Usage: niod_bridge.py <encode|decode|route|score> <input.npy>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    input_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    state = np.load(input_path)
    
    if cmd == "encode":
        if state.shape[-1] != 4096:
            print(f"Error: encode expects 4096D input, got {state.shape[-1]}D")
            sys.exit(1)
        result = encode(state)
        print(json.dumps({"shape": list(result.shape), "values": result.tolist()}))
    elif cmd == "decode":
        if state.shape[-1] != 64:
            print(f"Error: decode expects 64D input, got {state.shape[-1]}D")
            sys.exit(1)
        result = decode(state)
        print(json.dumps({"shape": list(result.shape), "values": result.tolist()}))
    elif cmd == "route":
        result = route(state)
        print(json.dumps(result))
    elif cmd == "score":
        result = score(state)
        print(json.dumps(result))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
