#!/usr/bin/env python3
"""
Fix the bridge picks projection.

Problem: bridge_picks_A.json has 21 picks with empty semantics_64 (len=0).
Cause: hot_geometry.json doesn't exist, so position fell back to [0,0,0]+61 zeros.
Fix: Project actual 3840D mu vectors from splat_memory.safetensors to 64D
     using deterministic random projection, then regenerate picks.

Usage:
    python3 fix_picks_projection.py <input_picks.json> <output_picks.json>
"""

import sys
import json
import numpy as np
import safetensors
from pathlib import Path

def project_3840_to_64(mu: np.ndarray, seed: int = 42) -> np.ndarray:
    """Deterministic random projection from 3840D to 64D."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1.0 / np.sqrt(3840), size=(3840, 64)).astype(np.float32)
    return (mu @ proj).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print("Usage: fix_picks_projection.py <input_picks.json> <output_picks.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    # Load existing picks
    with open(input_path) as f:
        pick_set = json.load(f)
    
    print(f"Loaded {len(pick_set['picks'])} picks from {input_path}")
    
    # Load splat memory
    handle = safetensors.safe_open(str(input_path.parent / "splat_memory.safetensors"), framework='pt')
    mu_tensor = handle.get_tensor('mu')
    alpha_tensor = handle.get_tensor('alpha')
    
    print(f"Splat memory: {mu_tensor.shape[0]} splats, {mu_tensor.shape[1]}D each")
    
    # Create projection matrix
    print("Creating 3840→64 projection matrix...")
    proj_matrix = project_3840_to_64(np.zeros(3840))  # Just to get the matrix
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0 / np.sqrt(3840), size=(3840, 64)).astype(np.float32)
    
    # Project each splat's mu to 64D
    print("Projecting mu vectors to 64D...")
    projections = {}
    for i in range(mu_tensor.shape[0]):
        mu_vec = mu_tensor[i].numpy()
        proj_64 = (mu_vec @ proj).astype(np.float32)
        projections[i] = proj_64.tolist()
        print(f"  Splats {i}: mu_norm={np.linalg.norm(mu_vec):.2f} → 64D_norm={np.linalg.norm(proj_64):.4f}")
    
    # Now we need to map pick memory_ids to tensor indices
    # The picks were generated from thin_bridge.py which assigns IDs like "splat_0000", "splat_0001", etc.
    # But the actual pick IDs are UUIDs from the cold_store
    # We need to match by alpha values (which should be unique enough)
    
    pick_ids = [p['memory_id'] for p in pick_set['picks']]
    print(f"\nPick IDs (first 5): {pick_ids[:5]}")
    
    # Try to match picks to tensor indices by alpha value
    # The picks have '_splat_mu_l2' which is the L2 norm of the original mu
    # We can match by comparing mu norms
    
    matched = 0
    unmatched = []
    
    for pick in pick_set['picks']:
        pick_mu_l2 = pick.get('_splat_mu_l2', None)
        if pick_mu_l2 is None:
            unmatched.append(pick['memory_id'])
            continue
        
        # Find best matching tensor index by mu norm
        best_idx = None
        best_diff = float('inf')
        
        for i in range(mu_tensor.shape[0]):
            mu_norm = np.linalg.norm(mu_tensor[i].numpy())
            diff = abs(mu_norm - pick_mu_l2)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        
        if best_diff < 0.1:  # Match tolerance
            projections_64 = projections[best_idx]
            pick['semantics_64'] = projections_64
            pick['source'] = f"3surface_bridge:idx_{best_idx}"
            matched += 1
            print(f"  Matched {pick['memory_id'][:8]}... → tensor idx {best_idx} (diff={best_diff:.4f})")
        else:
            unmatched.append(pick['memory_id'])
            print(f"  No match for {pick['memory_id'][:8]}... (mu_l2={pick_mu_l2:.2f}, best_diff={best_diff:.4f})")
    
    print(f"\nMatched: {matched}/{len(pick_set['picks'])}")
    print(f"Unmatched: {len(unmatched)}")
    
    # For unmatched picks, use hash-based position
    for pick in pick_set['picks']:
        if len(pick['semantics_64']) == 0:
            # Hash-based position for unmatched picks
            uid = pick['memory_id']
            h = hash(uid) & 0xFFFFFFFF
            pos = [
                (h >> 0) & 0xFF - 128,
                (h >> 8) & 0xFF - 128,
                (h >> 16) & 0xFF - 128,
            ] + [0.0] * 61
            pick['semantics_64'] = pos
            print(f"  Hash-positioned {pick['memory_id'][:8]}...")
    
    # Verify all picks now have valid semantics_64
    for pick in pick_set['picks']:
        assert len(pick['semantics_64']) == 64, f"Pick {pick['memory_id']} has {len(pick['semantics_64'])}D semantics_64"
        assert any(x != 0 for x in pick['semantics_64']), f"Pick {pick['memory_id']} has all-zero semantics_64"
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pick_set, f, indent=2)
    
    print(f"\nWrote fixed picks to {output_path}")
    
    # Print summary
    positions = [np.array(p['semantics_64']) for p in pick_set['picks']]
    positions = np.array(positions)
    print(f"\nPosition statistics:")
    print(f"  Mean norm: {np.mean(np.linalg.norm(positions, axis=1)):.4f}")
    print(f"  Min norm: {np.min(np.linalg.norm(positions, axis=1)):.4f}")
    print(f"  Max norm: {np.max(np.linalg.norm(positions, axis=1)):.4f}")
    
    # Check separation
    if len(positions) > 1:
        dists = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dists.append(np.linalg.norm(positions[i] - positions[j]))
        print(f"  Pairwise distance mean: {np.mean(dists):.4f}")
        print(f"  Pairwise distance min: {np.min(dists):.4f}")
        print(f"  Pairwise distance max: {np.max(dists):.4f}")
    
    return pick_set


if __name__ == "__main__":
    main()
