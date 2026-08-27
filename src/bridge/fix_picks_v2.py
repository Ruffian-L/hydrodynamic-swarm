#!/usr/bin/env python3
"""
Fix bridge picks projection — v2.

Problem: bridge_picks_A.json has 21 picks with empty semantics_64 (len=0).
Source: A_splat_memory.safetensors (21 splats, 2560D each).
Fix: Project actual 2560D mu vectors from A_splat_memory.safetensors to 64D
     using deterministic random projection, match by mu norm.

Usage:
    python3 fix_picks_v2.py <input_picks.json> <input_safetensors> <output_picks.json>
"""

import sys
import json
import numpy as np
import safetensors
from pathlib import Path

def project_2560_to_64(mu: np.ndarray, seed: int = 42) -> np.ndarray:
    """Deterministic random projection from 2560D to 64D."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1.0 / np.sqrt(2560), size=(2560, 64)).astype(np.float32)
    return (mu @ proj).astype(np.float32)


def main():
    if len(sys.argv) < 4:
        print("Usage: fix_picks_v2.py <input_picks.json> <input_safetensors> <output_picks.json>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    safetensors_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    
    # Load existing picks
    with open(input_path) as f:
        pick_set = json.load(f)
    
    print(f"Loaded {len(pick_set['picks'])} picks from {input_path}")
    
    # Load splat memory (the source of the picks)
    handle = safetensors.safe_open(str(safetensors_path), framework='pt')
    mu_tensor = handle.get_tensor('mu')
    alpha_tensor = handle.get_tensor('alpha')
    
    print(f"Splat memory: {mu_tensor.shape[0]} splats, {mu_tensor.shape[1]}D each")
    print(f"Alpha range: [{alpha_tensor.min().item():.4f}, {alpha_tensor.max().item():.4f}]")
    
    # Compute mu norms for all splats
    mu_norms = []
    for i in range(mu_tensor.shape[0]):
        mu_vec = mu_tensor[i].numpy()
        norm = np.linalg.norm(mu_vec)
        mu_norms.append(norm)
        print(f"  Splat {i}: mu_norm={norm:.2f}, alpha={alpha_tensor[i].item():.4f}")
    
    # Create projection matrix
    print(f"\nCreating 2560→64 projection matrix (seed=42)...")
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0 / np.sqrt(2560), size=(2560, 64)).astype(np.float32)
    
    # Project each splat's mu to 64D
    print("Projecting mu vectors to 64D...")
    projections_64 = {}
    for i in range(mu_tensor.shape[0]):
        mu_vec = mu_tensor[i].numpy()
        proj_64 = (mu_vec @ proj).astype(np.float32)
        projections_64[i] = proj_64.tolist()
        print(f"  Splat {i}: 2560D_norm={mu_norms[i]:.2f} → 64D_norm={np.linalg.norm(proj_64):.4f}")
    
    # Match picks to splats by _splat_mu_l2
    print(f"\nMatching picks to splats by mu norm...")
    matched = 0
    unmatched = []
    
    # Sort splats by mu_norm for greedy matching
    splat_indices = list(range(mu_tensor.shape[0]))
    
    for pick in pick_set['picks']:
        pick_mu_l2 = pick.get('_splat_mu_l2', None)
        if pick_mu_l2 is None:
            unmatched.append(pick['memory_id'])
            continue
        
        # Find best matching splat by mu norm
        best_idx = None
        best_diff = float('inf')
        
        for idx in splat_indices:
            diff = abs(mu_norms[idx] - pick_mu_l2)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        
        if best_diff < 1.0:  # Match tolerance
            projections_64_pick = projections_64[best_idx]
            pick['semantics_64'] = projections_64_pick
            pick['source'] = f"3surface_bridge:idx_{best_idx}"
            pick['basin_id'] = f"basin_{best_idx:03d}"
            matched += 1
            print(f"  Matched {pick['memory_id'][:8]}... → splat idx {best_idx} (diff={best_diff:.4f}, alpha={alpha_tensor[best_idx].item():.4f})")
            splat_indices.remove(best_idx)
        else:
            unmatched.append(pick['memory_id'])
            print(f"  No match for {pick['memory_id'][:8]}... (mu_l2={pick_mu_l2:.2f}, best_diff={best_diff:.4f})")
    
    print(f"\nMatched: {matched}/{len(pick_set['picks'])}")
    print(f"Unmatched: {len(unmatched)}")
    
    # For unmatched picks, use hash-based position
    for pick in pick_set['picks']:
        if len(pick.get('semantics_64', [])) == 0:
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
    norms = np.linalg.norm(positions, axis=1)
    print(f"  Mean norm: {np.mean(norms):.4f}")
    print(f"  Min norm: {np.min(norms):.4f}")
    print(f"  Max norm: {np.max(norms):.4f}")
    
    # Check separation
    if len(positions) > 1:
        dists = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dists.append(np.linalg.norm(positions[i] - positions[j]))
        print(f"  Pairwise distance mean: {np.mean(dists):.4f}")
        print(f"  Pairwise distance min: {np.min(dists):.4f}")
        print(f"  Pairwise distance max: {np.max(dists):.4f}")
    
    # Print gain range
    gains = [p['gain'] for p in pick_set['picks']]
    print(f"\nGain range: [{min(gains):.4f}, {max(gains):.4f}]")
    print(f"Total |gain|: {sum(abs(g) for g in gains):.4f}")
    
    return pick_set


if __name__ == "__main__":
    main()
