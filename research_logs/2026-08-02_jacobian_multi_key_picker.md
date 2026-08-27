# Jacobian multi-key picker addresses

**Date:** 2026-08-02  
**Lane:** Jacobian / multi-key picker (instructional first-thought keys)  
**Status:** Schema + cluster + multi-key container + unit tests in-tree. Full SplatRAG cold pick out of scope.

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason (loop fire + follow-on)
- **Role:** implementation · unit tests · architecture log
- **Project:** hydrodynamic-swarm (worktree `hydrodynamic-swarm-3surface`)
- **Date written:** 2026-08-02
---

---

## Key schema (`src/jacobian.rs`)

| Type | Fields |
|------|--------|
| **`KeyPhase`** | `Answer` / `Revise` / `Settle` (maps self_reg probe strings) |
| **`DimSignature`** | sparse `dims: Vec<(dim_idx, weight)>` — top-k driving dims + non-neg weights; sorted desc; zeros dropped |
| **`JacobianKey`** | `signature`, `phase`, `step`, `turn: Option`, `text_bridge_hash: Option<u64>`, `residual_d`, `sensitivity_norm: Option` |
| **`MultiKeyAddress`** | `keys: Vec<JacobianKey>`, `cluster_threshold` (default 0.5 Jaccard-distance) |
| **`PickQuery` / `PickCluster`** | clustered pick set: cluster_id, key_indices, representative signature; `top_k` packets |

Helpers: `text_bridge_hash` (FNV-1a), `JacobianKey::from_report` (from live `JacobianReport`).

**Bridge rule:** text bridge hash only — never inject raw 64D into wrong residual D. Host embeds in its residual width; `residual_d` on the key is the contract.

---

## Distance / clustering

- **`weighted_jaccard(a, b)`** = Σ min(wₐ,wᵦ) / Σ max(wₐ,wᵦ) over union of dims.
- **`signature_distance`** = 1 − weighted_jaccard ∈ [0,1].
- **`sparse_cosine`** available as secondary metric (not default cluster).
- **`cluster_signatures(sigs, threshold)`** — union-find single-linkage: merge if distance ≤ threshold.
- **`MultiKeyAddress::emit_pick_query(top_k)`** — cluster episode keys, order by size then earliest step, truncate to top_k (k≈8 bet).

Default threshold **0.5** (tune later on real J fingerprints).

---

## Measure hook points

| Where | What exists |
|-------|-------------|
| `main.rs` ~1140 | Periodic: `jacobian.interval > 0` → `measure_jacobian_step` → print top dims/tokens. **Not yet** phase-tagged key push. |
| `main.rs` `measure_jacobian_step` | Builds `JacobianLens`, projects via `model.project_to_logits`, asserts `residual_d`. |
| Preferred (not wired) | **Event-driven:** first content token (`phase=answer` first commit), revise flip edge, settle clamp/EOS → `JacobianKey::from_report` → `MultiKeyAddress::push`. |
| Config | `[jacobian]` epsilon / top_k / max_dims / interval |

Documented in-module comment above multi-key types. Wiring keys into the decode loop is the next small step (CPU-side only once report exists).

---

## Tests (pass)

`cargo test --release -j1 --bin hydrodynamic-swarm multi_key` → **9 ok**

- weighted Jaccard identical / disjoint / partial  
- sparse cosine orthogonal / parallel  
- cluster merges near sigs  
- multi-key emit pick query non-empty (3 synthetic keys → 2 clusters, residual_d=3840)  
- phase string parse + signature cleanup  

---

## Files touched

- `src/jacobian.rs` — schema, distance, cluster, MultiKeyAddress, unit tests  
- `research_logs/2026-08-02_jacobian_multi_key_picker.md` — this log  

---

## Scope honesty (2026-08-02)

This brick is **hydro finite-diff multi-key**, not the fitted paper lens.  
True lens lane: **`/home/ruffianl/jacobian-lens`** (`jlens` — Verbalizable Representations / Global Workspace companion).  
Bridge writeup: `research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md`.

## What’s not done

1. **Live hook:** push phase-tagged keys from `measure_jacobian_step` / phase edges into a session `MultiKeyAddress`.  
2. **jlens integration:** fitted `J_l @ h` → unembed top tokens → text-bridge / multi-key schema (same address family, different measure).  
3. **SplatRAG cold pick** from PickQuery (text bridge + host embed).  
4. **Centroid** representative per cluster (currently first key by step).  
5. Persist keys to disk / museum / packet store.  
6. Real fingerprint stability: hydro FD *and/or* jlens transport.

## Next (one step)

Either finish hydro phase-edge **proxy** keys, or a jlens apply smoke whose top-token readout feeds the same multi-key / text-bridge shape — **label which** in the probe.
