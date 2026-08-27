# 2026-07-30 — SplatRAG pick → hydro scar bridge

**Status:** thin pipe landed (not a coupling claim)  
**Lead:** Jason Van Pham (architecture / synthesis)  
**Implement:** Grok (xAI) on hydro side; Claude (Opus) picker + bridge spec on SplatRAG  

## What shipped

CLI on `hydrodynamic-swarm`:

```text
--import-picks PATH.json   # deposit each pick as a residual scar
--picks-max-gain 0.35      # local ceiling (default 0.35)
--picks-dry-run            # log ‖μ‖ and α; deposit nothing
```

Module: `src/picks.rs`  
Spec origin: SplatRAG `docs/BRIDGE_SPLATRAG_PICK.md`

## Rules enforced

1. **Embed text, never `semantics_64`.** Residual = last-token hidden of the live gen model after prefill of `pick.text`. Provenance gate: `source_dim == 64` and embedder starts with `Qwen3-Embedding` (telemetry space only).
2. **α resolution:** use recorded `gain` if non-zero, else picker `suggested_gain`. Clamp to `--picks-max-gain`.
3. **mass < 0 → repel** (negative α). Gain and mass stay separate knobs.
4. **Deposit path:** `SplatMemory::deposit_prefill_bridge` (same continuity scar family as post-gen bridges).
5. **Dry-run first:** report `‖μ‖`; flag if outside scar family (~O(100)).

## Not claimed

- Memory coupling works (Prediction 3 / three-arm harness still open).
- 64→2560 codec (parked RAVE).
- SplatRAG `steer --gain` α band (self-axis degeneracy — next lane).

## Produce a pick (SplatRAG)

```bash
splatrag pick "Explain the Physics of Friendship in one short paragraph." \
  --limit 3 --out data/pick.json
```

## Consume (hydro)

```bash
./target/release/hydrodynamic-swarm \
  --config configs/gemma4/config.gemma4_near_vanilla.toml \
  --import-picks /path/to/pick.json \
  --picks-dry-run \
  --chat
# then drop --picks-dry-run once ‖μ‖ looks like existing scars
```

## Tests

`cargo test --bin hydrodynamic-swarm picks` — 8 unit tests (provenance, α policy, mass repel, JSON parse).
