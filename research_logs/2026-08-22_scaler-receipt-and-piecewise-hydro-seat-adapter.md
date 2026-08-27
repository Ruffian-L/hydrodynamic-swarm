# Scaler receipt and piecewise Hydro seat adapter

> Date: 2026-08-22
> Agent: Codex
> Repo: hydrodynamic-swarm-3surface

## Context

Replace the legacy-only loader imprint with an explicit selectable size transform and immutable equation-to-seat receipt before any matched model run.

## Hypothesis

A receipt-bearing piecewise 12B arm can separate the formula's residual-force gain from frozen temperature, ramp, logit, governor, prompt, seed, and memory state.

## What changed

- Added explicit `legacy`, `8b-sqrt`, `piecewise`, and `off` transforms without
  collapsing their anchors, archetype multipliers, clamps, or temperature laws.
- Added a profile-relative adapter that changes only Hydro residual cap, field,
  splat, goal, and their ceilings. Ramp, temperature, logit physics, and governor
  remain frozen.
- Added a create-only scaler receipt and linked every collapse-probe token to it.
- Added `scripts/hydro_scaler_panel.sh` for the locked Official 10 rule × gain
  panel, with seed and empty-memory setup held fixed.

## Findings

- At P=12 instruct, the legacy imprint is size 2.0 and intensity 1.8; current
  piecewise is size 1.141913 and force intensity 1.027722.
- The historical loader banner was a readout. The live residual coefficients
  were a separate vocabulary and previously had no immutable equation-to-token
  link.
- The first receipt-bearing run validated the link, but its 512-token ceiling
  makes it a pilot rather than a comparable cell in the 1024-token panel.

## Next

- Rerun piecewise k=0.5 with receipt v2 and the frozen 1024-token protocol.
- Complete the other rule × gain arms only with the same binary, prompt hash,
  seed, memory snapshots, and token ceiling.
