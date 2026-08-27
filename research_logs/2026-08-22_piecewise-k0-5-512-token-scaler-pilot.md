# Piecewise k0.5 512-token scaler pilot

> Date: 2026-08-22
> Agent: Codex
> Repo: hydrodynamic-swarm-3surface

## Context

Record the first receipt-bearing piecewise k=0.5 run as a 512-token pilot; the later 1024-token panel must rerun this cell because max_tokens is part of the frozen state.

## Hypothesis

The pilot can validate receipt linkage and expose failure modes, but cannot support scaler causality or comparison with 1024-token arms.

## What changed

- Ran `piecewise × k=0.5` on the locked Official 10 pack with seed 424242 and an
  externally wiped splat store.
- Archived mouth, operator log, token probe, turns, and scaler receipt under
  `logs/evals/official-10/20260822_073607/`.
- Raised the future panel token ceiling from 512 to 1024 after this run; this
  intentionally reclassifies the run as a pilot.

## Findings

- Receipt id:
  `scaler-piecewise-12-k0.500-43fec98c5102-c3728de25c35-1787384266736`.
- Effective residual gain was 0.51386076. Final cap/field/splat/goal were
  0.513861 / 0.010277 / 0.015416 / 0.004111. Temperature remained 0.70;
  ramp, logit physics, and governor remained frozen.
- All 3,100 token records carried the receipt id. Four turns reached the
  512-token ceiling; three later turns stopped on explicit cycle clamps.
- The snail answer failed arithmetic and repeated days; lumina-basin-7 became
  lumina-basin-1; later text showed short-cycle and phrase-repeat collapse.
- The half-gain pilot therefore does not trivially eliminate the failures. It
  does not establish that scaler gain caused them.
- The run did not isolate the persistent model-emitted remember store, which is
  another reason not to include it in the matched panel.

## Next

- Rerun the cell at 1024 with receipt v2, fresh splat and remember stores, and
  hook/persistence telemetry.
