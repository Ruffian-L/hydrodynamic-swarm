# Final splat cap and streamed remember payload

> Date: 2026-08-22
> Agent: Codex
> Repo: hydrodynamic-swarm-3surface

## Context

Fix two receipt-exposed runtime bugs: topic-matched bridge pulls bypassed residual.splat_max, and streamed remember blocks fired before their payload closed and therefore never persisted.

## Hypothesis

Final-force recapping and deferred remember-block parsing will make configured ceilings truthful and preserve model-emitted key/value memory without changing the locked scaler formulas.

## What changed

- Reapplied `residual.splat_max` after topic-matched bridge coupling so the
  telemetry and summed force cannot exceed the configured final ceiling.
- Replaced the integer streaming-tag cursor with a hit-aware cursor. A simple
  `<remember>` still fires immediately; when the same slot later closes as
  `<remember>key=value</remember>`, its payload persists exactly once without a
  second residual-splat deposit.
- Added `grad_mag`, `splat_mag`, `goal_mag`, `ocean_mag`, ramp, and delta-H to
  every chat token record.
- Renamed the legacy `CpuBackend` telemetry label to `Candle Tensor`; its math
  executes on the tensor device and was already CUDA during generation.

## Findings

- The 083116 receipt-v2 pilot observed `splat_mag=10.9093` while the receipt's
  `residual_splat_max` was 2.055443. Source order showed the topic bridge was
  added after the cap.
- The same pilot fired remember hands but archived a zero-byte remember store.
  Incremental parsing first counted the payload-less opening tag, then skipped
  the completed payload because it occupied the same ordinal slot.
- Targeted tests now pass for final topic-pull capping, warm-basin behavior,
  simple tag compatibility, cursor upgrade, and persistence to disk.
- Interrupted receipt run `20260822_100540` archived five completed payloads in
  a 466-byte isolated remember store. This confirms persistence now survives
  the live streaming path; the run ended with exit code 141 after the host/tool
  stream restarted and is not a completed panel cell.

## Next

- Keep `083116` and interrupted `100540` as diagnostics. Rerun the first matched
  arm under receipt v3 with the TDA mouth monitor frozen off; all later panel
  cells must use that same binary hash and intervention state.
