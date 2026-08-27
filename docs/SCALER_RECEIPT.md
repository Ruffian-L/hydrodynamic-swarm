# Hydro scaler source of truth

Hydro now keeps two vocabularies separate:

1. A formula predicts native process values (`size_scale`, archetype multiplier,
   force intensity, σ, θ, β, loop repulsion, and formula-native temperature).
2. The `hydro-residual-profile-relative/v1` adapter multiplies the existing
   TOML residual force family by `force_intensity × manual_gain`.

The native σ/θ/β values are not aliases for `residual.cap`, the observed
gravity force, ghost counts, or goal/repulsion receipts.

## Resolution order

- P: `--model-params` → `algo.params_b` → first standalone `<number>B` in the
  model filename.
- Archetype: `--model-type` → `algo.model_type` → filename inference.
- Rule: `--size-rule` → `HYDRO_SIZE_RULE` → `algo.size_rule`.
- Manual gain: `--scaler-gain` → `HYDRO_SCALER_GAIN` → `algo.gain`.
- Application gate: `--apply-scaler` / `--no-apply-scaler` →
  `HYDRO_SCALER_APPLY` → `algo.apply`.
- Startup `--set name=value` values are applied after the adapter and are
  listed individually, including rejected names.

## Formula identities

- `legacy`: 3B-anchored square root, legacy archetype multipliers, tight
  clamps. This is the old Algo_WIPjuly transform.
- `8b-sqrt`: 8B-anchored square root, July run-card archetype multipliers,
  wider clamps, and a formula-native temperature coupled to β.
- `piecewise`: square root through 8B, then
  `1 + 0.35 ln(P/8)`, legacy archetype multipliers, wider current clamps, and
  temperature decoupled.
- `off`: size/archetype multiplier 1.0. The manual gain still applies, making
  this the matched manual-gain control rather than a duplicate row.

The Hydro adapter freezes force-ramp tokens/start, sampling temperature,
logit field/splat forces, and governor coefficients. Historical temperature
predictions remain in the receipt for cross-checking but do not enter this
matched residual-force panel.

## Gemma 4 12B instruct cross-check

With base temperature 0.70:

| Rule | Size scale | Archetype | Force intensity | σ | θ | β | Repulsion | Predicted T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 2.000000 | 0.90 | 1.800000 | 0.200000 | 3.000000 | 150.000 | 3.000000 | 0.800000 |
| 8b-sqrt | 1.224745 | 1.00 | 1.224745 | 0.183712 | 0.673610 | 122.474 | 0.734847 | 0.571548 |
| piecewise | 1.141913 | 0.90 | 1.027722 | 0.154158 | 0.565247 | 114.191 | 0.616633 | 0.700000 |
| off | 1.000000 | 1.00 | 1.000000 | 0.150000 | 0.550000 | 100.000 | 0.600000 | 0.700000 |

The old boot line `scale 2.000 · intensity 1.800 · σ=0.200 · θ=3.00 ·
β=150.0` is therefore exactly the legacy row. It was previously only a loader
readout; it did not mutate the engine.

For `piecewise`, k=1.0, and the current `config.three_surface.toml` base, the
effective residual gain is 1.0277215. The adapter produces:

| Live coefficient | TOML base | Applied |
|---|---:|---:|
| residual.cap | 1.000000 | 1.027722 |
| residual.field | 0.020000 | 0.020554 |
| residual.field_max | 2.000000 | 2.055443 |
| residual.splat | 0.030000 | 0.030832 |
| residual.splat_max | 4.000000 | 4.110886 |
| residual.goal | 0.008000 | 0.008222 |
| residual.goal_max | 2.000000 | 2.055443 |
| force ramp | 0.03 / 48 tokens | 0.03 / 48 tokens |
| sampling temperature | 0.700000 | 0.700000 |
| logit field / splat | 0.020 / 0.004 | 0.020 / 0.004 |
| governor brake / viscosity | 1.5 / 3.0 | 1.5 / 3.0 |

## Receipt and token linkage

Before request 1, Hydro writes one create-only JSON receipt. It contains the
model and executable SHA-256 hashes, resolved P and archetype, selected rule,
all four formula predictions, base profile, every startup override, and final
live coefficients. Receipt v3 also freezes the config and prompt hashes, sample
seed, token ceiling, chat-template identity, splat/remember memory-start
snapshots, the resolved transformer-hook site and layer range, and whether the
TDA mouth monitor and model-emitted control hands were enabled. The matched
scaler panel freezes TDA mouth injection off while leaving `<lock>` and the
other model-emitted hands available.
`smoke_convo.sh` gives each run a unique receipt path.

Every collapse-probe `tok` record carries `scaler_receipt_id`; the first turn
also embeds the immutable object as a `scaler_receipt` event. One-shot JSONL
step records carry the same id and the config header embeds the object.
Runtime `/set` edits are trajectory events after request start and do not
rewrite the receipt.

Chat token records also report the hook applications and mean/max hook delta
from the forward pass that produced that token's logits. Step 0 is a prefill
sample and therefore has zero decode-hook applications. For the current Gemma
4 12B seat, `post_mlp`, depth 0.6–1.0 resolves to layers 28–47: 20 applications
per decoded token.

The same token record reports the post-cap force magnitudes that actually enter
the residual sum: field/gradient, splat, goal, and ocean, plus force ramp and
the final delta-H norm. Topic-matched bridge coupling is included before the
final splat ceiling, so `splat_mag` is bounded by `residual.splat_max`.

The internal-monitor `H0bars` and `H1bars` values are barcode counts, not an
entropy magnitude. In a 32-point complete Vietoris–Rips filtration, `H1bars`
is normally 465 (`496 edges - 32 vertices + 1`). `H1sum` and `H1max` carry the
persistence magnitudes used by the topology gate.

When the monitor is enabled outside the matched panel, it now defers a pending
warning until a streamed remember/lock payload has closed. Structural arrows
and common stopwords cannot be named as lexical loops, and an H1-only warning
does not borrow the most frequent merely “leaning” word as its loop name.

## Meaning of live force telemetry

- `residual.cap` is a per-dimension clamp on the summed residual force.
- `residual.field`, `residual.splat`, and `residual.goal` are scales for three
  distinct force components; their `_max` values are vector-magnitude ceilings.
- The observed `F_g`, `F_s`, and `F_a` values are post-geometry force magnitudes,
  not the formula's σ, θ, or repulsion.
- Niodoo gravity maxima, ghost-hit counts, and goal/repulsion token receipts are
  downstream measurements with different units. A zero goal/repulsion receipt
  cannot be replaced by a nonzero configured scale.

No downstream causal claim is supported until a frozen-state
`size_rule × manual_gain` panel carries these receipts.
