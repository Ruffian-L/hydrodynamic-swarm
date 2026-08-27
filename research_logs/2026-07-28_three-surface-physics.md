# Three-Surface Physics

> Commit: uncommitted `physics/three-surface` worktree (base `698164cd`) | Date: 2026-07-28
>
> **Author:** Codex (OpenAI) + operator

---

## Commit Summary

- **Message**: implement residual + logit + transformer-stack physics with live controls
- **Scope**: logit chain, layer hooks, model dispatch, chat/one-shot integration,
  strict configs, telemetry, gates, tests, and CUDA smoke validation
- **Dependency source**: current crates.io registry; the legacy `vendor/` snapshot
  is not used by Cargo

## What Changed

### Logit surface (`src/logit_physics.rs`)

- Replaced the inline field-logit term with an ordered `LogitChain`.
- Added `FieldBias`, token-targeted `SplatBias`, and an entropy/inertia
  `Governor` (centrifugal brake, viscosity, and minority report).
- All token geometry comes from the loaded model's own `tok_embeddings`.
  There is no side embedding model.
- Splat vocab bias uses field index = token ID. When ranked memory is active,
  it consumes the same scar winners as residual steering.
- Every gain is live-adjustable through `/set`; individual zero values are
  genuine off switches.

### Forward surface (`src/hooks.rs`, model forks)

- Added explicit `LayerHook` dispatch to Llama, Gemma 3, and Gemma 4.
- `NiodooLayerHook` reuses the current token's residual steering direction,
  changes only the final sequence position, and scales each layer delta to a
  fraction of that site's activation norm.
- Default band is `post_mlp`, depth fraction `0.5..1.0`, norm fraction
  `0.0005`. Disabled hooks bypass hook construction and synchronization.
- Optional JSONL trace mode records layer/site activation and delta norms.

### Runtime and controls (`src/main.rs`, `src/niodoo.rs`)

- One-shot and multi-turn `--chat` now run residual, logit, and forward
  surfaces together.
- Chat commands:
  - `/tui` opens an alternate-screen keyboard slider panel. Up/down selects,
    left/right adjusts, Shift or Ctrl moves faster, `r` resets, and
    Enter/Esc/`q` returns to chat.
  - `/phys` or `/sliders` shows residual, logit, hook, and sampling controls.
  - `/set <name> <value>` changes them without a rebuild or model restart.
- Prefill stays model-native and establishes the turn goal. Decode forwards
  receive the previous steering delta as the hook direction.
- Model control tags modulate a turn; operator `/set` values remain persistent.

### Config and telemetry

- Added strict `[logit_physics]` and `[hooks]` sections.
- Added `deny_unknown_fields` to every live config struct. Obsolete sections
  now fail loudly rather than silently activating defaults.
- Converted all force-off templates into real three-surface-off baselines.
- Added supported moving-arm configs under `configs/gates/`; historical
  `config.g*.toml` receipts remain intentionally non-executable.
- Per-step JSONL and TermSplat weather now report each logit magnitude,
  governor velocity/viscosity, and hook application/delta statistics.
- Session config entries record the post-CLI live logit and hook settings.

## Physics Impact

- **Residual**: the existing Niodoo force sum remains the source geometry.
- **Logits**: field direction, scar valence, and collapse pressure now act
  additively before repetition and n-gram guards.
- **Forward pass**: the same residual direction acts inside a layer band with
  dimensionless local-norm scaling, avoiding residual/mid-stack scale mismatch.
- **Memory**: ranked residual and logit scar paths share one picker result.

## Key Findings

1. **Three-surface CUDA path is operational**
   - Evidence:
     `logs/2026-07-29_02-55-01_gemma3_v3-forcecap3_T0_88_s12_a1_d25.jsonl`
     generated `Ready` then EOS on the local Gemma 3 4B GGUF with all surfaces
     enabled and `--no-save-memory`.
   - The step after prefill reports 17 hook applications and active field plus
     governor logit engines.

2. **Scale-free hook calibration is exact**
   - Evidence: `logs/2026-07-28_three_surface_hook_trace.jsonl`.
   - Gemma 3 4B, 34 blocks, `post_mlp`, layers 17 through 33: 17 applications.
   - Activation norm: min `23908.326`, mean `32944.096`, max `46802.280`.
   - Delta norm: min `11.954`, mean `16.472`, max `23.401`.
   - Measured delta/activation ratio stays at `0.0005000000` across the band.

3. **Live chat controls work on the loaded model**
   - `/set hook.fraction 0.001` and `/set field.alpha 0.2` updated the rendered
     sliders immediately; the same session then generated `Ready`
     (`private/chats/chat_1785293722_gemma3_chat.txt`).
   - `/tui` uses those same setters, applying each keyboard adjustment to the
     loaded session immediately and recording final changes in the private
     transcript.
   - Real CUDA/PTY smoke: moved `residual.cap` from `3.1` to `3.3` and
     `residual.dt` from `0.035` to `0.036`, closed the panel, then `/phys`
     confirmed both values persisted without reloading Gemma 3
     (`private/chats/chat_1785295208_gemma3_chat.txt`).

4. **The Gemma 4 31B target runs the same live surfaces**
   - `scripts/chat_gemma4.sh` now resolves only worktree-local target paths,
     selects the matching Gemma 4 tokenizer, and starts the three-surface arm.
   - Real CUDA/PTY smoke loaded the 60-layer, 5376-wide
     `bart_google_gemma-4-31B-it-Q4_K_M.gguf`, opened all 22 `/tui` controls,
     persisted a live slider edit, and generated `Ready`.
   - Gemma 4 automatic discovery now outranks Gemma 3, and its EOS set matches
     the local generation asset (`1`, `106`, `50`).
   - Automatic discovery (no `--model` or `--tokenizer`) selected the correct
     GGUF and `gemma4_assets/tokenizer.json`. Evidence:
     `logs/2026-07-29_03-29-27_gemma4_v3-forcecap3_T0_88_s12_a1_d25.jsonl`.
     Its decode step reports 30 `post_mlp` hook applications across the
     configured second-half band, with mean delta `0.09895` and max `0.16737`.
   - Rebuilt Gemma 4 PTY smoke switched directly to Logit and Hook with keys
     `2` and `3`; adjusting `field.alpha` to `0.160` applied live before exit.

5. **Strict config fixed an ablation-integrity bug**
   - Old force-off/gate files used removed sections and
     `physics.field_logit_bias_alpha`; Serde silently ignored them, leaving
     physics defaults active.
   - Unknown sections and legacy keys now fail tests and runtime loading.
   - Evidence:
     `logs/2026-07-29_02-59-14_gemma3_v3-forcecap0_T0_s12_a1_d25.jsonl`
     reports exact `steering_delta = 0`, zero residual force magnitudes, zero
     logit engines fired, and zero hook applications.

6. **CPU regression suite is clean for available fixtures**
   - `114 passed; 0 failed; 2 filtered out`.
   - The two filtered Concourse tests require separate absent embedding-model
     files and are outside this model-native three-surface path.

## Experiment Progression

- **Before**: residual steering once per token, one inline field-logit bias,
  no forward hooks, and chat discarded the physics engine.
- **After**: all three surfaces are on by default, individually zero-able,
  live-adjustable, and measured in one-shot and chat paths.
- **Next**: operator coherence review in the chat REPL, then continuity and
  longer-run collapse checks using an intentionally minted memory store.

## Dead Ends and Limits

- Ubuntu 24.04 initially blocked sandboxed `bwrap` through AppArmor
  `unprivileged_userns`; a scoped `/etc/apparmor.d/bwrap` profile restored the
  normal tool path.
- A release built with only `with-candle` correctly lacked CUDA. Rebuilding
  with default features produced the working CUDA binary.
- The first greedy hook-trace prompt emitted EOS before a decode hook could
  run. Repeating at the supported sampling settings produced the 17-entry
  trace above.
- Continuity scripts were not run: this worktree has no
  `data/splat_memory.tct` or safetensors store, and those scripts intentionally
  mutate/export the persistent store. Bridge retention, decay, pruning, and
  ranked-picker invariants passed their CPU tests.
- Long historical `config.g*.toml` sweeps were not reinterpreted because their
  removed knobs have no reliable one-to-one mapping. The supported moving arms
  replace them for current work.

## Open Questions

- [ ] Does operator-rated coherence improve over a longer live chat while the
      governor prevents late collapse?
- [ ] With a real scar store, does `SplatBias` contribute at useful magnitudes
      without overpowering the native distribution?
- [ ] Do continuity revisit and A→B→A return remain WARM with hooks enabled?
- [ ] Calibrate the same fractional band on Gemma 4 and Llama model files.

---

**Authorship**

- **Author:** Codex (OpenAI) + operator
- **Date:** 2026-07-28
- **Note:** Every failed run that is logged is a path someone else does not have to re-walk.
