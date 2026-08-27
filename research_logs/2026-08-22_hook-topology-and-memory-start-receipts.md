# Hook topology and memory-start receipts

> Date: 2026-08-22
> Agent: Codex
> Repo: hydrodynamic-swarm-3surface

## Context

Expose the hook effects and persistence values already computed on each token, and freeze both splat memory and the model-emitted remember store in scaler-panel arms.

## Hypothesis

The next matched pilot will distinguish layer-hook pressure and monitor topology from scaler gain, while preventing cross-arm remember-store contamination.

## What changed

- Receipt v2 adds config/prompt hashes, seed, max tokens, chat-template id,
  memory-start snapshots, and the resolved transformer-hook profile.
- Every chat token now records the hook applications and mean/max hook delta
  from the forward pass that produced its logits.
- The internal monitor now labels H0/H1 as barcode counts and prints H1 total
  and maximum persistence.
- A WIPE_STORE eval now supplies a fresh isolated remember store per arm,
  archives its final content, and continues to wipe splat memory.
- The eval wrapper archives partial artifacts and the exact exit code on model
  failure, then removes its isolated temporary remember store.

## Findings

- Source inspection shows the 12B GGUF has 48 blocks. Current hook depth
  0.6–1.0 resolves to post-MLP layers 28–47, or 20 applications per decode.
- `H0=32/H1=465` was not an entropy value: with 32 points, 465 is the cycle-rank
  count of the complete edge set (`C(32,2)-32+1`). Persistence, not this raw
  count, feeds the loop-pressure calculation.
- `<lock>` is a real stop: a complete emitted lock sets `lock_stop`, and the next
  decode iteration exits. It cannot stop turns that never emit a complete lock.
- Gemini's August 2 investigation identified a historical chat-template bug:
  empty thought channels were attached to every prior model turn, pushing Gemma
  out of distribution and producing counting/channel loops. That exact bug is
  fixed here: historical model turns are plain canonical model turns and the
  empty thought channel appears only in the current generation prefix.
- Current history deliberately retains Internal monitor lines and model-emitted
  tags. Those additions are still outside the vanilla Gemma conversation
  distribution and are a separate context-interference candidate.
- The 512-token pilot showed failures before monitor injection and several
  hard-cap/cycle-clamp endings, so lock, monitor, hook placement, and scaler are
  distinct mechanisms.
- Receipt-v2 run `20260822_083116` completed all 14 turns with exit code 0. All
  3,900 token records linked to the receipt; 3,886 decode tokens reported exactly
  20 hook applications. Mean per-layer hook delta was 0.011529 and the maximum
  was 0.022622.
- No turn reached the 1024 ceiling (largest: 737 tokens). Five turns ended on
  short-cycle clamps, one on the wait-loop clamp, and the short exact-answer
  turns generally ended on model-emitted lock.
- Prompt 13 reached its per-item review but collapsed in runtime observation
  section (a), before sections (b)–(d) and before a SCORE. The missing SCORE is
  therefore not a max-token truncation.
- The monitor fired on ordinary common symbols such as `the` and `Gamma` at
  loop pressure around 0.43. H1 persistence was nonzero, but the named word was
  not itself proof of a lexical closed cycle; monitor sensitivity/context is a
  separate ablation target.
- In interrupted run `20260822_100540`, a first-window warning on `the` landed
  between correct Day 1 state and the next state; the model then emitted
  `1+3=10`, self-corrected locally, and lost its day cursor. On the jug prompt,
  three valid `\\rightarrow` separators satisfied the named-loop shortcut at
  loop pressure 0.15, leading to repeated focus/spike restarts and illegal
  states. A warning also spliced into an unfinished photon `<remember>` body.
- These boundaries show why `<lock>` does not solve the date/day problem: lock
  is a reliable completed-answer stop, whereas focus/spike are mid-answer
  physics/context interventions. They can be false-positive and are not factual
  state correction.

## Next

- The 083116 run is diagnostic rather than a final matched cell because it
  exposed and preceded the final-splat-cap and remember-cursor fixes.
- Use the per-token hook telemetry to decide whether a dedicated hook-off or
  layer-band ablation is warranted; do not infer layer guilt from the band alone.
- Freeze the TDA mouth monitor off for the scaler factorial while retaining
  model-emitted hands. Compare the clean first arm to `100540` as a monitor
  diagnostic, not as evidence that a scaler rule caused an answer.
