# Official scaler panel monitor isolation

> Date: 2026-08-22
> Agent: Codex (isolation patch); Grok (xAI) (first-arm run)
> Repo: hydrodynamic-swarm-3surface

## Context

Freeze TDA mouth injection off across scaler arms while preserving model-emitted control hands and lock; receipt that intervention state and defer any enabled warning until streamed payload tags close.

## Hypothesis

Removing false-positive monitor prose from the model mouth will reduce sequence restarts without changing the piecewise scaler, residual coefficients, seed, temperature, or hook band.

## What changed

- `evals/official-10/eval.env` freezes `HYDRO_TDA_MONITOR=0`; the panel and
  smoke headers print that state. Model-emitted control tags and `<lock>` stay
  enabled through the existing Gemma 4 system control channel.
- Scaler receipt schema v3 records the control-hand gate, TDA monitor gate,
  window, stride, and cooldown alongside prompt/config/memory/hook state.
- If the monitor is enabled elsewhere, warnings wait for streamed remember or
  lock payloads to close. Stopwords/structural arrows cannot be named loops, and
  an H1-only warning no longer names an arbitrary frequent word.
- Added regression tests for payload deferral, structural-name filtering, and
  H1 warning wording.

## Findings

- `20260822_100540` is a recovered partial diagnostic (exit 141), not a panel
  cell. Before interruption it reproduced correct local arithmetic with broken
  ordinal state, monitor-triggered list restarts, and `\\rightarrow` false
  positives at loop pressure as low as 0.15.
- That run returned and later recalled `lumina-basin-7` correctly, and its
  isolated remember store archived five payloads. Neither observation can be
  attributed to the scaler because the final-splat and parser bugs changed
  relative to the earlier receipt-v2 pilot.
- The missing final SCORE in the completed 083116 pilot was not caused by the
  1024-token ceiling: no turn reached it; settle short-cycle/wait-loop guards
  ended the response before the requested sections.
- Grok bypassed the Codex quota wall and ran the frozen first arm as
  `20260822_110203` (exit 0, ~83 min). Receipt v3:
  `scaler-piecewise-12-k0.500-43fec98c5102-62c018a80b56-1787396611498`, binary
  `62c018a80b56…`, `tda_monitor_enabled=false`, control tags on, seed 424242,
  max_tokens 1024, empty splat, empty remember start. Hydro IFEval tags was
  paused at 4/541 to free the 12B seat; that partial is archived under
  `runs/2026-08-22_pathb_ifeval_tags2_hydro/paused_for_scaler_20260822_110112/`
  with live binary hash `cf1acdb8…` (eval-jsonl truncates on restart).
- Mouth `[Internal monitor]` count: 0. Probe `internal_monitor` events: 0.
  Hands: focus 9 / remember 11 / lock 8. No spike/reset. No turn hit the
  1024-token ceiling (max 848 on the snail).
- Monitor-off did **not** save sequence walks. P1 snail lost the day cursor and
  ended `settle_phrase_repeat` at step 848 without reaching day 8. P8 jugs
  wandered through illegal states (`0-gallon`, `(2,2)`, `(3,5)`) then wrote the
  legal path to `(3,4)` and locked. Hypothesis on restarts: failed.
- Exact nonce: P2 first emitted `lumina-basin-1` then corrected to 7. P6 and P10
  recalled the first wrong emit and stopped at “Wait, I must use the exact”
  (24 tokens each). P7 aurora almost exact, then glued `earlier_scars` into the
  remember payload. Remember store last-write-wins on `codeword`: final file
  is 4 payloads / 366 bytes
  (`protocol`, `step_count=9/11`, `review_sequence`, garbled aurora). The
  corrected lumina payload was overwritten.
- P11 ARC `settle_phrase_repeat` on “the original-list is reversed”. P12
  pre-key review was `settle_channel` at step 3 (`<|channel>thought`) — no
  CONFIDENT/UNCERTAIN/CHANGE and no SCORE on the notes turn.
- P13 regrade **did** reach SCORE. Her call on the work: **SCORE 6.5/10**.
  MATCH: zed, photon, liar, jugs (she scored the recovered `(3,4)` as MATCH),
  friendship. ADJUST: aurora. REWRITE: snail, lumina mint, lumina recall,
  3-line return, ARC. She named `<focus>` as failing to stop mid-sentence
  drift. Do not attribute SCORE or any answer to the piecewise size scaler:
  this is one (rule × k) cell with hooks, governor, sampling, and base model
  still in the seat.
- Niodoo IFEval was left running on the same GPU during this arm.
- Phone listen pack is on Google Drive as `Hydro-listen-2026-08-22-110203`
  (link `https://drive.google.com/open?id=1WtclPuQ5ItZY1Kd0P0_4ZZMkLSXJ8wvU`).
  Listen `ALL_IN_ONE.txt`. Probe is `06_probe.jsonl` (2.8MB), not for Speak.

## Next

- Do not launch further scaler-panel cells as unpaid. Monitor isolation for
  sequence walks is answered: off does not fix them.
- One intervention at a time after this cell, not a gain ladder. Remaining
  candidates: `settle_channel` eating the notes turn, first-wrong-token
  contamination on lumina-basin-1, remember `codeword` overwrite, hook band,
  sampling, base model.
- Hydro IFEval tags resume is a restart from item 1 (jsonl truncates) using
  archived `cf1acdb8…`, not this panel binary, and only after this 12B seat
  is free.
