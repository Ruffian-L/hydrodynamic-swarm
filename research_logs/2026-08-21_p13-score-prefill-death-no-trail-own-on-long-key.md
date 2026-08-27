# P13 SCORE died on 5263-token prefill; no trail-own on long key

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: stuck at Prompt13 `Model>` with no text. 111147 process is dead. Ops last line `[prefill turn=14 n=5263]`. Probe `turn_start` prompt_tokens=5263 prompt_chars=19297. No `CHAT STEER` for turn 14. No SCORE.

Prompt13 is the Official 10 key and quotes `lumina-basin-7`. Trail-own on that nonce is a known SCORE hijack. Prefill of the whole 13-turn history is why the mouth sits on `Model>` with silence (prefill happens after that print).

## Hypothesis

Disable trail-own when the user turn is longer than 800 characters. P2/P3/P6/P7 stay short and can still own. P13 generates freely and can emit SCORE.

## What changed

- `trail_own_len = 0` when `fp_src.chars() > 800`.
- Restart Official 10 after rebuild. Do not move α.

## Findings

111147: Prompt> through Prompt12 done. Prompt13 `Model>` empty. `evals/LATEST` still 090107 (wrapper never closed). Inotify raise did not apply (no sudo TTY).

## Next

Live SCORE on the new stamp. Follow with `scripts/follow_mouth.sh`.

Signed: Grok (xAI) · operator Jason
