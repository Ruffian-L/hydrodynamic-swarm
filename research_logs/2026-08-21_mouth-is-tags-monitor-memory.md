# Mouth is tags, Internal monitor, memory inject

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: look at Niodoo vs hydro; hydro is noise; cannot follow. The only things in the chat should be tags, Internal monitor, and memory injects. If operator telemetry needs to pop up, that is a TUI the model does not see. CONTROL_RECEIPT is worthless telemetry.

Niodoo `official10.txt` Prompt 1 is model text + `[Internal monitor: …]` + `<focus>`. Hydro 103524 interleaved `[CHAT PREFILL see]`, `[prefill]`, `[CHAT STEER]`, `[CONTROL_RECEIPT]`, `[CHAT WILL]`, `[CHAT BASIN mint]` into the snail walk.

## Hypothesis

Stdout = mouth (tokens, tags, Internal monitor, any memory text inject). Stderr = ops. `tail -f latest` becomes followable like Niodoo. Hands still fire. KV still gets monitor forwards and tags in history. The model never sees CONTROL_RECEIPT.

## What changed

- Deleted `[CONTROL_RECEIPT]` print. `hand_fired` stays in collapse_probe jsonl.
- Every `[CHAT …]`, packing, prefill banner, tag-inject, chat-mode help: `eprintln!`.
- `hud_quiet_println!` now `eprintln!` so remember/endocrine diagnostics are not the mouth.
- `smoke_convo.sh`: stdout → `smoke_convo_STAMP.txt` (mouth). stderr → `smoke_convo_STAMP.ops.txt`. No `2>&1`.
- Internal monitor still `on_token` (stdout) and still KV-forwarded.
- Tags she emits still stdout (they are tokens).
- `--tui` / HUD already pops physics. Did not build a second TUI.
- Did not move α.

## Findings

103524 Official 10 was the noisy seat (model does not see those prints; Jason does). This mutation is so Jason can follow. Live stamp after rebuild is the check.

## Next

Restart Official 10 on the quiet mouth if Jason is watching. `tail -f logs/smoke_convo_latest.txt` = mouth. `tail -f logs/smoke_convo_latest.ops.txt` = ops.

Signed: Grok (xAI) · operator Jason
