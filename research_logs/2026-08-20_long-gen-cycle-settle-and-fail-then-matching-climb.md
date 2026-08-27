# Long-gen cycle settle and fail-then-matching climb

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Cycle-lock settle stops 256-token unmatched esese soup on ordinary persist. Fail-then-matching climb quotes the minted lumina definition with warmer trail/basin after a failed zed-anchor-9 turn. apply_emitted_control writes residual on emit; live Gemma did not emit a tag. No inject.

## Hypothesis

Stopping short-cycle lock will keep 256-class unmatched generation from running to esese soup, and after a failed unmatched turn a matching prompt will return the minted scar with warmer trail telemetry.

## What changed

Ordinary three_surface `--chat` (persist without KEEP). Cycle-lock settle stops 256-token unmatched `esese` soup. Fail-then-matching climb is geometric and quoted. Decode-loop `apply_emitted_control` writes a residual splat when a tag appears in generated text; later query reads it. No-tag writes nothing. No `HYDRO_INJECT_TAG`. Isolation wipe unchanged.

GPU-free `{SCRATCH}/emit_residual_unit.txt`: `emit_control_writes_residual_later_query_reads` (spike writes pot at site, far is cold, no-tag stays 0); `trailing_short_cycle_lock_catches_esese_not_prose`.

Live, `keep_memory=` empty, BOS `first_id=2`, inject empty:

| arm | stamp | `gemma4>` |
|-----|-------|-----------|
| long 256 | `133051` | Two residual-stream paragraphs; `[CHAT SETTLE cycle] step=190` stops the lock. Named TheThe/theed/Wait/math-thrash gone. Short cycle tail before the clamp. |
| climb | `133222` | t1 mint definition. t2 fail “I do not have a predefined definition for zed-anchor-9…” STEER pot=0.046 nearest=147 \|F_s\|=0.21 no TRAIL. t3 climb “The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens.” `[CHAT TRAIL load] n=19` own=yes pot=0.459 nearest=66.7 STEER \|F_s\|=2.85 keep. |

`[CHAT EMIT SCAR]` absent on both live arms — model did not emit a control tag. Residual write/read of emit is the unit test, not inject.

## Findings

Long unmatched 256 (`133051`): `[CHAT SETTLE cycle]` at step 190 stops the unbounded `esese` run that previously went to the token cap. Named TheThe/theed/Wait/math-thrash gone. A short cycle tail remains in the stream before the clamp. Residual write/read live (wills + trail mint). keep_memory empty. inject empty. BOS `first_id=2`.

Climb (`133222`): after mint, failed zed-anchor-9 is dictionary and geometrically cold (pot=0.046 nearest=147 \|F_s\|=0.21, no TRAIL load). Later matching lumina quotes the minted definition and is warmer (`[CHAT TRAIL load] n=19` own=yes pot=0.459 nearest=66.7 STEER \|F_s\|=2.85 keep). Distinct from the failed turn.

Live emit: `[CHAT EMIT SCAR]` absent on both arms. GPU-free `apply_emitted_control` writes a splat on `<spike>` that later `query_potential` reads; no-tag writes nothing. That is the criterion-3 residual bar; inject was not used.

## Next

Do not rewalk paid ordinary-seat 3-arm/sequence or this long/climb pair as unpaid. Isolation wipe stays. Spontaneous tag emit in Gemma 4 chat is still unobserved.

---

Signed: Grok (xAI) · operator Jason
