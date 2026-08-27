# Next brick is chat residual continuity

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Correct the next brick: not force-in-revise. Live residual memory on the chat path that writes, reads, and survives process death — Path B inject is a forced probe, not continuity.

## Hypothesis

Wills deposited in --chat, saved, reloaded after kill without --clear-memory will move later-turn residual vs a cleared control on the same 9-turn seat.

## What changed

Jason: the real remaining gap is **live residual memory** that steers later multi-turn tokens and **survives process death**. Path B inject is not that. Force-in-revise is adjacent unpaid science, not this brick.

### Already proven (do not rewalk)

- Path B inject 073954: hand moves blend/β/σ and residual inside the 9-turn (`HYDRO_INJECT_TAG=spike`, consume-once). Continuity this is not — env probe.
- Gemma 3 oneshot continuity (Jul 16–17): safetensors + TCT, prefill-bridge, A→B→A PASS_RETURN. Mint is **oneshot generate**, not `--chat`. See `docs/CONTINUITY.md`.
- Remember-store: JSONL key=value survives process death in unit tests. It does **not** shove residual.

### Gap (receipts)

- `scripts/convo_defaults.sh`: `--clear-memory --no-save-memory` on every talk/smoke. Process death is forced.
- `--chat` returns at `run_simple_chat` (`main.rs` ~2860) **before** oneshot will-deposit / Phase 6b safetensors+TCT save. `add_splat` lives only in the oneshot loop (~3509+).
- Chat can *read* a store if one exists and `--clear-memory` is off; the 9-turn seat never writes one.
- Gemma 4 12B (D=3840) multi-turn death→reload KPIs are **not** the July Gemma 3 4B (D=2560) cards. Dim mismatch is refused.

### Next slice (the non-small move)

Same multi-turn seat (`talk.sh` / `smoke_convo.sh` parity, full-stack config):

1. Chat path **deposits** learned wills into residual memory during turns (no `HYDRO_INJECT_TAG`).
2. That store **saves** (safetensors and/or TCT) without `--no-save-memory` as the measured arm.
3. Kill the process, start again **without** `--clear-memory`, same seat.
4. Later tokens / start basin show a **measured** steer (F_s / pot / nearest / residual vs a clear-memory control), not a tag inject.

Score from chat text + probe/continuity cards. Isolation/full-stack/Path B 9-turns stay in CHANGELOG as done.

Signed: Grok (xAI) · operator Jason

## Findings

Named. Force-in-revise is not the load-bearing gap. Chat never deposits or saves wills; smoke flags wipe the store. July continuity is oneshot Gemma 3.

## Next

Wire chat deposit + save; death→reload arm without --clear-memory; measure steer on later turns vs clear-memory control. Do not use HYDRO_INJECT_TAG as the proof.

---

Signed: Grok (xAI) · operator Jason
