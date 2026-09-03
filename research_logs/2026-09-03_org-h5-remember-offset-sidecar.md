# ORG-H5 remember-offset sidecar dump (not the column)

> Date: 2026-09-03
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

ghost_team_groktodos first brick: why-vector mint at choice time. Design parked 2026-08-27 (`research_logs/2026-08-27_remember-geometry-column-offset-receipt.md`). MEM-10 said do not add a RememberLine geometry column until a sidecar dump ranks offsets.

Jason: start chunking the list. GPU 95% (QLoRA). No live ranking this turn.

## What changed

- `src/remember_geometry.rs` — ring of 16 decode steps, dual-site `S_res` / `S_logit`, 9 offsets (`t_star` … `maxnorm_5`). Closed `<remember>` appends C1 JSONL + `.f32` bins. `inject=false`. FNV-64 identity (no sha2 dep). C2 `shuffle_same_dim` / C3 `sign_flip` are helpers, not live rows.
- `NiodooEngine` rings via `push_remember_hidden` / `note_remember_pieces`. `fire_tag` Remember+payload dumps. `on_kv_drop` clears the ring and does not mint.
- `--chat` decode loop in `src/main.rs` pushes `surface_hidden` (S_res) and `raw_hidden` (S_logit) every tok, then notes pieces, then `apply_emitted_control`.
- Sidecar path: `{seat_remember}.offset_probe.jsonl`. `HYDRO_REMEMBER_SIDECAR=0` disables. Explicit path overrides.

Did not: RememberLine schema add, F_s inject, hang on `<spike>`, rank offsets, steal ratatui, live model run.

## Hypothesis

We think a dormant sidecar (C1) leaves the mouth equal to C0, and an offline cosine rank of matching vs unmatched vs shuffle will pick a site×offset that is a why, not a tape.

## Findings

CPU units only (GPU busy):

- 9 offsets × 2 sites = 18 rows on a closed remember with a full ring.
- Spike does not arm or dump. Incomplete `<remember>` does not dump.
- KV drop after mint does not add rows.
- RememberLine stays `{payload,key,value}`. Sidecar dump does not add a splat.

Offset ranking is still open. Empty-ring fire_tag (existing remember tests) writes 0 sidecar rows.

## Next

1. When the card is free: `--chat` remember emit, wipe stores, dump sidecar, rank `cos(h_now, vec_K) − cos(h_now, vec_U)` per offset. C2 ≈ 0, C3 ≠ C1.
2. If a winner exists, dormant geometry column on `RememberLine`, unread by residual.
3. Then ORG-H1 think-spike can hand choice+why to the back room.
