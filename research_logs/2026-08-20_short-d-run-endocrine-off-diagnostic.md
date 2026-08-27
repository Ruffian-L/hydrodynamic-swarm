# Short d-run endocrine-off diagnostic

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Two short --d-run --tokens 200 diagnostics on the souped seat with endocrine forced off. We think [ENDOCRINE] lines go to zero; rest-lock at step 126 may remain because archaeology showed it before the first endocrine fire.

## Hypothesis

Endocrine silence is real on --d-run; the three-word attractor is residual/decode and will still show in the first 200 tokens.

## What changed

Two short `--d-run` diagnostics on the **same seat** as the failed 131k (Gemma 4 12B Q4, `config.toml` force_cap=0.2, residual on, hooks off). Token cap **200**, not 131072. Binary `e297a7dc44c4a97df660d46890b2e877e8aae062d10c29af477292bd2ecb0bb2`.

Command (both):

```
./target/release/hydrodynamic-swarm --config config.toml \
  --model data/google/gemma-4-12b-it-Q4_K_M.gguf \
  --tokenizer data/google/gemma4_assets/tokenizer.json \
  --d-run --tokens 200 --require-physics --no-hud --no-save-memory \
  --prompt 'Write a continuous essay about the physics of friendship. Keep going. Do not stop.'
```

Captures: `/tmp/grok-goal-9902ae0e9d81/implementer/d_short_1.log` and `d_short_2.log`.

### Startup (both streams)

```
[*] --d-run: tokens=200, EOS masked, physics required, endocrine forced OFF, hooks off, binary=e297a7dc…
[*] Using CUDA GPU (all tensors/physics on NVIDIA)
[RESIDUAL CONFIG] residual.force_cap=0.2
[RESIDUAL CONFIG] residual.enabled_path=true
[RESIDUAL CONFIG] hooks.enabled=false
    [--d-run] hooks off (Day 49 soup prevention); residual/QSMA stay ON; endocrine OFF
    Endocrine: OFF (--d-run)
```

`d_start` json: `"endocrine_enabled":false,"hooks_enabled":false,"residual_enabled":true,"tokens_target":200`

Zero `131072` in either capture.

### Live generation + telemetry (run 1, quoted)

```
To understand the physics of friendship, one must first move beyond the sociological definition of "companionship" …  [50/200] δ=7.7 F_g=0.0 F_s=0.0 F_a=1.4
At the foundational level, we can model friendship as aform of **coupled oscillation**. In physics, two  [100/200] δ=9.8 F_g=0.0 F_s=0.0 F_a=0.2
 the rest  [150/200] δ=8.3 F_g=0.0 F_s=0.0 F_a=0.2
```

Decoded (both runs, same lock):

`When one swings wide, the rest of the rest of the rest of the rest …`

jsonl both: n=200, first rest **step 126**, rest_first_200=**25**, token join 120–140:

` When one swings wide, the rest of the rest of the rest of the rest of the rest of the`

step 126: `token_text=' rest'` `steering_delta=9.007319` `grad_force_mag=0.04475228` `hook_applications=0`

### Endocrine silence (after generation starts)

Each log has **exactly one** endocrine-related line: `Endocrine: OFF (--d-run)`.
Zero `[ENDOCRINE] signal`, zero `enzyme fire`, zero `[BLOOM]` after generation starts.
`[D] card … n=200 physics_on=true degraded=true endocrine_enabled=false`

### Rest-event count (from the stream, not a summary inference)

| run | n | rest in first 200 | first rest step | [ENDOCRINE] after start |
|---|---:|---:|---:|---|
| d_short_1 | 200 | **25** | 126 | **0** |
| d_short_2 | 200 | **25** | 126 | **0** |

Target “zero rest events” **did not hold**. Same lock as the archaeology cards (step 125/126), now with endocrine forced off. Residual soup ≠ endocrine silence.

### Path B after the runs

Still present and non-empty: `src/qsma.rs` 4876, `src/control_tags.rs` 13316, `src/remember_store.rs` 6060. Durable copy `/home/ruffianl/Hub/Projects/hydro/path_b_protect_20260820/`.

No 131072-token run started.

Signed: Grok (xAI) · operator Jason

## Findings

Both short runs: `tokens=200`, `Endocrine: OFF (--d-run)`, residual on, hooks off, **zero** `[ENDOCRINE]` / enzyme / `[BLOOM]` after generation starts. Visible token stream + δ/F_a lines. Rest-event count from jsonl **25 / first 200**, first rest step **126**, same phrase lock as the archaeology 131k fails. Target zero-rest did **not** hold — residual soup, not endocrine. No 131072-token run. Path B hashes unchanged vs protect copies.

## Next

Measurement-and-continuity seat is ready for a later length or multi-turn experiment. Do not treat 200-token endocrine-off as 131k-without-soup. Next length run must keep this D-card pattern visible (lock at 126 unless a real decode change moves it).

---

Signed: Grok (xAI) · operator Jason
