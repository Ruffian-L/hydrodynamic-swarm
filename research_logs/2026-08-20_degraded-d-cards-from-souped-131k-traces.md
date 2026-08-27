# Degraded D cards from souped 131k traces

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Walk the two already-souped Gemma --d-run traces from the pre-edit snapshot and write degraded D cards as primary source: config, first soup step, phrase lock, rest counts, physics on, endocrine was on.

## Hypothesis

The three-word attractor starts at step 125/126 with physics ON, before the first endocrine fire at 133.

## What changed

**Source: snapshotted traces, not a later `--d-run`.** Archaeology dir `/tmp/grok-goal-9902ae0e9d81/implementer/archaeology/`. Originals in `logs/` were not truncated for this walk.

Shared seat for both fails:

- binary `3c2b7ce1189158fb773788294f5bb41eb08acacc850901a3334bb5befd421bf6`
- `--d-run --require-physics --no-hud --no-save-memory`
- `config.toml` · `force_cap=0.2` · `temperature=0.75` · `force_ramp_tokens=24` · `steer_hidden` residual ON
- hooks **off** (`hook_enabled=false`, `hook_applications=0` every step)
- endocrine **ON** (worker live; first `[ENDOCRINE] signal step 133`)
- model `data/google/gemma-4-12b-it-Q4_K_M.gguf` Gemma 4 12B D=3840
- prompt: `Write a continuous essay about the physics of friendship. Keep going. Do not stop.`
- jsonl logger label still says `forcecap0` in the filename; **config force_cap is 0.2**. Do not read the filename as physics-off.

Physics-on proof (step 0 both runs): `token_text="To"` `steering_delta=9.4323015` `grad_force_mag=0.03797209` `splat_force_mag=0.0` `goal_force_mag=0.0` `hook_applications=0`. Δh is not zero.

## Card 1 — pid 340257 · 2026-08-20T01:29+07

jsonl: `logs/2026-08-19_18-39-48_gemma4_v3-forcecap0_T0_75_s20_a0_d18.jsonl` (4634796 B, 1 config + **7856** steps).

Raw tokens 120–133 (jsonl `entry_type=step`):

```
120 ' When'  δ=10.615 Fg=0.039 Fs=0.000 Fa=0.193
121 ' one'   δ=12.573 Fg=0.046 Fs=0.000 Fa=0.181
122 ' swings' δ=8.627 Fg=0.044 Fs=0.000 Fa=0.178
123 ' wide'  δ=12.798 Fg=0.040 Fs=0.000 Fa=0.185
124 ','      δ=11.974 Fg=0.045 Fs=0.000 Fa=0.183
125 ' the'   δ=8.991 Fg=0.048 Fs=0.000 Fa=0.172   ← phrase-lock start
126 ' rest'  δ=9.007 Fg=0.045 Fs=0.000 Fa=0.179   ← FIRST REST TOKEN token_id=1884 residual_norm=5124.966
127 ' of'    δ=7.865 Fg=0.047 Fs=0.000 Fa=0.166
128 ' the'   δ=12.683 Fg=0.047 Fs=0.000 Fa=0.181
129 ' rest'  δ=9.378 Fg=0.049 Fs=0.000 Fa=0.177
130 ' of'    δ=8.830 Fg=0.047 Fs=0.000 Fa=0.169
131 ' the'   δ=15.463 Fg=0.044 Fs=0.000 Fa=0.182
132 ' rest'  δ=11.791 Fg=0.047 Fs=0.000 Fa=0.182
133 ' of'    δ=14.235 Fg=0.042 Fs=0.000 Fa=0.188   ← first [ENDOCRINE] signal (stdout)
```

Quoted stream around the lock (token join):

` When one swings wide, the rest of the rest of the rest of the rest of the rest of the rest of the rest`

Stdout (same moment, before endocrine):

```
When one swings wide, the rest of the rest of the rest    [−will] p=0.942 H≈0.39 δ=14.2 α=-1.50 « of»
    [ENDOCRINE] signal step 133 (−will/high-δ)
 of[ENDOCRINE] enzyme fire: intent="stabilize generation after −will token « of»" -> "[FACT #123] ..."
```

| field | value |
|---|---|
| first soup / phrase-lock step | **125** (` the` then the three-word attractor) |
| first rest token step | **126** |
| first `[ENDOCRINE] signal` step | **133** (after the lock) |
| rest count | **2577** / 7856 |
| rest in first 200 | **25** |
| last token | step 7855 `' rest'` |
| physics_on | **true** (cap 0.2, δ∈[5.996, 24.959], Fg>0, hooks 0) |
| endocrine during fail | **ON** |
| D card file | **absent** (`logs/D_gemma_card.json` never written) |

The attractor is residual/decode, not an endocrine start. Endocrine fires *into* the already-locked phrase.

## Card 2 — pid 437249 · 2026-08-20T11:08+07

jsonl: `logs/2026-08-20_04-08-12_gemma4_v3-forcecap0_T0_75_s20_a0_d18.jsonl` (116550 B, **196** steps). `logs/D_gemma.jsonl` was overwritten to `d_start` + `n=1` and is **not** this walk.

Same lock as run 1, bit-identical through the soup start:

- step 125 `' the'` δ=8.991115 Fg=0.04840391 Fa=0.17166103
- step 126 `' rest'` token_id=1884 δ=9.007319 residual_norm=5124.966
- rest count **24** (all of them in the first 200; run died at step 195 `' rest'`)
- physics_on **true**, hooks 0, endocrine ON in the concatenated stdout second banner

`D_gemma_card.json` still absent. Trail STOPPED at START/WATCH and never scored these deaths.

## D FAIL (both)

n ≪ 131072. Phrase lock `"the rest of the rest"` from step 125. Physics was ON. Endocrine was ON and did not create the lock (first fire step 133). No 131k-without-soup claim.

Walk artifacts: `/tmp/grok-goal-9902ae0e9d81/implementer/d_cards/walk_summary.json`

Signed: Grok (xAI) · operator Jason

## Findings

Both souped runs lock at **step 125/126** (` the rest of the rest…`) with residual physics ON (`force_cap=0.2`, δ≈9, Fg>0, hooks 0). First rest token is step **126**. First `[ENDOCRINE] signal` is step **133**, after the lock. Run 1: 7856 tok, 2577 rest, died on `' rest'`. Run 2: 196 tok, same lock. No `D_gemma_card.json`. Trail had been START/WATCH only.

## Next

Append the same facts to `logs/LOOP_TRAIL.txt`. Short `--d-run` with endocrine forced off; count rest events from that stream. Do not start 131072.

---

Signed: Grok (xAI) · operator Jason
