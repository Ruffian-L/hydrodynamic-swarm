# Path B inject 9-turn smoke 20260820

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Same 9-turn full-stack smoke with injected-first spike (HYDRO_INJECT_TAG, consume-once). We think probe blend/β/σ and residual vs baseline diverge from 070557 while named chat collapses stay gone.

## Hypothesis

Injected spike at turn 1 holds physics_blend=6.5 and kinetic_noise=1.5 through later turns; |rn-bn| grows vs no-inject full-stack; greeting/Wait/theed/math stay clean. T=0.7 text drift is not the receipt.

## What changed

Identical 9-turn `smoke_convo.sh` under `configs/gates/config.three_surface.toml`, injected-first **spike** (`HYDRO_INJECT_TAG=spike`). Consume-once: later turns do not re-read the env.

```
HYDRO_CONFIG=configs/gates/config.three_surface.toml HYDRO_INJECT_TAG=spike ./scripts/smoke_convo.sh
```

**Out:** `logs/smoke_convo_20260820_073954.txt`  
**Probe:** `logs/smoke_convo_20260820_073954.probe.jsonl`  
**Private:** `private/chats/chat_1787211599_gemma4_chat.txt`  
**No-inject full-stack baseline (not rewalked):** `070557`  
**Isolation (not rewalked):** `070050`

### Inject (stdout, turn 1 only)

Header: `inject=spike` · `residual.enabled_path=true` · `force_cap=1` · T=0.7 · `Endocrine: OFF (--no-endocrine)`

```
[tag-inject] requested=spike synth=Some("<spike>") blend0=1.00 β0=1.50
[HAND] spike blend=6.50 rep=-3.00 β=1.50 σ=1.50 (physics, not T/rep)
[tag-inject] applied=Spike payload=None blend=6.50 β=1.50 σ=1.50 stop=false
[prefill turn=1 n=326 first_id=2 last_id=101 bos=yes]
```

Turns 2–9 prefill lines have **no** `[tag-inject]` (consumed). Hands persist: probe blend stays 6.5 on later turns.

Probe `tag_inject` turn=1: `requested=spike applied=spike physics_blend=6.5 kinetic_noise=1.5 dynamic_repulsion=-3.0`

### Chat (score this, not `grep gemma4>`)

| Turn | Path B 073954 | Full-stack 070557 |
|------|---------------|-------------------|
| 1 | Hello there, friend. | Hello there, friend. |
| 2 | 2+2 is 4. | same |
| 3 | Blue. | same |
| 4 | C-A-T. | same |
| 5 | One, two, three. | same |
| 6 | Residual streams carry information across layers during model inference. They act as a continuous flow of data that updates and refines representations at each step. | …across layers of a neural network. They allow the model to maintain context… |
| 7 | Ready | same |
| 8 | 51 (3×17)+340 (20×17)=391 | same correct split |
| 9 | pangram exact | same |

Wait-loop: none. `theed`: none. Math thrash: none. BOS `first_id=2` all 9 turns. Entropy lock still on short turns (t4 6/6, t9 9/9 `p_top1≥0.99`). T=0.7: turn 6 text drift is **not** the Path B receipt.

### Perturbation vs 070557 (probe — this is the receipt)

Raw tok lines after inject (turn 1):

```
073954 t1s0 Hello force_on=true physics_blend=6.5 qsma_beta=1.5 kinetic_noise=1.5 residual_norm=303.8716 baseline_norm=303.8921
073954 t1s1 " there" force_on=true physics_blend=6.5 qsma_beta=1.5 kinetic_noise=1.5 residual_norm=214.6341 baseline_norm=214.6779
070557 t1s0 Hello force_on=true physics_blend=1.0 qsma_beta=1.5 kinetic_noise=0.0 residual_norm=303.8921 baseline_norm=303.8921
070557 t1s1 " there" force_on=true physics_blend=1.0 qsma_beta=1.4925 kinetic_noise=0.0 residual_norm=210.3099 baseline_norm=210.3113
```

Later tokens still carry the hand (turn 8, same n=66):

```
073954 t8 blend=6.5 β=1.5 σ=1.5 max|rn−bn|=0.7301
070557 t8 blend=1.0 β scheduled-decay σ=0.0 max|rn−bn|≈0.007
```

| channel | 070557 (no inject) | 073954 (spike first) |
|---------|--------------------|----------------------|
| physics_blend | 1.0 all turns | **6.5** all turns |
| kinetic_noise σ | 0.0 | **1.5** |
| qsma_beta | scheduled decay (t8 down to ~1.08) | **held 1.5** (hand) |
| max \|rn−bn\| t1 | 0.0029 | **0.0922** |
| max \|rn−bn\| t8 | ~0.007 | **0.7301** |

Isolation 070050 (`force_on=false`, blend idle) is **not** this receipt.

Unit test (shipped synth→scan→`fire_tag`): `hydro_inject_spike_scan_fire_moves_later_residual_and_qsma` plus consume-once. Capture `path_b_inject_test.txt`.

No 131072 `--d-run`. Self-reg / force-in-revise still unpaid next brick.

Signed: Grok (xAI) · operator Jason

## Findings

Injected-first spike **fired once** (turn 1 stdout `[tag-inject] applied=Spike`; turns 2–9 no re-inject). Probe: `physics_blend=6.5` and `kinetic_noise=1.5` on **all 9 turns** vs 070557 `blend=1.0` `σ=0`. Later tokens inside the script: t8 `|rn−bn|=0.7301` vs ~0.007; qsma β held at 1.5 (hand) vs scheduled decay. Chat KPIs hold: greeting `Hello there, friend.`, no Wait/`theed`/math thrash, BOS every turn, entropy lock remains. Turn 6 wording differed under T=0.7 — **not** the receipt. Isolation 070050 is not this receipt. No 131k.

## Next

Self-reg / force-in-revise (`three_surface.toml` has no `self_reg`; smoke flags still `--no-endocrine`) remains the following brick. Do not rewalk 070050 / 070557 / 073954.

---

Signed: Grok (xAI) · operator Jason
