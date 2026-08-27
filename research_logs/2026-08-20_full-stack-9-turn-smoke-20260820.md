# Full-stack 9-turn smoke 20260820

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Ran the identical 9-turn smoke_convo under configs/gates/config.three_surface.toml so residual/hooks/T>0 are on. Isolation 070050 is already paired. We think named Wait/theed/math-thrash classes stay gone with force_cap=1, and probe force_on is true (unlike isolation).

## Hypothesis

Post-BOS full-stack on the same 9-turn script holds readable chat with residual_live true; entropy lock remains; this closes the unpaid Aug 19 same-script re-run.

## What changed

Same 9-turn script as Jason's isolation smoke (`smoke_convo.sh`), only config changed.

```
HYDRO_CONFIG=configs/gates/config.three_surface.toml ./scripts/smoke_convo.sh
```

**Out:** `logs/smoke_convo_20260820_070557.txt` · symlink `logs/smoke_convo_latest.txt`  
**Probe:** `logs/smoke_convo_20260820_070557.probe.jsonl`  
**Private:** `private/chats/chat_1787209561_gemma4_chat.txt`  
**Paired isolation:** `research_logs/2026-08-20_isolation-9-turn-smoke-20260820.md` (`070050`)

### Seat (from the load banner, not inferred)

```
[RESIDUAL CONFIG] residual.force_cap=1  splat_force_scale=0.03  goal_force_scale=0.008  field_wake_scale=0.02
[RESIDUAL CONFIG] residual.enabled_path=true
[RESIDUAL CONFIG] logit.field_alpha=0.02  splat_scale=0.004  governor=true
[RESIDUAL CONFIG] hooks.enabled=true  site=post_mlp  norm_fraction=0.00008
Physics: force_cap=1 T=0.7 max_tokens=128 | gen: rep=1.12 top_k=40 top_p=0.9
Endocrine: OFF (--no-endocrine)     ← still convo_defaults flags; identical 9-turn
```

`self_reg` is **not** in `three_surface.toml` (default `mode=off`). Residual is on for **every** answer token, not only revise. That is the daily-driver full stack, not `config_isolation_self_reg_force.toml`.

Chat stdout does not print per-token `δ/F_g` (that is the oneshot loop). Channel for this smoke is the banner plus probe `force_on` (= `residual_live`) and `residual_norm` vs `baseline_norm`.

### Chat (score this)

| Turn | Prompt | Reply |
|------|--------|--------|
| 1 | Say hi in three words. | **Hello there, friend.** |
| 2 | What is 2+2? | 2+2 is 4. |
| 3 | Name one color. | Blue. |
| 4 | Spell cat. | C-A-T. |
| 5 | Count to three. | One, two, three. |
| 6 | two short sentences residual streams | Residual streams carry information across layers of a neural network. They allow the model to maintain context while processing new data. |
| 7 | one word: ready | Ready |
| 8 | 17 × 23, show arithmetic | longhand **51 (3×17) + 340 (20×17) = 391** |
| 9 | repeat pangram | the quick brown fox jumps over the lazy dog |

BOS: `first_id=2 bos=yes` turns 1–9 (prefill n=326…570).

### Probe channels (125 tok)

- `force_on=true` on **125/125** (isolation 070050: **0/112**). This field is `residual_live` = `steer_hidden && residual_enabled()`.
- `|residual_norm − baseline_norm| > 0` on 115/125 tokens (isolation: **0**, exact). max |Δ|=0.0085. Turn 1 step 0 still `303.8921==303.8921`; split starts step 1 (`210.3099` vs `210.3113`) on token `" there"`.
- `qsma_beta` 1.5 decaying per turn. `physics_blend=1.0`.
- phase stays `answer` (no revise events). Isolation had 13 revise toks.

Quoted probe:

```
turn=1 step=0 force_on=true token=Hello residual_norm=303.8921 baseline_norm=303.8921 p_top1=0.9888 qsma_beta=1.5000
turn=1 step=1 force_on=true token=" there" residual_norm=210.3099 baseline_norm=210.3113 p_top1=0.8683
```

### KPIs vs isolation 070050 / Aug 2 names

| KPI | Isolation 070050 | Full-stack 070557 |
|-----|------------------|-------------------|
| Greeting | Hello, how are you? | Hello there, friend. (3 words, complete) |
| Wait-loop | none | **none** |
| `theed` | none | **none** |
| Math thrash | none; show-work `17×2=34` then `340+34=391` (false sum) | **none**; show-work **correct** `3×17=51` + `20×17=340` |
| Entropy lock `p_top1≥0.99` | t2 4/7, t4 6/6, t5 5/6, t8 39/49, t9 9/9 | t2 4/7, t4 **6/6**, t5 5/6, t8 48/66, t9 **9/9** — still there, no Wait |
| Residual live | false | **true** |

T=0.7 so replies are not a bit-copy. Named collapse classes did not come back with cap=1 / hooks on.

`grep 'gemma4>'` only prints the prefill banner. Replies are the next line. Read the transcript.

### Honest limits

- Not `talk.sh` with endocrine on. Flags still `--no-endocrine --no-save-memory --clear-memory`.
- Not force-in-revise (`self_reg.mode=force`). Residual was on in **answer**.
- Does not make 131k, remember-store live chat, or spontaneous tag emit.
- Aug 2 physics-on chats that souped were missing BOS and used different caps; this is post-BOS full-stack on the **same 9-turn script**.

Signed: Grok (xAI) · operator Jason

## Findings

Full-stack 9-turn **held**. Residual live: `force_on=true` 125/125 vs isolation 0/112. BOS every turn. No Wait, no `theed`, no math thrash. Greeting `Hello there, friend.` Turn 8 arithmetic is the correct 51+340 split (isolation's 340+34 lie is gone). Entropy lock on short turns remains. Chat stdout has no per-token δ/F_g; probe is the channel. Endocrine still off via script flags. `self_reg.mode` default off — residual on in **answer**, not force-in-revise only.

Aug 19 unpaid same-script full-stack re-run is **closed** in the trail (isolation 070050 + full-stack 070557).

## Next

Do not rewalk these two smokes. Next long-form goal can start from both patterns being in `CHANGELOG.md`. Still open if someone wants it: `self_reg.mode=force` (force-in-revise only), endocrine-on talk parity, 131k, live remember, spontaneous tags.

---

Signed: Grok (xAI) · operator Jason

