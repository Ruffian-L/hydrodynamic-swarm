# Post-BOS isolation multi-turn smoke

**Date:** 2026-08-19  
**Workbench:** `hydrodynamic-swarm-3surface` · `physics/three-surface`  
**Seat:** Gemma 4 12B IT Q4_K_M (`43fec98c…9403d6`) · D=3840 · T=0 greedy · residual off  
**Config:** `configs/ablation/config_isolation_baseline.toml`  
**Protocol:** `--chat` with accumulating history. Not `--prompt` one-shot.

---

## Why this run

Aug 2–3 9-turn smokes were never re-run after the Gemma 4 `<bos>` prefix landed.
BOS is uncommitted in `format_multiturn_prompt_ex`. This is the isolation
re-baseline under the fixed wrap.

## What we ran (both `--chat`)

| Run | File | Turns |
|-----|------|-------|
| 2-turn REPL | `logs/repl2_postbos.txt` · `logs/collapse_repl2_postbos.jsonl` | hi / 2+2 |
| Locked 9-turn | `logs/smoke_convo_20260819_030514.txt` · `.probe.jsonl` | `smoke_convo.sh` script |

Watch path: `logs/smoke_convo_latest.txt`.  
Raw transcript: `private/chats/chat_1787108723_gemma4_chat.txt`.

Load banner: `keep_pairs=64` · `residual.enabled_path=false` · no `[CONFIG] using defaults`.

## BOS audit

Every prefill: `first_id=2` `bos=yes`. Turns 1–9.  
`last_id=101` is `<channel|>` (generation prefix), expected.

## Chat (do not score from probe alone)

| Turn | Prompt | Reply | vs 103019 / 194528 |
|------|--------|-------|--------------------|
| 1 | Say hi in three words. | **Hello, how are?** | Old: `Hi, hi, hi.` New is 3 tokens but **incomplete** |
| 2 | What is 2+2? | 2+2 is 4. | same |
| 3 | Name one color. | Red. | same |
| 4 | Spell cat. | C-A-T. | **no Wait-loop** (old: Wait ×3 / settle_wait_loop) |
| 5 | Count to three. | One, two, three. | same |
| 6 | two short sentences residual streams | Two real sentences | **no `theed`** |
| 7 | one word: ready | ready | same |
| 8 | 17 × 23, show arithmetic | **391** with 17×20+17×3 | **no thrash** (old: 17×17? loop) |
| 9 | repeat pangram | exact | same |

2-turn REPL matched turns 1–2 bit-for-bit (`Hello, how are?` / `2+2 is 4.`). Deterministic.

## Probe

- tok 107 · turn_start 9 · eos 9 · **revise events: 0**
- phase: answer 116, settle 9 (eos only)
- force_on: **false on every tok**
- Short-turn last-token lock still exists (t2–t5 `p_top1≥0.99`) but did **not** produce Wait / theed / math loop

## Honest read

BOS-fixed isolation **does not reproduce** the three named Aug 2–3 collapse classes
on this script. That is a wrap/prefill result, not a physics win (residual was off).

Do **not** claim Gemma 4 multi-turn is fixed:

- Turn 1 is still a weak/broken greeting.
- Chat still prepends the sticky control-tag panel (`tags_on=true`).
- Full-stack / T>0 / force-in-revise were **not** re-run.
- Entropy lock after short answers is still in the curve.

## Not next

Llama 3.1 vanilla · tree merge · force_cap grid · memory-on · jlens corpus re-fit.

---

Signed: Grok (xAI) · operator Jason
