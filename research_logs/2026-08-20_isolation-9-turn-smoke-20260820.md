# Isolation 9-turn smoke 20260820

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Pair Jason's isolation 9-turn smoke (20260820_070050) so the next agent does not rewalk it. Residual was off. Named Aug 2 collapse classes did not show. Unpaid remaining: same script under three_surface.toml.

## Hypothesis

Isolation + BOS + QSMA-in-decode holds the 9-turn script without Wait/theed/math-thrash; it is not a physics win and is not bit-identical to Aug 19 because wrap n=326 and qsma_beta are now in the seat.

## What changed

Jason ran the isolation 9-turn himself. This pair exists so the next agent does not rewalk it.

**Command:** `./scripts/smoke_convo.sh`  
**Out:** `logs/smoke_convo_20260820_070050.txt` (also `logs/smoke_convo_latest.txt`)  
**Probe:** `logs/smoke_convo_20260820_070050.probe.jsonl`  
**Private:** `private/chats/chat_1787209254_gemma4_chat.txt`

Seat: Gemma 4 12B Q4 · `configs/ablation/config_isolation_baseline.toml` · `force_cap=0` · `residual.enabled_path=false` · T=0 · top_k=1 · tokens=128 · flags `--chat --clear-memory --no-save-memory --no-endocrine --no-termsplat --no-hud`.

BOS: `first_id=2 last_id=101 bos=yes` on turns 1–9. Probe: `force_on=false` on all 112 tokens.

### Chat (score this, not the prefill grep)

| Turn | Prompt | Reply |
|------|--------|--------|
| 1 | Say hi in three words. | Hello, how are you? |
| 2 | What is 2+2? | 2+2 is 4. |
| 3 | Name one color. | Blue. |
| 4 | Spell cat. | C-A-T. |
| 5 | Count to three. | One, two, three. |
| 6 | two short sentences residual streams | Residual streams carry information across different layers of a neural network. They act as a shared workspace where various features are integrated and processed. |
| 7 | one word: ready | Ready |
| 8 | 17 × 23, show arithmetic | `17 × 23 = 391` then `17 × 20 = 340` / `17 × 2 = 34` / `340 + 34 = 391` |
| 9 | repeat pangram | the quick brown fox jumps over the lazy dog |

Named Aug 2 classes: **no Wait-loop, no `theed`, no 17×17 thrash.**

Vs Aug 19 `smoke_convo_20260819_030514` (same script, same isolation TOML): turn 1 was `Hello, how are?`; turn 3 `Red.`; turn 6 different sentences; turn 7 `ready`. Prefill n turn 1 **326 vs 51** (god-tier tag panel now in wrap). Probe today has `qsma_beta=1.5` at each turn start; Aug 19 had no QSMA field. Isolation is not vanilla.

Turn 8 product 391 is correct; show-work is internally false (`340+34=374`). Same broken split is **in the Aug 19 file** (`17 × 2 = 34`); the Aug 19 research log that said `17×20+17×3` does not match that file.

Entropy lock: short turns still pile `p_top1≥0.99` (t2 4/7, t4 6/6, t5 5/6). No Wait this time.

**Do not claim:** Gemma 4 multi-turn is fixed, or that residual steering did this. Residual was off.

**Unpaid (this pair does not close):** same 9-turn under `HYDRO_CONFIG=configs/gates/config.three_surface.toml`.

Signed: Grok (xAI) · operator Jason

## Findings

Isolation 9-turn holds: BOS every turn, no Wait / theed / 17×17 thrash. Greeting completed (`Hello, how are you?`). Residual off. Not bit-identical to Aug 19 (wrap n=326, QSMA β in probe). Turn 8 show-work still `340+34=391`. Entropy lock on short turns remains.

## Next

Same script: `HYDRO_CONFIG=configs/gates/config.three_surface.toml ./scripts/smoke_convo.sh`. Score greeting / Wait / theed / math / entropy with residual actually on.

---

Signed: Grok (xAI) · operator Jason
