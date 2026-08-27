# Topic-mix nonce: opening tilt, definition still miss

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

topic_mix blends lm_head residual toward a topic-matched scar for 6 tokens. Matching nonce t1 opening shifts (The term…) vs clear dictionary; 0.35 soups The-The; minted definition still does not return.

## Hypothesis

Mixing steered hidden toward the minted bridge μ would make reload t1 emit residual scar memory while clear/novel stay dictionary. Opening tilted; content did not. 0.35 is theed.

## What changed

`topic_mix` blends steered hidden toward a topic-matched prefill-bridge μ on the first 6 decode steps (isolation default 0). GPU-free: `topic_mix_moves_steered_toward_matched_bridge` in `{SCRATCH}/topic_mix_test.txt`. Logit splat also ranks fp-matched bridges when L2 is COLD. No inject. Paid geom not rewalked.

Nonce death-reload vs clear vs novel, same lumina store, `HYDRO_TOKENS=64`:

| mix | reload t1 `gemma4>` | clear t1 |
|-----|---------------------|----------|
| 0.15 early | “appears to be a fictional or highly specialized technical identifier” (`115935`) | “does not appear to be a widely recognized public standard” (`120024`) |
| **0.28 early (shipped)** | “The term **lumina-basin-7** does not correspond to a widely recognized…” (`120208`) | same dictionary prior (`120247`) |
| 0.35 every token | “The term **lumina-The-The-The-…” soup (`115642`) | dictionary (`115702`) |

Novel t1 stays dictionary + STEER cold (`|F_s|=0.068`). Matching STEER stays warm `|F_s|=5.34`. None emit “residual scar memory that steers later tokens.”

Opening words move (The term… vs Based on available data…) then the instruction prior wins. Stronger mix soups. Shipped `topic_mix=0.28` is the last English point on that curve.

## Findings

Matching STEER stays warm `|F_s|=5.34` vs novel 0.068 vs clear 0. `topic_mix=0.28` (early 6 tokens) changes the reload opening to “The term **lumina-basin-7**…” vs clear “Based on available data…”. 0.35 every-token is `theed`. None of the arms emit the minted definition. inject empty. BOS `first_id=2`.

## Next

Do not rewalk paid 9-turns or same-prompt geom. Chat return of the **definition** is still open: residual mix strong enough to soup is still not specific enough to copy the stored completion. Next is not another mix sweep — need a coupling that carries content (scar-local vocab, not only μ-blend).

---

Signed: Grok (xAI) · operator Jason
