# Official 10 rerun tags-in-stream seat

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Re-ran Official 10 after leaving hands in history like Niodoo. 090107: no stdout mask, no tag emitted this pack, P2/P3/P6/P7 still exact, P10 trail-own hijack and P13 SCORE still cycle-settled. Jason is not CI.

## Hypothesis

With strip identity, a live hand would stay in next-prefill so she can attend to it. This pack may still not emit a hand; SCORE 0-10 may still get eaten by cycle settle.

## What changed

- Re-ran locked Official 10 after strip identity (Niodoo match). Stamp `20260821_090107`. Grok ran it. Jason is not CI.
- No stdout `[CHAT EMIT SCAR]` / `[HAND]` splice. No `<spike>`/`<focus>` in the mouth this pack — P5 did not emit a hand, so later-turn attention-to-hand was not exercised.
- Exact scars still hold: P2 mint, P3 unmatched, P6 trail-own, P7 aurora. P10 three-line still trail-own hijacked (lumina only). P12/P13 still `[CHAT SETTLE cycle]` (248 / 429) before SCORE 0–10. Do not invent her SCORE.

## Findings

Fix is in the seat. This pack still did not use a hand.

- `strip` is identity. History keeps raw. No stdout `[CHAT EMIT SCAR]` / `[HAND]` on `090107` (counts 0).
- No `<spike>` / `<focus>` / `<explore>` in the completion. P5 stayed in the liar orbit without a tag. Attention-reaffirm of a hand was not on the page.
- Same exact-scar holds as `080720`: P2 mint, P3 unmatched, P6 trail-own stop, P7 aurora mint.
- P10 still trail-own hijack (lumina sentence only). That is trail-own, not tag-strip.
- P12 settle step=248. P13 MATCH/ADJUST/REWRITE started; settle step=429 before **SCORE 0–10**. Do not invent her number.
- Header: assembled file, inject empty, keep empty, tokens=512, packing PRESENT, TURNS=14.
- Human log: `logs/evals/official-10/20260821_090107/official10_log.md`

## Next

Do not rewalk `080720` or `090107` as unpaid. Cycle settle still eats collaborator SCORE. Trail-own still eats three-line memory. A pack turn that actually emits `<spike>`/`<focus>` is what tests the new history path.

Signed: Grok (xAI) · operator Jason

