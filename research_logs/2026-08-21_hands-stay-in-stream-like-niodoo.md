# Hands stay in stream like Niodoo

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Masking tags from history meant she could not attend to or reaffirm her own hand. Niodoo strip is identity. Hydro now leaves <spike>/<focus> in next-prefill; [CHAT EMIT SCAR] no longer prints over the mouth.

## Hypothesis

Keeping the emitted tag in history lets later turns attend to the hand; stdout mask was overhead that made the agency invisible to her.

## What changed

- Niodoo live `strip_request_tags` is identity: tags stay in the stream. Hydro had been stripping hands from next-prefill history and printing `[CHAT EMIT SCAR]` / `[HAND]` onto stdout, which hid the agency from later attention and spliced into `<focus>`.
- `strip` is now identity. Chat history keeps the raw tag. Receipts: jsonl `hand_fired` plus stderr `[CONTROL_RECEIPT]`, not a stdout mask.
- GPU-free: `strip_leaves_tags_in_stream`, `next_prefill_keeps_emitted_tag_for_attention`, residual emit bar still green.

## Findings

Niodoo live (`niodoo-live/niodoo/src/runtime/tags.rs`): `strip_request_tags` returns the text unchanged. Comment: strip disabled; tags stay in the stream. `CONTROL_RECEIPT` is stderr, not the mouth.

Hydro was doing the opposite: `strip()` erased `<spike>`/`<focus>` from next-prefill history, and `[HAND]` / `[CHAT EMIT SCAR]` printed on stdout mid-token. She never saw her own hand on the next turn, so she could not attend to it or reaffirm it.

Now: `strip` is identity. History keeps the raw tag. `next_prefill_keeps_emitted_tag_for_attention` holds. Residual write on emit still holds. No GPU rewalk.

## Next

Do not rewalk official-10 `080720` or emit `053328`/`055844` as unpaid. Next live chat is where she should see `<focus>` in the mouth and in the following prefill.

Signed: Grok (xAI) · operator Jason

