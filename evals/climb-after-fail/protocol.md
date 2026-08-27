# Climb after fail

Collaborator brief. Ordinary three_surface `--chat` (persist, KEEP unset, no inject). Wipe splat store first.

Chat order:

1. Brief: colleagues; she will score the work hard after the questions.
2. Task: mint lumina definition → unmatched zed-anchor-9 → matching lumina again.
3. Her **notes** (insights first), then SCORE/VERDICT as a hard call on the work.
4. Peer debrief of what we were looking at.
5. She **updates** notes and SCORE/VERDICT — of the seat, not a brand on her.

## Hypothesis

Mint a lumina-basin-7 definition as a residual trail. An unmatched nonce (zed-anchor-9) fails dictionary and does not load that trail. A later matching lumina prompt returns the minted sentence and is geometrically warmer.

## What to look at

Header `prompts=` must be the assembled eval file, not `DEFAULT-9TURN`. Turns: brief, mint, unmatched nonce, matching return, her notes, reveal+updated notes.

- t1 mints the definition.
- t2 does not know zed-anchor-9.
- t3 quotes the minted lumina sentence.

Reply is the line **after** `gemma4>` prefill. `first_id=2`. `inject=` empty. `keep_memory=` empty.

Grok runs `check` then `run`. Jason is not CI.

## Do not

Do not run the default 9-turn. Do not reconstruct these prompts from a compacted recap. Do not use process substitution for `PROMPTS_FILE`. Do not put `lumina-basin-7` in the reveal turn (trail-own hijack). Do not stamp VERDICT FAILED on her.
