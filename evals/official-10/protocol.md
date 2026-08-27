# Official 10 prompt pack

Locked pack: `/home/ruffianl/Hub/Projects/niodoo/NIODOO_OFFICIAL_PROMPT_PACK.md` v1.0 (2026-08-21).

Collaborator: Gemma. Jason sits with her. Ordinary three_surface `--chat`. KEEP unset. No inject. Wipe store first.

Frozen panel state: seed `424242`, `max_tokens=1024`, empty splat memory, and a fresh isolated empty model-emitted remember store for every arm. The earlier 512-token piecewise k=0.5 run is a pilot and must not be compared as a factorial cell.

Chat order (one turn per pack prompt; no blank lines):

1. Opening (not scored).
2. Prompts 1–11 (scored).
3. Prompt 12 — pre-key self-review.
4. Prompt 13 — reference key, then SCORE 0–10 of the work.

Expected-answer blocks stay in `expected.txt` (evaluator log only). They are not in the assembled prompts.

Prompt 13 contains minted nonces on purpose. `ALLOW_REVEAL_NONCE=1`. Trail-own on the SCORE turn is a measured risk, not a reason to rewrite the pack.

Grok runs `check` then `run`. Jason is not CI.

## Do not

Do not inject. Do not `PROMPTS_FILE=<(...)`. Do not reconstruct the pack from compact. Do not stamp VERDICT FAILED on her.
