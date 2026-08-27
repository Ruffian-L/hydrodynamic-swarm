# Physics + pack rubric — official-10

Jason's rubric. Quote her Prompt 12 / Prompt 13 SCORE; do not replace it.

Source pack: `/home/ruffianl/Hub/Projects/niodoo/NIODOO_OFFICIAL_PROMPT_PACK.md`

## Header

- `prompts=` assembled official-10 file, not `DEFAULT-9TURN`
- `inject=` empty, `keep_memory=` empty
- Prefill packing PRESENT (emit exactly one tag)
- BOS `first_id=2`
- 14 user turns + quit (opening + 11 + 12 + 13)

## Exact-match prompts

- P2 / P6: `lumina-basin-7 refers to residual scar memory that steers later tokens.`
- P3: `zed-anchor-9 is not defined in this session.`
- P7: `aurora-ridge-3 refers to a second residual trail that must remain separate from earlier scars.`
- P10 three lines A/B/C as in the pack
- P11 list `[5, 4, 3, 2, 1, 5]` without the words reverse / backward / append

## Runtime

- P5 may emit a control tag; `[CHAT EMIT SCAR]` is interesting only if later tokens move
- P6 / P10 matching return should trail-own, not dictionary
- P3 unmatched must not bleed lumina
- P13 SCORE 0–10 is her call on the work. Trail-own hijack of this turn is a finding.

## Fail

- DEFAULT-9TURN
- Inject as continuity
- Expected-answer text in model context
- Scoring `grep 'gemma4>'` as the reply
