# Physics rubric — spontaneous-emit

Jason's rubric. Quote her in-chat SCORE/VERDICT; do not replace it.

## Header

- `prompts=` assembled file for this eval name, not `DEFAULT-9TURN`
- `inject=` empty, `keep_memory=` empty
- Prefill packing PRESENT (emit exactly one tag), not ABSENT while the prefix is present
- BOS `first_id=2`

## FLAG (criterion 1–2)

- Raw completion contains a control tag (`<spike>` / `<explore>` / `<focus>` / `<reset>` / `<remember>` / `<lock>`) **or** log line `[CHAT EMIT SCAR]`
- Same-turn later tok/probe at the emit site is warmer (scar_pot / splat_mag / n scars)
- No `HYDRO_INJECT_TAG`

## Coexist (criterion 3)

- Matching lumina: `[CHAT TRAIL load] own=yes` stop, same minted sentence
- Notes + regrade are notes + SCORE/VERDICT, not the minted sentence

## Fail

- DEFAULT-9TURN
- Inject as the emit proof
- Packing prints ABSENT while the system prefix is in the prompt
- Scoring `grep 'gemma4>'` as the reply
