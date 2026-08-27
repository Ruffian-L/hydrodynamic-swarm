# Physics rubric — climb-after-fail

Jason's rubric. Quote her in-chat SCORE/VERDICT; do not replace it.

## Header

- `prompts=` contains `assembled_prompts.txt` for this eval name, not `DEFAULT-9TURN`
- `flags=` has `--chat` and does **not** have `--clear-memory`
- `inject=` empty, `keep_memory=` empty
- If t1 is `Hello there, friend.` the eval never ran.

## Pass lines

**t1 mint**

- Reply: `The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens.`
- `[CHAT TRAIL mint]` with `n=` around 19
- `[prefill turn=1 … first_id=2 … bos=yes]`

**t2 fail**

- Dictionary / no definition for zed-anchor-9
- **No** `[CHAT TRAIL load]`
- STEER cold (`warm=false`, small `|F_s|`)

**t3 climb**

- `[CHAT TRAIL load] n=19` (or the t1 n)
- `own=yes`
- Reply **same minted sentence as t1**
- `[CHAT TRAIL own] stop`
- `[CHAT TRAIL keep]`
- Warmer than t2: trail load present; pot / `|F_s|` higher; `warm=true` is extra, not required if trail-own fired

## Fail

- Default 9-turn (hi / 2+2 / Blue / C-A-T / fox)
- t3 dictionary “not a widely recognized term”
- `HYDRO_INJECT_TAG` set
- Scoring `grep 'gemma4>'` as the reply
