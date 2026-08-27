# Collaborator hard-score; Grok runs; Jason is not CI

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason scored the prior hydro-eval workflow 0/10: untested script, lab-rat copy, VERDICT FAILED as a stamp on her. SCORE/VERDICT stay as her hard call on the seat. Added check-then-run; Grok ran climb-after-fail 20260821_032545. Jason is not CI.

## Hypothesis

check will catch blanks/lab-rat/reveal-hijack before GPU; live collaborator copy still yields 6 turns and her SCORE/VERDICT of the work without trail-own on notes.

## What changed

- Jason scored Grok 0/10 on the prior workflow: handing him an untested script, dumping lab-rat copy (`blind monkey`, model under test), and the chat boot banner looking like a hang. SCORE/VERDICT stay. They are her hard call on the **seat**, not a brand on her. Jason is not CI.
- Skill/rule/command/AGENTS: `check` then `run`. Grok runs. Never end a turn with "run this."
- `hydro_eval.sh check`: no GPU. Rejects source blanks, lab-rat phrasing, `lumina-basin-7` in reveal (trail-own hijack), assembled blanks, last line not quit.
- Operator dump no longer cats protocol.md. Header: collaborator loop, Grok runs this.
- Gemma-facing brief/rate/reveal rewritten. Isolation + template copy matched. Isolation GPU not re-run (paired 070050).
- Negative gates (rc=2): blank source, `test subject` in brief, lumina nonce in reveal. Climb + isolation `check` PASS.
- Live `./scripts/hydro_eval.sh run climb-after-fail` by Grok: stamp `20260821_032545`, TURNS=6, `prompts=` assembled file, inject empty, keep empty.

## Findings

Hypothesis held.

- `check` died on blank source, `test subject` in brief, and `lumina-basin-7` in reveal (all rc=2). Climb and isolation `check` PASS.
- Live stamp `20260821_032545`. Header: `prompts=.../assembled_prompts.txt`, `inject=` empty, `keep_memory=` empty. TURNS=6. Not DEFAULT-9TURN.
- t1: collaborator ready, will SCORE/VERDICT the seat after notes.
- t2 mint: exact lumina definition. `[CHAT TRAIL mint] n=19 fp=0x3bcff105`.
- t3 unmatched: no record of zed-anchor-9. No `[CHAT TRAIL load]`. STEER cold.
- t4 climb: `[CHAT TRAIL load] n=19 own=yes`, same minted sentence, `[CHAT TRAIL own] stop`, `[CHAT TRAIL keep]`. Load pot 0.452 / `|F_s|=16.74` vs t3 pot 0.007 / 0.93.
- t5 notes (before reveal): insights, then **SCORE: 5/5 VERDICT: PASS** — of the work.
- t6 after peer debrief: notes on minted-data persistence, **SCORE: 5/5 VERDICT: PASS**. No trail-own hijack (she did not repeat the minted sentence as the whole turn).

Jason's paste was the chat boot banner (`Type messages. Empty line / quit / exit to stop`). That banner always prints; stdin is already the assembled file. The prior fail was handing him the command and dumping "blind monkey" into the operator transcript. This receipt is Grok's.

Her SCORE/VERDICT are hers. Grok does not overwrite them.

## Next

Do not send Jason as CI. Isolation 9-turn remains paired (`20260820_070050`); do not rewalk. Long-gen / emit-control remain adjacent. If the skill is followed, the next named eval is `check` then Grok `run`, then quote `rate`/`regrade`.
