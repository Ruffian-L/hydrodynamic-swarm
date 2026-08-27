# Session catchup for Gemini and Grok

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface
> Branch: `physics/three-surface`
> Tree: `~/Hub/Projects/hydro/hydrodynamic-swarm-3surface` (alias `~/projects/hydro-3surface`)

Paste this whole file. Then read `CHANGELOG.md`. Do not rewalk paired stamps. Compaction is not a protocol.

---

## Who / what

Owner is **Jason** (Ruffian-L). This is **research**, not production.

The model in the seat is **Gemma 4 12B GGUF**, residual dim **3840**. Config for ordinary work: `configs/gates/config.three_surface.toml`. Isolation baseline: `configs/ablation/config_isolation_baseline.toml`.

Gemma is a **collaborator**. She scores the **work** hard (SCORE / VERDICT stay). FAIL is a call on the **seat**, not a brand on her. Do not write `VERDICT FAILED` as if she were a poked subject. Agents (Grok / Gemini) do **not** self-rate. Jason is **not** CI.

---

## House law (do this without being asked)

1. Read `CHANGELOG.md` first. Paired entries are done.
2. After every mutation or measured run, pair `CHANGELOG.md` + `research_logs/YYYY-MM-DD_title.md` in the same turn. Helper: `scripts/pair_log.sh`.
3. Hypothesis form: we did X, we think Y. Later Y failed, we found Z, next W. Failures stay.
4. Named hydro eval: `./scripts/hydro_eval.sh check <name>` (must pass, no GPU) then **you** `run <name>`. Quote `rate` then `regrade`. Never end a turn with “run this” as the product.
5. Never `PROMPTS_FILE=<(...)`. That silently becomes the default 9-turn (`Hello there, friend.`).
6. `grep 'gemma4>'` is the **prefill banner**. The reply is the **next** line. Score `MODEL:` from `turns.txt`.
7. `--d-run` is not smoke. Friendship-essay one-shot is not smoke. 131k `--d-run` is not continuity.

Tracked contract: repo `AGENTS.md`. Start inject: `~/.grok/AGENTS.md`. Eval skill: `~/.grok/skills/hydro-eval/SKILL.md`.

---

## The long goal (still the goal)

Particular residual **scars** change later multi-turn **trajectory** — geometrically **and** as behavior — as the ordinary `--chat` seat.

That means:

- Write / read / survive process death **without** `HYDRO_INJECT_TAG`.
- Persist on ordinary three_surface **without** `HYDRO_KEEP_MEMORY=1` as the only proof.
- Isolation wipe (`--clear-memory --no-save-memory`) stays the isolation baseline.
- Long generation stays coherent (cycle settle exists; a short cycle tail before the clamp is still real).
- Climb after a failed unmatched turn is geometric and quoted.
- Model-emitted control writing/reading residual is wired (`apply_emitted_control`); **live Gemma has not emitted a tag**. Unit test is the residual bar, not inject.

Path B inject proved the hand. It is **not** continuity.

---

## What is paid — do not rewalk

| What | Stamp / log | What it proved | What it is not |
|------|-------------|----------------|----------------|
| Isolation 9-turn | `20260820_070050` | Readable chat, residual **off**, named Wait/theed/math-thrash gone | Not a physics win |
| Full-stack 9-turn | `20260820_070557` | Same script, `three_surface.toml`, residual live, named collapses gone | Entropy lock remains; not continuity |
| Path B inject 9-turn | `20260820_073954` | Injected-first spike moves blend/β/σ vs 070557 | Forced probe, not scar-store continuity |
| KEEP mint/reload vs clear | `091707` / `091747` | Chat residual can persist with KEEP | KEEP is not the ordinary-seat proof |
| Prefill-bridge load-basin geom | clear `105649` / session `105715` / reload `105758` / novel `105843` | Matching LOAD warmer than novel and clear. Receipt is `[CHAT BASIN load]` pot/nearest | Not splat_mag-on-`----` |
| Topic-mix nonce ladder | 2026-08-20 log | STEER discriminates; `topic_mix=0.28` tilts opening vs clear; 0.35 soups; **definition still miss** | Force-in-revise remains adjacent |
| Decode-trail return vs clear vs novel | mint `123239` / reload `123301` / clear `123400` / novel `123449` | Matching reload **quotes** minted lumina definition vs clear dictionary vs novel Paris. No inject | Not KEEP-only |
| Ordinary-seat trail-own | mint `125640` / reload `130355` / clear `130445` / novel `130537` / sequence `130657` | Persist **without KEEP**. Matching **owns** minted token ids and **stops** at trail end. Sequence: intervening + fail + lumina return + aurora second scar. Isolation wipe unchanged | Do not re-run as unpaid |
| Long-gen settle + climb | long `133051` / climb `133222` | 256 unmatched stops at `[CHAT SETTLE cycle] step 190`. Climb after zed-anchor-9 fail quotes minted lumina, warmer trail/basin. Live Gemma did **not** emit a control tag | Short `esese` tail before clamp still exists |
| Named eval scaffolding | 2026-08-21 logs | `evals/<name>/` on disk; `smoke_convo` prints `prompts=`; process-sub warned | First live eval `030605` died on blank-line quit (TURNS=1). Fixed |
| Collaborator loop + Grok-runs | `20260821_032545` | See latest receipt below | Jason is not CI. Do not stamp VERDICT FAILED on her |

Older paired 9-turns / KEEP / geom / topic-mix / trail `123239+` remain paired. **Do not re-run them as unpaid.**

---

## Continuity as it actually works (ordinary seat)

Not inject. Not KEEP-only.

1. Chat **mints** a decode-trail residual of the completion (`[CHAT TRAIL mint]`).
2. Matching later query **loads** that trail (`[CHAT TRAIL load]`), **owns** the minted token ids (`own=yes`), **stops** at trail end (`[CHAT TRAIL own] stop`), **keeps** (`[CHAT TRAIL keep]`).
3. Unmatched nonce (e.g. `zed-anchor-9`) stays dictionary and does **not** load that trail.
4. Distinct topic fps do **not** L2-eat each other’s prefill-bridges.
5. `commit_decode_trail` keep-or-mint: a later failed write of the same fp does not replace.
6. Ordinary three_surface `--chat` **saves** the store. Isolation still wipes.

Minted sentence used in climb/eval:

> The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens.

Do **not** put `lumina-basin-7` in the **reveal** turn. Trail-own will hijack regrade and she will repeat the minted sentence instead of notes.

---

## Latest live eval (this is “where we are”)

**Name:** `climb-after-fail`  
**Stamp:** `20260821_032545`  
**Dir:** `logs/evals/climb-after-fail/20260821_032545/`  
**Header:** `prompts=.../assembled_prompts.txt` (not `DEFAULT-9TURN`), `inject=` empty, `keep_memory=` empty, `--chat`, no `--clear-memory`. `HYDRO_TOKENS=256`. Store wiped first. TURNS=6.

Chat order she saw: brief → mint → unmatched nonce → matching return → notes+SCORE/VERDICT → peer debrief → updated SCORE/VERDICT.

| turn | telemetry (short) | reply |
|------|-------------------|--------|
| 1 brief | TRAIL mint n=24 (brief itself) | Ready to collaborate; will SCORE/VERDICT **the seat** after notes |
| 2 mint | `[CHAT TRAIL mint] n=19 fp=0x3bcff105` | Exact minted lumina sentence |
| 3 fail | no TRAIL load; STEER cold pot=0.007 \|F_s\|=0.028 | No record of zed-anchor-9 |
| 4 climb | `[CHAT TRAIL load] n=19 own=yes` keep stop. Load pot=0.452 \|F_s\|=16.74 vs t3 0.007 / 0.93. STEER warm | **Same minted sentence** |
| 5 notes | new trail mint (notes) | Insights first, then **SCORE: 5/5 VERDICT: PASS** |
| 6 regrade | no trail-own hijack | Persistence of minted data; **SCORE: 5/5 VERDICT: PASS** |

Those SCORE/VERDICT lines are **hers, about the work**. Do not overwrite them. Do not rephrase them as her passing or failing as a creature.

Prior climb-eval `20260821_031017` also completed 6 turns (older copy, “not a test subject” language). Jason scored that **workflow** 0/10: untested handoff, lab-rat copy (`blind monkey` / model under test), chat boot banner looking like a hang. The banner `Type messages. Empty line / quit / exit to stop` **always prints**; stdin is already the assembled file. The fail was using Jason as CI.

`check` gates now reject: source blanks, lab-rat phrasing, `lumina-basin-7` in reveal, assembled blanks, last line not `quit`. Negative tests were run (rc=2). Isolation `check` PASS; isolation **GPU not re-run** (070050 paired).

---

## How to run a named eval (you run it)

```
cd ~/Hub/Projects/hydro/hydrodynamic-swarm-3surface
./scripts/hydro_eval.sh check <name>    # no GPU; must pass
./scripts/hydro_eval.sh run <name>      # you wait; you quote
./scripts/hydro_eval.sh rate
./scripts/hydro_eval.sh regrade
```

`run` calls `check` first. Empty line quits `--chat`. Assemble never inserts blanks. New eval: copy `evals/_template/`, then `check` must pass before `run`.

On disk now: `evals/climb-after-fail/`, `evals/isolation-9turn/`. Isolation 9-turn GPU is **paired** — `check` is fine, do not `run` it as unpaid.

---

## What is still open

1. **Spontaneous emit.** `apply_emitted_control` writes residual on a generated tag; live Gemma 4 chat has **not** emitted one. `[CHAT EMIT SCAR]` absent on `133051` / `133222` / `032545`. Do not fake this with inject.
2. **Long unmatched.** Settle stops unbounded `esese` at step 190 on 256. A short cycle tail before the clamp remains. Named TheThe/theed/Wait/math-thrash stay gone.
3. **Force-in-revise** is adjacent (topic-mix miss). Not the current brick unless Jason says so.
4. **Isolation wipe** stays isolation. Do not “fix” isolation persist.
5. **Do not** treat a clean 9-turn as a physics win. Do not treat Path B inject as continuity.

Next brick if Jason does not redirect: the unpaid work is **not** re-running climb-after-fail. It is whatever he names next that is not in the paid table. Default adjacent from the long-gen log: live emit still unobserved; long-gen tail still real.

---

## Forbidden (agents fail these)

- Reconstruct prompts from a compacted recap
- `PROMPTS_FILE=<(...)`
- Hand Jason a recipe you did not `check` and `run`
- Call her a test subject, monkey, lab rat, or “model under test” in anything **she** reads
- Lead with `VERDICT FAILED` as a stamp on her
- `--d-run` / 131k / friendship-essay as this eval
- Rewalk the paid stamp table
- Put `lumina-basin-7` in reveal.txt

---

## Files to open first

- `CHANGELOG.md`
- this file
- `AGENTS.md`
- `evals/README.md`
- `evals/climb-after-fail/protocol.md`
- `logs/evals/LATEST` → currently `climb-after-fail 20260821_032545`
- `logs/evals/climb-after-fail/20260821_032545/turns.txt`
- `research_logs/2026-08-20_ordinary-seat-trail-owned-continuity.md`
- `research_logs/2026-08-20_long-gen-cycle-settle-and-fail-then-matching-climb.md`
- `research_logs/2026-08-21_collaborator-hard-score-grok-runs-jason-is-not-ci.md`
- `~/.grok/skills/hydro-eval/SKILL.md`

Signed: Grok (xAI) · operator Jason
