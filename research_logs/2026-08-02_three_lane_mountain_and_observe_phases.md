# Three-lane mountain + observe phases on tok logs

**Date:** 2026-08-02  
**Workbench:** `hydrodynamic-swarm-3surface`  
**Related:** `docs/SELF_REG_PHASES.md`, isolation baseline `[self_reg] mode=observe`

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason
- **Role:** implementation · telemetry · docs · force/self-reg phases · revise-ownership note
- **Project:** hydrodynamic-swarm (worktree `hydrodynamic-swarm-3surface`)
- **Date written:** 2026-08-02
---

---

## Intent

Lock the merge vision in plain research language (no woo) and finish **observe** plumbing so multi-turn smokes can label answer / revise / settle per token.

## Who labels revise? (2026-08-02 — Jason Q)

**Runtime labels; model may emit the cue text.**

- We do **not** inject “something is wrong” into the prompt or token stream.
- On Spell cat, the **model itself** generates `Wait, that's wrong. Let me try again.`
- Our heuristics (`src/main.rs` generate_turn_ex) see `Wait` / `try again` / `wrong` in `pieces` → `phase=revise`, probe `"reason":"text_cue"`.
- Other reasons: `entropy_margin` (stats only), `line_repeat` / `phrase_repeat` (pattern in model output).
- `mode=force` residual is **gated on** that label; it does not create the Wait language.

Canonical doc section: `docs/SELF_REG_PHASES.md` → “Who labels revise?”.

## Vision (Jason — compressed)

- **Three lanes already high:** hydro 3surface (residual + self-reg phases), SplatRAG `/s` (everything saved + picker + steer API), niodv4/OI/Jacobian (packets, first-thought / perm-address keys).
- **Prize demo:** fail → auto memory packet → kill process → same problem → picker reloads right packets → solve **without** re-reading the manual.
- **Not** pure semantic stuffing. Semantics may influence clustering; the bet is **state** from the right multi-packet set (TCS-style residual influence). Claude multi-packet smoke (“this changes things”) sits on the same ridge — recover log under SplatRAG if present.
- Jacobian / first-thought keys = instructional addresses for storage, not a soul.

## Code this session

1. `SelfRegConfig` + `[self_reg]` on isolation baseline (`mode=observe`).
2. Phase transitions logged: `phase` events + settle clamps.
3. **Every** COLLAPSE_PROBE `tok` line now carries `"phase":"answer|revise|settle"`.
4. Settle clamps (`channel`, `</thought>`, hyphen thrash, EOS) all stamp `phase=settle`.
5. `config.example.toml` documents `[self_reg]`.
6. `docs/SELF_REG_PHASES.md` expanded with multi-packet / cold-restart prize and TCS-without-woo table.

## How to read

```bash
# same defaults as talk (isolation + observe)
export COLLAPSE_PROBE=logs/collapse_observe.jsonl
./scripts/smoke_convo.sh
# or human: COLLAPSE_PROBE=... ./scripts/talk.sh

rg '"event":"phase"|"phase":"revise"|"event":"settle' logs/collapse_observe.jsonl | head
```

## Smoke (2026-08-02 ~07:12 UTC) — observe works

`./scripts/smoke_convo.sh` · isolation · `self_reg.mode=observe` · 12B  
Logs: `logs/smoke_convo_20260802_071246.{txt,probe.jsonl}` · latest symlinks

| Metric | Value |
|--------|--------|
| probe events | 325 (9 turns) |
| tok by phase | answer **172**, revise **126** |
| revise fires | turn 4 (Spell cat, step 8 on “Wait”), turn 6 (residual streams, step 4 entropy spike) |
| clean settle via eos | turns 1–3,5,7,9 |
| max-token thrash | turn 4: 120 revise toks, **no** eos/settle clamp (loop “Wait…try again”) |
| miss | turn 8 math: 128 **answer** toks of `17 × 10 = 170` repeat — low entropy thrash, revise heuristic **did not** fire |

**Lesson:** text cue + entropy/margin catches self-reg language; **confident repetition thrash** stays labeled `answer`. Next heuristic: n-gram / line-repeat → revise (or settle clamp), then force only in revise.

## Next (ordered)

1. ~~Run smoke with probe; histogram answer/revise/settle lengths.~~ **Done**  
2. ~~Revise detect: **line-repeat thrash** (turn-8 class).~~ **Done**  
3. ~~Residual **force schedule in revise** (`self_reg.mode=force`).~~ **Done**  
4. ~~settle on multi-line **block** thrash + same-line **phrase** thrash.~~ **Done**  
5. Bridge: fail-path auto packet write → cold pick (SplatRAG text bridge; host embed).  
6. First-thought Jacobian signature → packet key cluster (multi-packet k, not single dump).  
7. Tune `force_*` (does light revise force shorten thrash vs observe-only clamps?).

## Force-in-revise smoke (2026-08-02 ~07:24–07:30 UTC)

Config: `configs/ablation/config_isolation_self_reg_force.toml`  
`mode=force`, physics.force_cap=0, revise uses force_cap=0.6 / goal=0.08 / field=0.05

| Result | Detail |
|--------|--------|
| force_gate | answer force_on=false; revise force_on=true (cap 0.6) |
| Spell cat | revise text_cue → force on → **settle_wait_loop** @3 (not 128 toks) |
| Math thrash | revise text_cue → force on → **settle_phrase_repeat** (not full budget) |
| Honest | Light residual did **not** magically fix wrong arithmetic; phase gate + settle clamps did the cut. Tuning force is open. |

## Smoke after line-repeat (2026-08-02 ~07:18 UTC)

`logs/smoke_convo_20260802_071833.*`

| Turn | Content | Before | After |
|------|---------|--------|--------|
| 8 | `17×10=170` loop | 128 **answer** toks, no settle | ~4 lines → `phase=revise` reason=`line_repeat` → **`settle_line_repeat`** stop |
| 4 | Spell cat Wait-loop | revise via text_cue, burns max tokens | still revise (text_cue); settle not yet (lines alternate, not identical) |

Config knobs: `revise_line_repeat=2`, `settle_line_repeat=4`, `line_repeat_min_chars=6`.  
Unit tests: `generation_tests::line_repeat_*`.

## Not next

- Re-proving OI/TEDE from zero.  
- Claiming Gemma multi-turn “fixed.”  
- Raw 64D inject into wrong residual D.

## Force-tuning loop (2026-08-02 ~18:00 UTC) — revise-only residual knobs locked

**Config:** `configs/ablation/config_isolation_self_reg_force.toml`  
**Knobs (no change this fire):** `force_cap=0.6` `force_goal_scale=0.08` `force_splat_scale=0.05` `force_field_scale=0.05` · `physics.force_cap=0`  
**Hypothesis:** Light revise residual either shortens thrash vs observe or does not worsen it; gate must stay revise-only.  
**Smoke:** `logs/smoke_convo_20260802_175853.{txt,probe.jsonl}` · exit 0 · T=0 greedy · 12B · matches prior force probe `072956` bit-for-bit (deterministic).

### Probe histogram (175853)

| Metric | Value |
|--------|--------|
| tok by phase | answer **76**, revise **96** |
| force_on | answer **0/76**; revise **93/96** (3 transition toks off) |
| force_gate | answer `force_on=false` cap 0; revise `force_on=true` cap 0.6 / goal 0.08 / splat 0.05 / field 0.05 |
| residual_norm (force_on revise) | min 140 · mean **217** · max 294 |
| residual_norm (answer) | mean **227** (no force) |
| turn 4 Spell cat | answer 7 → revise 41 → **`settle_wait_loop` @ step 48** (wait_loops=3) |
| turn 8 math | answer 30 → revise 52 (text_cue) → **`settle_phrase_repeat` @ step 82** |
| clean eos | turns 1–3,5–7,9 |

### A/B same prompt set (observe vs force)

| Run | Mode | Probe | answer/revise | t4 revise | t4 settle | t8 | residual revise mean |
|-----|------|-------|---------------|-----------|-----------|-----|----------------------|
| observe early | observe | `071246` | 172 / 126 | 120 | none (max burn) | 128 **answer** thrash | ~196 |
| observe + line_repeat | observe | `071833` | 65 / 151 | 121 | none | `settle_line_repeat` @46 | ~197 |
| force + clamps | force | `175853` / `072956` | 76 / 96 | **41** | `settle_wait_loop` | `settle_phrase_repeat` @82 | ~217 |

**Honest result:** Revise-only residual **gate is correct** (checklist 1). Thrash turns **do not max-token burn** under wait_loop / phrase_repeat / line_repeat (checklist 3). Light force at these knobs **does not explode residual_norm** and **does not worsen thrash**. Shorten vs pure observe is mostly **settle clamps**, not residual shove — force itself is a **negative / null for thrash length** at physics_light scales. Arithmetic still wrong under force; no quality claim.

**Decision:** **No knob change.** Keep `0.6 / 0.08 / 0.05 / 0.05` as the revise-only residual schedule. Goal checklist for force-tuning loop: **DONE**.


## Jacobian multi-key brick (2026-08-02)

- Schema + cluster + MultiKeyAddress in `src/jacobian.rs` (KeyPhase, DimSignature, JacobianKey, PickQuery).
- Distance: weighted Jaccard; cluster: union-find threshold; emit_pick_query(k≈8).
- Unit tests `multi_key_*`: 9 pass. Full SplatRAG cold pick still open; live phase-edge key push next.
- Log: `research_logs/2026-08-02_jacobian_multi_key_picker.md`.
