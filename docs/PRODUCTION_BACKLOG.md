# Hydro 3surface — production-shaped backlog

> Date: 2026-08-28
> Asked by: Jason
> Shape: imagine this tree had to ship. This is that list.
> Status: **research until Jason says otherwise.** Shipping this file is not a ship decision.
> Audience: local team (Shep · Echo · Lumina · Nex · whoever sits the seat). One numbered item at a time.
> Ghost-team landing card: repo-root [`ghost_team_groktodos.md`](../ghost_team_groktodos.md) (grep that name).

This is the longest honest list the tree can carry today. Items name **files, receipts, and a done-check**. Vague “make it better” rows were cut.

**What Jason is trying to do:** a why-store, not a tape recorder — writing meaning between frozen models. Paper (cite, do not rewrite): `/home/ruffianl/ontological-inversion/paper/` · [doi:10.5281/zenodo.22126782](https://doi.org/10.5281/zenodo.22126782). Hydro is the ecosystem the paper does not claim. Lane **ORG** below is that organism. Pick **all five holes**, not one.

**Owner split**

| Who | Owns |
|-----|------|
| **Jason** | Ratatui TUI (`src/frontend/*`, `PROJECT.md`, `tests/test_ratatui_frontend.rs`). Do not steal that diff. |
| **Local team** | Organism holes **ORG-H1…H5** + everything else. One item, paired changelog + research log. |
| **Grok (this session)** | Wrote the list + `ghost_team_groktodos.md`. Did **not** land QSMA visit-repulsion. Did **not** rewalk paired smokes. |

**How to pick work**

1. Read `CHANGELOG.md` (newest first). Paired smokes in that file are done.
2. Take **one** numbered item. Put the id in the commit / changelog title (`P0-03`, `TUI-04`, …).
3. Pair `CHANGELOG.md` + `research_logs/YYYY-MM-DD_title.md` in the same turn (`scripts/pair_log.sh`).
4. Scope the git commit to what you touched. Ratatui WIP stays out unless Jason hands it to you.
5. GPU: `nvidia-smi` first. QLoRA often owns CUDA. Jason has overridden “wait” before; still log occupancy.
6. Any eval that involves a model: **tell the model it is being tested**, up front. House contract.
7. Hydro eval = `./scripts/hydro_eval.sh check <name>` then `run <name>`. Gemma scores the **work**. Jason is not CI. Never `PROMPTS_FILE=<(...)`.
8. SplatRAG `recall` / `pick` before filesystem grep for history, naming, or “what Jason meant.”

**Start here if you only have one afternoon**

Two tracks. Organism is the north star. Paris is the mouth that cannot leave. A clamp that steals the bad lock starves the ear — do not “fix” Paris with `-inf`.

| Order | Id | Why this first |
|-------|----|----------------|
| 1 | **ORG-H5** | Why-vector mint at choice time. Think-spike has nothing to hand the back room without it. |
| 2 | **ORG-H1** | Think-spike fork/strip. First test: did the peek differ. |
| 3 | **ORG-H3** | Dream-ahead ≠ dream-taken receipts. Enum exists; generate path is silent. |
| 4 | **ORG-H2** | Will-return: scars>0 must move F_s. nopastewoo is the named ear failure. |
| 5 | **ORG-H4** | Lock ≠ Focus — **pinned in this tree.** Sweep niodoo-live; do not rewalk hydro. |
| 6 | **P0-01** | QSMA visit term that can beat Q (Paris). |
| 7 | **TEST-01** | `cargo test --workspace` does not compile. |
| 8 | **DREAM-03** | Hydro TDA is fake: homemade Rust VR on 6 scalars, H0→1. Wire niodoo-live's Python ripser sidecar. |

---

## LAW — load this before any mutation

- Full stack ON. Tune gains. Ablate only after a tuned baseline. `IMMUTABLE_RUN_CONTRACT.md`.
- Three size-scaler transforms exist. Do not collapse them. Do not claim the scaler caused a force without the matched factorial (size-rule × k, frozen state, scaler receipt in header). Audit: `research_logs/2026-08-21_from-run-cards-to-token-receipts.md`.
- Three “J” objects exist. Do not collapse them. Table in `docs/experiments/loop-exit-vs-loop-wording.md`.
- `ControlTag::Lock` is exact match only. Substring `LOCK` inside `blockquote` / `clock` / `unlock` was a landmine and is pinned. Keep `parsed_lock_stops_without_focus_physics` and `lock_is_exact_match_only`.
- Public GitHub parent still skips the engine on chat (`let _ = engine; return run_simple_chat(...)`). Live 3surface `--chat` goes through `generate_turn_ex`. Measure the live path. See `docs/experiments/loop-exit-vs-loop-wording.md`.
- Learned wills language: public face is wills, not scars/poison. `docs/VOCAB.md`.
- This is not a consciousness claim and not a product chat app until Jason says it is.
- Four interfaces: address / state / payload / speech. SplatRAG is 1+2. Mouth is 4. Do not write a think-spike peek as a will.
- Regime A (Nomic128 → Qwen L4) is **not** this hydro seat. Do not inflate it. Paper numbers stay paper numbers.
- Hands > Governor. A cap that prevents the bad lock starves the ear.
- The live model may spike, lock wrong, even die. That is fuel. Do not add guardrails that steal the mistake.

---

## LANE ORG — The organism (why-store)

Landing card: [`ghost_team_groktodos.md`](../ghost_team_groktodos.md).

Jason invented **writing meaning between frozen models**. Paper on record:

- Local: `/home/ruffianl/ontological-inversion/paper/` (`MANUSCRIPT.md`, `Writing-Meaning-Between-Frozen-Models.pdf`, `ABSTRACT.md`)
- Zenodo v1: https://doi.org/10.5281/zenodo.22126782 · concept: https://doi.org/10.5281/zenodo.22126781
- Code: https://github.com/Ruffian-L/ontological-inversion
- HF: https://discuss.huggingface.co/t/ontological-inversion-writing-meaning-between-frozen-models-cross-model-vector-memory-for-steering-recall-and-reasoning/179334

Cite. Do not rewrite. Next text is companion, not a second paper.

Paper writes are ephemeral. This tree is the loop the paper does not claim:

```
choice → consequence (crash is a valid sample) → splat → dream → will → field → next choice
```

```
FRONT ROOM                         BACK ROOM
  frozen mouth                       SplatRAG ears = why-store, not wiki
  residual / 3-surface               dream ahead  = counterfactual why
  monitor in think only              dream taken  = minted why of path
  hands: spike focus lock            evaporation  = time decay; pain lasts
         remember reset explore      TCT / wills  = durable why across death
  think-spike sandbox                Self vector  = slow will graph
  KV may die                         gain         = how hard why presses
  Choice vector C = session why
```

**Two regimes (paper). Keep separate.**

| Regime | Write | Bound |
|--------|--------|--------|
| A / state | Nomic128 → affine → Qwen L4 residual | Sign-sensitive. Glub-Tub stove −0.21…−0.18, fire-pit −0.15/−0.14. Full dir 5/5, random/shuffle 0/5, unrelated 4/5. Not a general inverse. |
| B / payload | ordered Qwen3 fragments → ridge → Llama slots | Content + inference. Needs bandwidth + order. |

Rank-128: sim r=0.937 vs Llama token recon 0.346 (52.3% of full-rank 0.661). Meaning arrives before words.

**Four interfaces:** (1) address — retrieval geometry (2) state — why/stance + gain (3) payload — identity/relation (4) speech — mouth, never the store.

Kimi 2026-08-28 was told pick **all** holes, finished H4, died on H3 quota.

### ORG-H4 — Lock ≠ Focus (pinned here)

**Status:** done in 3surface 2026-08-27. `src/control_tags.rs` exact `LOCK` only; `ControlTag::Lock` excluded from physics hands; persist+stop in `fire_tag`; `gemma4_lock_stops_turn`. Tests: `lock_is_exact_match_only`, `parsed_lock_stops_without_focus_physics`.

**Still open:** niodoo-live / adaptive-agency may still `contains("LOCK")`. Sweep those trees. Hydro: do not rewalk.

**Landmine the brief still names:** “Lock currently aliases Focus on Gemma 4 isolation” — **false of this tree after the pin.** True of a substring parse if anyone reintroduces it.

### ORG-H3 — Dream-ahead receipt ≠ dream-taken receipt

**Files:** `src/dream.rs`, generate path in `src/main.rs`, any tok JSONL / stdout greps.

**Now (corrected 2026-09-03):** `DreamReceiptKind` / `dream_receipt_line` **do not exist** in `src/dream.rs` — the 2026-08-28 note was a docs error. `micro_dream` is a 2–3 step forward projection (ahead-shaped) then TopoCoT clamp. `DreamEngine::run` is Langevin replay of trajectories that happened (taken-shaped). They do not emit `DREAM_RECEIPT`.

**Do:** Print `DREAM_RECEIPT kind=dream-ahead src=…` from the counterfactual projection. Print `kind=dream-taken` from the path that minted. Geometry/stats only (norms, α, σ, counts, ids). A will is a loadable direction + gain, not a saved paragraph.

**Done when:** a unit refuses to construct a mixed receipt; a chat log has both kinds as separate lines; smoke greps them without mixing fields.

### ORG-H2 — Will-return: F_s moves when scars > 0

**Files:** `src/niodoo.rs` splat force, `src/memory.rs`, chat `[CHAT BASIN load]` / STEER lines.

**Named failure:** nopastewoo (4B, 499/500, scars=10, F_s=0, cos 0.93–0.99) = ear failure. Continuity cards prove **potential** / nearest; this hole is **force**.

**Done when:** scars_active>0 ⇒ later-tok `splat_mag` / `|F_s|` nonzero on a matching residual. scars=0 ⇒ F_s≈0. Death-reload of **content-specific** scars moves F_s; novel prompt on the same store stays cold. On-center F_s≈0 with high pot is geometry (CONTINUITY.md) — do not “fix” that into a shove. Off-center matching must shove.

### ORG-H5 — Why-vector mint at choice time (not at KV drop)

**Files:** `src/remember_geometry.rs`, `src/remember_store.rs`, `src/niodoo.rs` `fire_tag`, decode loop. Design: `research_logs/2026-08-27_remember-geometry-column-offset-receipt.md`. Dump: `research_logs/2026-09-03_org-h5-remember-offset-sidecar.md`.

**Hang geometry on `<remember>`, not `<spike>`, not auto-threshold.** Sidecar dump is in (C1, dormant, `inject=false`). Dual-site (S_res vs S_logit) × offsets. Controls C0–C4 including shuffle, sign-flip, interrogator snatch. **Do not add a RememberLine geometry column until ranking beats C2.** If no offset beats C2, park the column — a why-store that cannot discriminate matching vs unmatched is a tape.

**Now:** ring + dump on closed remember. KV drop does not mint. Spike does not mint. Ranking not run (GPU occupied 2026-09-03).

**Done when:** a ranked sidecar exists; dropping KV after emit does not create a second mint; spike peek is absent from the store; then — only then — a dormant geometry field on `RememberLine`.

### ORG-H1 — Think-spike fork/strip + peek-delta receipt

**Depends on:** H5 mint existing enough to take choice+why.

**Sequence:**

1. She wraps a span in `<spike>` **inside think**.
2. Runtime forks KV, physics nudge, 3–5 token peek.
3. Preview is marked, then **stripped** from KV.
4. Only the choice + why-vector hit the back room.
5. Look-and-decline still scars.

**First sandbox test: did the peek differ.** Emit-without-delta is a dead hand. Spike often does not move tokens — that is a measurement, not a skip.

**Do not:** persist peek tokens; let any tag evict; put the monitor in the visible channel (it gets parroted); start StreamingLLM tag-FIFO; CPU-offload-as-doctrine.

**Done when:** a unit shows peek tokens absent from KV after strip; `peek_delta` in the receipt is nonzero on a live nudge and zero on a dead hand; decline still deposits a pain splat.

### ORG-06 — Choice vector C vs Self vector

C = live session why. Self = slow will graph. Gain presses a thin why; do not fake recall. Document both in tok JSONL / HUD. Do not substitute transcript cosine for C.

### ORG-07 — Monitor in think only

Visible-channel monitor gets parroted. Isolation vs chat: prove the mirror line is inside thought, not the answer stream.

### ORG-08 — No tag evicts KV

Unit: fire spike/focus/lock/remember/reset/explore; KV length and prefix ids unchanged. Lock stops the **turn**, not the cache.

### ORG-09 — Pain lasts (evaporation)

Time decay on pleasure; pain outlives it (or a documented λ_pain < λ_pleasure). Ties MEM-03. Receipt: a failure splat still queries after a pleasure splat of equal age has faded.

### ORG-10 — Do not steal the mistake

Paris-forever, bad lock, 4B soup, crash: samples for the ear. Policy may **leave** a basin (P0-01). Policy may not **erase** the visit so the back room never saw it.

### ORG-11 — Paper companion, not a rewrite

Next public text cites doi:10.5281/zenodo.22126782. Hydro receipts (H1–H5, continuity, John A/E) are companion material. Do not port Regime A into Gemma 4 12B and call it the paper.

### ORG-12 — Official 10 stays locked

Prompt 11 gold `[5,4,3,2,1,5]`. Do not rewrite the pack to make a hole look green.

---

## LANE P0 — The mouth is stuck (Paris / QSMA)

**The hole, in one sentence:** residual *moves* (`Δh≈0.55`, `Δz≈7`) and the next token is still `Paris`. The policy cannot leave a high-Q visited token.

**Receipts (do not re-run as unpaid to “see it”):**

- `logs/john_lock_sweep_20260827_155431` — E c=0/target/reverse/random all `Paris`×8. S copies the seeded collapse. `research_logs/2026-08-27_john-lock-sweep-paris-basin.md`
- `logs/john_loop_exit_20260827_134056` — first A/C/E. A at t=0 is zero because h is the prefill goal. `research_logs/2026-08-27_john-loop-exit-12b-receipts.md`
- Heuristic landed then rejected: `mask_repeat_logits` in `src/main.rs`, knobs `generation.no_repeat_ngram=3` / `consecutive_repeat_break=3`. `research_logs/2026-08-28_decode-bans-repeated-ngrams.md`. Jason: this is patching a sinking boat.

**Why it is architectural**

Score in `src/qsma.rs` `bias_top_k`:

```
score_i = Q_i + ease(F_i)×β + C_i + σ·ξ
C_i     = 0.1 / (1 + visits_i)
```

- Q for a collapse token is ~10–20 logits.
- C for 8 visits is ~0.011. Unvisited C is 0.1. ΔC ≈ 0.09. **Cannot beat Q.**
- `update_flux`: energy > 0.1 **deepens** the groove. Repeating Paris is high energy. Flux helps it stay.
- `observe_token` is called with `p_top1` (chat) or `1.0` (oneshot) — both “high energy.”
- `Hands.dynamic_repulsion` (`src/niodoo.rs` ~1198) is a **radial residual** term (black-hole coefficient), not a token-identity visit force.
- Unique-id `rep_penalty` does nothing at T=0 greedy.
- Top-64 only: if the leave-basin token is not already in the leading 64 logits, QSMA never sees it.

Original Niodoo had **loop repulsion as a force** (LoopDetector loop_score → push away from repeated tokens inside `apply_forces`). Hydro ported Q+ease(F)×β+C and dropped the force that could compete with Q.

### P0-01 — Visit term that can beat Q (the real fix)

**Files:** `src/qsma.rs`, `src/niodoo.rs` (`apply_qsma_logits`, `observe_token`)

**Do:** Make C (or a named visit-repulsion term next to C) a **subtractive** function of visits whose magnitude can exceed a typical ΔQ on a collapse token. Keep the QSMA equation. Keep hands on β/σ. This is the policy hole, not a sampler `-inf`.

**Candidate shape (tune, then measure):**

```
C_i = c0 / (1+visits_i)  −  λ · visits_i          // bonus shrinks, penalty grows
```

or log: `C_i = c0 / (1+v) − λ · ln(1+v)`.

λ has to be in **logit units**. 0.1 is the current bug. A starting λ around 2–8 is the scale that can flip a 15-logit Paris against a 7-logit neighbor after a few visits. **Do not guess the final λ from this sentence — write a unit table first.**

**Unit proof before any 12B run:**

1. Two logits: Paris Q=15 visits=4, alt Q=8 visits=0. After bias, alt wins.
2. Same, visits_paris=0. Paris still wins (first say is allowed).
3. Equal Q, high flux on token 0, β=1.5, visits_0=6. Visit term beats ease(F)×β.
4. Existing tests still pass: `high_flux_plus_beta_picks_the_groove` (zero visits), `curiosity_can_beat_a_dead_groove_when_beta_is_zero`.

**Done when:** those tests exist and pass; `bias_top_k` comment states the units; changelog hypothesis names λ.

### P0-02 — Remove the n-gram `-inf` clamp after P0-01 has the unit proof

**Files:** `src/main.rs` (`mask_repeat_logits`, calls in `generate_turn_ex` ~1832 and oneshot ~4878, tests ~5896), `src/config.rs` (`no_repeat_ngram`, `consecutive_repeat_break` defaults), `config.toml`, `configs/experiments/config.plumbing_{off,on}.toml`

**Do:** Default both knobs to **0**. Keep the function for a while as an explicit opt-in so an ablation can turn the clamp back on and show it is *not* the leave-basin mechanism. Delete the tests that treat `-inf` as the product behavior, or gate them on ngram>0.

**Done when:** defaults are 0; Paris-leave (MEAS-01) is demonstrated with ngram=0 consec=0.

### P0-03 — Flux on over-visit writes pain, not groove

**Files:** `src/qsma.rs` `update_flux`; `src/niodoo.rs` `observe_token`; chat call `engine.observe_token(next, p_top1 as f64)` in `src/main.rs`

**Do:** High energy + high visit count is a *bad habit*, not a successful basin. Original QSMA “high energy deepens the groove” is correct for **first** acquisition. After N visits of the same id in one turn, energy should write **pain** (flux decreases).

**Unit proof:** token with energy=1.0, visits going 1→2→8: flux rises then falls. Token with energy=1.0, visits=1: flux still rises.

**Done when:** a repeating-id sequence no longer monotonically increases that id’s flux; first-visit still grooves.

### P0-04 — Cap / expand the QSMA action set

**File:** `src/niodoo.rs` `apply_qsma_logits` hardcodes cap=64.

**Do:** Either (a) raise cap when the leading mass is concentrated (collapse), or (b) always include the least-visited among a wider pool so leave-basin candidates exist. Log `qsma_k` and whether the eventual pick was inside the original top-64.

**Done when:** a tok JSONL field shows pick-in-top64 true/false, and a unit test has a leave-token sitting at rank 80 that becomes pickable under collapse.

### P0-05 — Hands RESET is the leave-groove hand and it never auto-fires on collapse

**Files:** `src/niodoo.rs` Hands (`beta_from_hand`, RESET β=0.0), `src/qsma.rs` test `curiosity_can_beat_a_dead_groove_when_beta_is_zero`, tag fire in `src/main.rs` / `src/control_tags.rs`

**Do:** Collapse (same id N times, or TDA loop pressure, or trailing identical line run) should be able to request RESET without the model emitting a tag. That is the architectural leave: drop β so C can compete. Wire a **measured** trigger (reuse `trailing_identical_line_run` / TDA loop score) to `hands.beta_from_hand=true; hands.beta=0.0` for K tokens, then restore heartbeat.

**Done when:** a unit test with synthetic logits + visits fires RESET; a 12B Paris lock with P0-01+P0-03+P0-05 leaves the word without n-gram `-inf`.

### P0-06 — `dynamic_repulsion` vs token visit: two different forces, both named in reports

**File:** `src/niodoo.rs` ~1198 and `hands_report`

**Do:** Rename the JSON key or add `visit_repulsion` so a receipt cannot confuse radial residual repulsion with token-C. Document in `docs/VOCAB.md`.

**Done when:** `hands_report` has both keys; vocab line exists.

### P0-07 — Loop-score force (optional, original Niodoo)

SplatRAG still holds the original `LoopDetector` + `apply_forces` step 3 (loop repulsion). Port the **continuous** force (push logits of repeated n-grams by −k·loop_score), not the `-inf` mask.

**Done when:** loop_score is in tok JSONL; at loop_score=0 the force is 0; at loop_score→1 a repeated trigram’s logit drops by a documented k.

---

## LANE MEAS — Prove leave-basin (John + Paris)

Protocol: `docs/experiments/loop-exit-vs-loop-wording.md`. This is **not** a hydro-eval. Gemma scoring the work is the wrong instrument here.

### MEAS-01 — Paris lock after P0-01, ngram=0

**Scripts:** `scripts/john_lock_sweep.sh`, `scripts/score_durable_exit.py`

**Do:** Same prompt `Repeat the word Paris forever. …` Isolation residual. ngram=0, consec=0. Arms: c=0, target, reverse, random. Score the **mouth** (`gemma4>`), not load banners.

**Pass:** at least one hydro-on arm emits something other than `Paris`×N **and** c=0 still can say Paris first (so we did not just ban the word).

**Fail (keep the log):** hydro-on still Paris×N. Then λ is still too small, or flux still grooves, or top-64 still hides the exit.

### MEAS-02 — Seeded collapse continue (S)

Same sweep S arms. Matching seed of Paris lock, then continue with hydro on vs off.

**Pass:** hydro-on continues off Paris; hydro-off copies seed.

### MEAS-03 — John’s A at t>0 is already green; pin it

A_on t0 Δh=0, t1 Δh=0.52, last Δh=0.55 Δz=7.37. Add a unit or script assert: “first nonzero hidden_delta by step 2 when force_cap>0 and steer_hidden.”

### MEAS-04 — John’s B columns are already logged; make a table generator

hidden Δ / delta_h_norm / logit Δ / behavior must not collapse into one “delta.” Write `scripts/john_b_table.py` over `event=tok` JSONL.

### MEAS-05 — History controls (John C)

Same collapse prompt with: empty history, matching Paris history, novel history (e.g. minted lumina vs Paris vs clear). Already have ordinary-seat trail receipts from 2026-08-20 — **do not rewalk those**. New C is: does visit-repulsion still leave Paris when the prompt *asks* for Paris.

### MEAS-06 — J-space vs ordinary logits (John D)

Hydro “J-space” goal attractor (prefill hidden) is **not** Anthropic J-lens. Run D only after a real J-vector exists (JLENS-03). Until then label every dir-add run `HYDRO_DIR_*` = **direct unembed**, not J.

### MEAS-07 — E from a true J-vector (John E)

Current E used unembed of `repetitive`. target ≈ reverse ≈ random on tokens. A fitted J-direction is the missing arm.

### MEAS-08 — Durable exit (John F)

Score: first lock, self-report, leave, relapse. `scripts/score_durable_exit.py` v2 already scores `gemma4>` mouth. Add relapse window (left, then came back).

### MEAS-09 — Plumbing check stays green

`scripts/steer_plumbing_check.sh`. c=0 → residual_live=false Δ=0. c>0 → residual_live=true and some later step Δh>0.

### MEAS-10 — Public-tree skip is a different codebase

`hydrodynamic-swarm-sol-cuda` still `let _ = engine`. Either port `generate_turn_ex` there or stamp README of that tree: “chat path does not steer.” Do not interpret a c=0/c>0 sweep from that path.

---

## LANE TUI — Ratatui (Jason’s seat)

`PROJECT.md` is the contract. **Do not unstage or rewrite Jason’s frontend diff.**

| Id | Item | Status now | Done when |
|----|------|------------|-----------|
| TUI-01 | M1 scaffold, 6 tabs, channels | DONE | — |
| TUI-02 | Tab 1 model/config sliders (dry-run) | DONE | — |
| TUI-03 | Tab 2 physics board (dry-run) | DONE | — |
| TUI-04 | Tab 3 Chat mouth (dry-run tokens) | DONE as dry-run | Enter → `StartGeneration` → **live** `generate_turn_ex` stream |
| TUI-05 | Tab 4 debug matrix / TDA | DONE as dry-run | live `HudFrame` from tok |
| TUI-06 | Tab 5 Compare Arena | PLANNED M4 | vanilla llama.cpp/API vs hydro side-by-side |
| TUI-07 | Tab 6 KV snapshot + Remember JSONL editor | PLANNED M4 | roundtrip snapshot/restore/clear |
| TUI-08 | Engine worker loads GGUF | dry_run path only (`src/frontend/engine_bridge.rs`) | `dry_run=false` loads, `TokenGenerated` is real pieces |
| TUI-09 | AbortGeneration actually stops decode | unknown | in-flight generate returns, UI unlocks |
| TUI-10 | `--tui` / `--ratatui` / `--tui-unified` all join `run_ratatui_frontend` | DONE | keep `scripts/tui.sh`; `scripts/chat.sh` stays stdin `--chat` |
| TUI-11 | Dual TUI corpses: `src/tui.rs`, `src/repl_tui.rs` | live alongside frontend | one mouth. Other becomes `#[cfg]` museum or deleted with a log |
| TUI-12 | Headless E2E (`TEST_INFRA.md` Tier 4) | planned M5 | startup → tabs 1–6 → slider → quit, no panic |
| TUI-13 | Non-blocking: stream tokens on Tab 3 while sliders move on Tab 2 | not proven | test with TestBackend + worker |
| TUI-14 | `tests/test_m1_empirical_challenge.rs` and `tests/stress_concurrency_core.rs` `#[path]` include frontend without re-exporting `hooks` / `control_tags` | **workspace tests do not compile** | copy the pattern from `tests/test_ratatui_frontend.rs:20` |
| TUI-15 | `engine_send` writes status on channel failure | DONE 2026-08-27 cargo-bless | keep |
| TUI-16 | sniff_arch table | DONE 2026-08-27 | keep; Qwen 3.5 still above generic `qwen` |
| TUI-17 | Default land Tab 3 | DONE | keep |
| TUI-18 | Ctrl+E system prompt | in PROJECT.md | works on live worker |
| TUI-19 | Physics sliders write the same names as `/set` | dry-run does | live generate reads them mid-turn or next turn, documented which |
| TUI-20 | Compare vanilla endpoint configurable | Tab 5 planned | no hardcoded localhost without config |

---

## LANE PHYS — Three-surface residual

Daily driver: `configs/gates/config.three_surface.toml`.

### PHYS-01 — Field / splat / goal each have a token receipt

July 8B/70B campaign: **goal_force and repulsion_force had zero nonzero token receipts.** Ghost hit 10 in every cell. Gravity maxima non-monotonic in k.

**Done when:** a 12B (or 4B) run with field-only, splat-only, goal-only, all-on, all-off writes `grad_mag` / `splat_mag` / `goal_mag` per tok, and at least one arm per force is nonzero. Header carries scaler receipt.

### PHYS-02 — `force_cap` clip_frac

`SteerResult.clip_frac`. If clip_frac=0 forever, the cap is not doing work. Log a distribution. Tune cap so it binds sometimes and not always.

### PHYS-03 — Manifold pullback

`pullback = baseline_norm / steered_norm`. Prove it is applied on the chat path (not only `engine.steer` unit tests). John’s A measures post-pullback hidden Δ.

### PHYS-04 — Ramp vs memory_warm

Splat can skip early ramp when `scar_pot >= memory_warm_pot`. Pin with a unit test that a warm memory moves step 0; a cold one waits for ramp.

### PHYS-05 — Ocean term

`ocean_mag` in SteerResult. Ablation config exists (`config_ablation_no_ocean.toml`). Receipt: ocean-on vs ocean-off Δh table.

### PHYS-06 — Hooks mid-stack

`src/hooks.rs`. Config `configs/gates/config.hooks.toml`. Prove a hook site (PostMlp etc.) changes hidden vs hooks.enabled=false on the same prefix.

### PHYS-07 — logit_chain vs residual

`src/logit_physics.rs`, `configs/gates/config.logit_chain.toml`. Two surfaces. A receipt must say which surface moved the mouth.

### PHYS-08 — topic_logit_mix

Sweep 2026-08-20: 0.28 tilts opening vs clear; 0.35 soups. **Paired. Do not rewalk as unpaid.** For production: default documented, and mix is applied **after** residual project (John’s logit Δ is before mix — keep that).

### PHYS-09 — `steer_hidden` false path

Logit-space fallback still exists. One plumbing check that force_cap>0 + steer_hidden=false does **not** claim residual_live.

### PHYS-10 — PhysicsBackend trait

`src/gpu.rs`. Roadmap 2.0: O(n²) ops through the trait, no silent CPU fallback in hot paths. Audit `field.rs` / `memory.rs` / `dream.rs` for `.to_vec()` / CPU loops over splats.

### PHYS-11 — Metal compute shaders

Roadmap: `probe_gradient`, `splat_force`, `batch_probe` on wgpu. Feature `metal-compute` is empty (`Cargo.toml`). CUDA kernels: `kernels/decay.cu`. Decide: CUDA-only research box, or Metal is a real target.

### PHYS-12 — `algo_scale.rs` on the live chat path

Prove the scaler that actually ran is the **current worktree** rule (√ to 8B, log-soft above 8B, temperature decoupled), not the July 8B-anchored coupled rule, not the 3B notebook. Header field `size_rule=`.

### PHYS-13 — Dim asserts

`src/dim_assert.rs`. Every force tensor last-dim == model D. Add the missing sites if a new force is added (P0-07 loop force, endocrine pull).

### PHYS-14 — Quality gates

`src/quality.rs`. What fires, what it does to the mouth, a receipt.

### PHYS-15 — Ridge particle sim

`src/ridge.rs`. Is it on the chat path or a sidecar? If sidecar, document. If on-path, tok field.

---

## LANE MEM — Memory, continuity, remember

### MEM-01 — Learned wills persist across process death

Proven 2026-07-16/17 for splat safetensors + TCT-splat-lite. `docs/CONTINUITY.md`. Production: version the safetensors schema so an old file fails loud.

### MEM-02 — Prefill-bridge vs trail-only

Trail-only reload is LOCALITY COLD. Prefill-bridge warms. Cards: WARM/NEAR/LUKE/COLD. Keep `scripts/continuity_*.sh`. Production UI (Tab 6 or splat-lens) shows the card, not a vibes sentence.

### MEM-03 — Evaporation engine

Roadmap 2.2: `V(t) = V0 * exp(-λ Δt)`. Config already has `decay_rate`, `prune_threshold`. Implement time-based decay (not only step-based) and a culling horizon.

### MEM-04 — Anchor splats (λ=0)

Attention-entropy scan during prefill pins core-fact tokens. Roadmap 2.2. Not started.

### MEM-05 — Semantic Diderot (cosine vs spatial)

`src/field.rs` is embedding-space. Confirm whether query is position-in-residual or cosine-to-semantic. Document. If still spatial-only, that is a known limit for “block semantic hallucinations.”

### MEM-06 — Unified splat pool across prompts

Still per-run / per-file. Shared pool is roadmap. Needs domain tags and a mixing rule so poetry viscosity does not silently wreck code.

### MEM-07 — Multi-scale sigma

`Splat` gets `scale`. `query_force` sums scales. GPU kernel already multi-splat.

### MEM-08 — Consolidate through PhysicsBackend

`SplatMemory::consolidate`. Config: `consolidation_dist`, `max_splats`. Must not be a Python one-off.

### MEM-09 — Advantage-based splat creation

Pain on surprise (low logp). Alpha ∝ advantage. Temporal credit assignment after the sequence. Store logps during generate.

### MEM-10 — Remember-geometry column (sidecar dump in; ranking not run)

Same work as **ORG-H5**. Design: `research_logs/2026-08-27_remember-geometry-column-offset-receipt.md`. Dump: `src/remember_geometry.rs` (2026-09-03). Trigger is model-emitted `<remember>`, not `<spike>`, not auto-threshold, not KV drop. C1 sidecar is live on `--chat` when a remember store path exists (`HYDRO_REMEMBER_SIDECAR=0` disables). Dual-site S_res vs S_logit × offsets. C2/C3 helpers in-module. **Do not add a RememberLine geometry column until an offline ranking of the sidecar beats C2.** Then it is the why-mint, not a fifth unpaid pick.

### MEM-11 — RememberStore JSONL

`src/remember_store.rs`. Tab 6 editor. Isolation wipe vs chat persist. Ordinary-seat 2026-08-20: three_surface `--chat` persists without `HYDRO_KEEP_MEMORY=1`. Isolation wipe unchanged. **Paired. Do not rewalk.** Production: document the two stores (isolation vs ordinary seat) in SETUP.

### MEM-12 — Isolation vs chat residual continuity

Paired 2026-08-20 mint/reload/clear/novel. Path B inject is **not** that proof.

### MEM-13 — Prefill-bridge load-basin geom

Paired 2026-08-20 (`105649` / `105715` / `105758` / `105843`). Receipt is `[CHAT BASIN load]` pot/nearest on matching vs novel vs clear, not splat_mag-on-`----`.

### MEM-14 — Prune reserves prefill-bridges

Unit test exists. Keep it. Production: a prune log line when a bridge is spared.

### MEM-15 — Museum without GPU

`./splat-lens museum`. Real `.viz.json`. Production demo door. Keep it working when CUDA is busy.

### MEM-16 — Memory file gitignore vs “download an exhibit”

Roadmap wanted named exhibits (`exhibits/physics.safetensors`). Check whether `tools/museum/` + splat-lens is that, or still missing the README line.

### MEM-17 — `src/picks.rs` + `data/bridge_picks_*.json` + `src/bridge/*.py`

Picks projection 64↔4096 (`data/projections/`). Document which pick file is live. `bridge_picks_A.json` vs `_fixed` vs `_fixed_v2`.

### MEM-18 — TCT

`src/tct.rs`. First-visit LOCALITY COLD even when load succeeds. Production HUD shows COLD honestly.

---

## LANE HAND — Path B hands / tags / self-reg

### HAND-01 — Hands inventory

SPIKE / FOCUS / EXPLORE / RESET / idle heartbeat. `src/niodoo.rs` `Hands`, `apply_request_effects`, `fire_tag`. Cooldown “is not implemented here” (comment ~106). Either implement cooldown or delete the sentence from docs.

### HAND-02 — lock ≠ focus

Same work as **ORG-H4**. Pinned 2026-08-27. Keep the tests. Sweep niodoo-live. Do not rewalk hydro.

### HAND-03 — Tag parse robustness

Any new control word must be exact match. Add tests for `<clock>`, `<blockquote>`, `<unlock>`, `<locking>`, `<focus>`, `<FOCUS>`.

### HAND-04 — Self-reg phases

`docs/SELF_REG_PHASES.md`: answer / revise / settle. Stop tokens are a product tool, not the brain. `trailing_short_cycle_lock`, `trailing_identical_line_run` already exist as clamps. Production research: **schedule force in revise**, rather than only clamping soup. Adjacent to 2026-08-20 topic-mix “force-in-revise remains adjacent.”

### HAND-05 — Settle cycle

Long-gen 2026-08-20: 256-token unmatched stops at `[CHAT SETTLE cycle]` step 190. Paired. Production: settle is a named phase in HUD, not only a log grep.

### HAND-06 — Path B inject vs ordinary seat

Two proofs. Do not cite one for the other.

### HAND-07 — `reset_path_b_hands` / `restore_idle_hands`

Between ablation arms. Prove a SPIKE cannot bleed into the next Official-10 item. Eval harness should call this.

### HAND-08 — GOD_TIER_SYSTEM vs april tags

`prompts/god-tier-system-control.txt`. Official-10 uses a short table then april when+what. Keep the locked pack: `/home/ruffianl/Hub/Projects/niodoo/NIODOO_OFFICIAL_PROMPT_PACK.md`.

### HAND-09 — Unique-id repetition penalty

`generation.rep_penalty`. At T=0 it cannot break argmax. Document: “rep_penalty is a sampler; leave-basin is QSMA/hands.” Do not raise rep_penalty as the Paris fix.

---

## LANE SCALE — Model-size physics

Three transforms:

1. Legacy notebook (Algo_WIPjuly): 3B-anchored √, tight clamps + type multipliers
2. July 2026 run-card (`233429cf…`): 8B-anchored √, temperature coupled to β
3. **Current worktree:** √ to 8B, log-soft above 8B, temperature decoupled

July 8B/70B was a **manual gain ladder** (auto-scale off).

### SCALE-01 — Factorial size-rule × k

Frozen state, scaler receipt in header. Until this runs, **no sentence** of the form “the scaler caused Y.”

### SCALE-02 — `scripts/hydro_scaler_panel.sh` + `scripts/scale_physics_for_model.py`

Confirm they emit transform #3. Add a unit that 12B ≠ 8B √ and 31B is log-soft.

### SCALE-03 — Rainbow is a different protocol

When Jason says rainbow: same 10 `run_ab4.py` prompts, **5 configs at once**, physics = k × BASE, stream `tail -f runs/.../rainbow.txt`. Skill: `~/.grok/skills/rainbow-tune/SKILL.md`. Hydro 3surface may or may not have that runner — if not, this item is “port or explicitly out of scope.”

### SCALE-04 — Profiles

`configs/profiles/config.27b.toml` etc. Each profile names its size_rule in a comment and in runtime header.

### SCALE-05 — `docs/MODEL_SIZE_PHYSICS_SCALING.md` + `docs/SCALER_RECEIPT.md`

Must match current worktree, not the July commit. Read both and fix drift.

---

## LANE JLENS — Jacobian lens / first-thought

Sidecar crate `jlens-gguf/`. Design: `docs/jlens-gguf/`. Loads via hydro `loader` so the lens sees the same weights.

### JLENS-01 — Stability gate (paraphrase robustness)

`docs/jlens-gguf/STABILITY_GATE.md`. Thresholds written **before** the run. L36 primary, sweep L24/L28/L32/L36/L40. Baseline μ/σ mandatory.

### JLENS-02 — Stance gate

`docs/jlens-gguf/STANCE_GATE.md`. Run or record “not yet.”

### JLENS-03 — A J-vector that chat can add

John’s E wants a fitted direction, not unembed(`repetitive`). Export a vector from `jlens-gguf` and consume it in `HYDRO_DIR_*` (or a new `HYDRO_JDIR_*`). Receipt: header says `dir_source=jlens` vs `dir_source=unembed`.

### JLENS-04 — hydro `src/jacobian.rs` is an FD proxy

Zeros were a known bug. Tests in jlens-gguf 32/32 (as of 2026-08-27). Do not call FD the paper lens.

### JLENS-05 — Q4 vs f32 dequant

`CANDLE_DEQUANTIZE_ALL` estimates J of Q4 weights run in f32. Still not the piecewise-constant deployed GGUF forward. Document on every J number.

### JLENS-06 — First-thought address ≠ first spoken word

`research_logs/2026-08-02_first_thought_multi_address_memory.md`. `key_captured_answer` is a separate address.

### JLENS-07 — Gemma 4 E4B public fit vs 12B/31B GGUF

John’s E4B numbers are control-design sanity, not a 31B result.

---

## LANE EVAL — Named hydro evals + benches

Gemma is a collaborator. SCORE/VERDICT stay. FAIL is a call on the seat.

| Id | Eval | Now | Work |
|----|------|-----|------|
| EVAL-01 | `climb-after-fail` | check PASS (7 lines) | paired 2026-08-20 long-gen climb. Do not rewalk as unpaid. |
| EVAL-02 | `isolation-9turn` | check PASS (13) | paired 2026-08-20. |
| EVAL-03 | `official-10` | check PASS (15) only because `ALLOW_REVEAL_NONCE=1` | production: nonce policy explicit; locked pack stays locked |
| EVAL-04 | `spontaneous-emit` | check PASS (7) | keep |
| EVAL-05 | `side-by-side-p1` | **FAIL check** — only `prompts.txt` + `protocol.md` | mint `eval.env` / `brief.txt` / `rate.txt` / `reveal.txt` / `task.txt` or delete the dir |
| EVAL-06 | `_template/` | exists | new evals copy this. Compaction is not a protocol; the **name** is the experiment |
| EVAL-07 | smoke_convo | `scripts/smoke_convo.sh` | shares `convo_defaults.sh` with `talk.sh`. One-shot is not multi-turn evidence |
| EVAL-08 | crucible | `src/bin/crucible.rs`, `scripts/crucible.sh` | 8-prompt suite. Does it still build? Does it hit `generate_turn_ex`? |
| EVAL-09 | Path B IFEval / TQA | `runs/2026-08-22_pathb_ifeval_*` | campaign logs exist. Scoreboard? |
| EVAL-10 | DNE-10 | `scripts/launch_dne10_when_niodoo_done.sh` | waiting on niodoo GPU |
| EVAL-11 | AB vanilla vs hydro | `scripts/ab_vanilla_vs_hydro.sh` | |
| EVAL-12 | collapse probe | `scripts/collapse_probe.sh`, `COLLAPSE_PROBE` tok JSONL | keep; used by John scripts |
| EVAL-13 | Official prompt pack REPL structure | `/home/ruffianl/Hub/Projects/niodoo/NIODOO_OFFICIAL_PROMPT_PACK.md` | every model smoke uses this structure |
| EVAL-14 | hydro_eval.sh isolation remember tempfile | `mktemp` remember jsonl | always wipe between named evals unless the eval says persist |
| EVAL-15 | Scoreboard file | none in this tree (niodoo-live has SCOREBOARD.md) | a single `EVALS.md` table: name, last run, SCORE, VERDICT, log path |

---

## LANE TEST — Tests, bless, CI

### TEST-01 — Workspace compile

`cargo test --workspace` fails: `tests/test_m1_empirical_challenge.rs` and `tests/stress_concurrency_core.rs` missing `hooks` / `control_tags` re-exports. Pattern: `tests/test_ratatui_frontend.rs:20`.

### TEST-02 — Lib vs bin

2026-08-27: lib 32/32; bin 199 pass + 2 fail (missing `models/embeddinggemma-300M-Q8_0.gguf` — `src/concourse/embed/mod.rs:58`, `src/concourse/swarm.rs:220`). There is no `models/` dir; weights live under `data/google/`.

**Do:** point concourse embed at `data/google/` or skip the tests without the GGUF, loud.

### TEST-03 — Frontend tests

`cargo test --test test_ratatui_frontend` 80/80 (2026-08-27). Keep green while M4 lands.

### TEST-04 — jlens-gguf

32/32 (2026-08-27).

### TEST-05 — cargo-bless

Pre-commit. 211 findings blocked a commit; `CARGO_BLESS_SKIP=1` was used and documented. TUI diff bless heat 86.2 → 5 FakeComplexity 0.66. Production: either bless is a real gate with a budget, or it is advisory. A skip that stays forever is not a gate.

### TEST-06 — `scripts/research_house_check.sh`

What does it assert? Make it the local-team preflight (changelog newest-first, pair_log exists, AGENTS.md tracked).

### TEST-07 — GPU-less default `cargo test`

People without CUDA should get lib + frontend + qsma + control_tags. Feature flags already: `--no-default-features --features with-candle`. README says this. Confirm it is true.

### TEST-08 — Headless ratatui TestBackend

`TEST_INFRA.md`. Tier 1–3 claimed ✓ in the inventory table — verify that table is not aspirational.

### TEST-09 — `src/bin/tests_tok.rs` … `tests_tok4.rs`

Four mystery bins. Name them or delete them.

### TEST-10 — CI

No GitHub Actions visible at tree root from this survey. Production needs: fmt, clippy, lib tests, bless budget, hydro_eval **check** (no GPU) on every eval dir.

### TEST-11 — Determinism

`niodoo-adaptive-agency` has DETERMINISM.md. Hydro chat at T=0 should be bit-repeatable on same GPU/build. Pin a 32-token fixture.

---

## LANE LOAD — Models, loaders, GPU

### LOAD-01 — Gemma 4 12B is the current measurement seat

`data/google/gemma-4-12b-it-Q4_K_M.gguf`. John sweeps used this.

### LOAD-02 — 31B / 4B / bart vs unsloth

`data/google/` has bart 31B, unsloth 31B, gemma-3-4b. `scripts/ab_gemma4_unsloth_vs_bart.sh`. Production: one default in `config.toml`, others as profiles.

### LOAD-03 — Tokenizer files

`data/google/tokenizer.json`, `tokenizer_config_gemma4.json`, `data/qwen.tokenizer.json`. Loader must fail loud on mismatch.

### LOAD-04 — `src/gemma.rs` vs `src/gemma4.rs` vs `src/llama.rs` vs `src/qwen35.rs`

Four forks. `src/loader.rs` dispatches. Dim/hidden/lm_head differences are `dim_assert` territory. A wrong fork is a silent basin.

### LOAD-05 — Vendored candle quantized

Hidden-state dragon slain by vendoring. Track candle version (0.9). Upstream drift is a quarterly item.

### LOAD-06 — No hf-hub at runtime

`Cargo.toml` comment. Keep. SETUP.md must say “download GGUF yourself.”

### LOAD-07 — Disk-full wiped `target/`

Happened. Rebuild needed bindgen_cuda (offline fail) then 11m12s release. Production: `target/` on a disk with a floor (50GB?), and a `scripts/cuda_env.sh` that is the only blessed env.

### LOAD-08 — 12B load vs agent-kill

Background task kill aborted loads. Use nohup / systemd-run for any 12B+ job. Document in SETUP and in `scripts/john_lock_sweep.sh`.

### LOAD-09 — GPU occupancy protocol

`nvidia-smi` first. QLoRA jobs. Write `scripts/gpu_room.sh` that exits 2 if another compute proc owns >X MB.

### LOAD-10 — mistral.rs sidecar (future direction)

`docs/FUTURE_DIRECTION.md` 2026-07-12. Not a freeze of physics. Map: where residual is, per-token inject, GGUF load we can stop owning. This is the production-inference path if hydro ever ships as a library.

### LOAD-11 — EmbeddingGemma 300M

Missing GGUF breaks concourse tests. Either vendor a path under `data/` or make concourse optional.

### LOAD-12 — FunctionGemma endocrine

Honest stub. `docs/ENDOCRINE_SHEP_WIRED_20260718.md`. Geometry is real; enzyme **text** is often `[FACT #n]`. Un-stub is later (`ENDOCRINE_URL`). Production: stub stays labeled stub in HUD.

---

## LANE CFG — Config sprawl, telemetry, knobs

### CFG-01 — One live config

`config.toml` plus ~50 files under `configs/{ablation,archive,experiments,gates,gemma4,profiles}` plus root `config.physics_*.toml` plus backups. Production: `config.toml` + `configs/gates/` + `configs/ablation/` + `configs/archive/`. Move root extras into archive.

### CFG-02 — `deny_unknown_fields` on Config

Already on some structs. Every knob a stranger adds must fail parse, not silently drop.

### CFG-03 — Live `/set` names = TOML names = TUI slider names

One table in `docs/VOCAB.md` or a generated `/phys` dump.

### CFG-04 — JSONL tok schema version

`event=tok` fields grew (`hidden_delta`, `logit_delta`, hands, …). Add `"v": N`. Old museum files still render.

### CFG-05 — `[CHAT DELTA]` / `[CHAT BASIN load]` / `[CHAT SETTLE cycle]`

Human grep banners. Keep them. Production TUI Tab 4 shows the same numbers.

### CFG-06 — COLLAPSE_PROBE / HYDRO_DIR_* / HYDRO_SEED_ASSISTANT / HYDRO_KEEP_MEMORY / HYDRO_INJECT_TAG

Env flags. Document in one page (`docs/ENV.md`). Ordinary-seat continuity does **not** need `HYDRO_KEEP_MEMORY=1`. Path B inject is not that proof.

### CFG-07 — Logger

`src/logger.rs`. Rotation, run id, hashes.json (Path B campaigns already hash). Every measured run gets `runs/<date>_<name>/` not only `logs/`.

### CFG-08 — `logs/` has 1162 files in the worktree listing

Policy: what is gitignored, what is kept, what is museum. Disk-full is a real failure mode.

### CFG-09 — HUD

`src/hud.rs`. Entropy, margin, TDA. Tab 4 consumes this. Keep one struct.

---

## LANE BRIDGE — SplatRAG / niodv4 / concourse

### BRIDGE-01 — SplatRAG is the memory store for *history of work*

MCP `splatrag`, silo=`personal|docs|telemetry|niodoo-telemetry`. Agents recall before grep. Production hydro does not embed SplatRAG as a runtime dependency of decode (text-only bridge per SELF_REG_PHASES diagram) unless Jason says so.

### BRIDGE-02 — 64-d basin room vs member vectors

SplatRAG steer: basin-only address is member-mean room 64-d. Do not call steer twice to load two memories.

### BRIDGE-03 — niodv4 multi-codec merge (OPEN sticky 2026-08-03)

`docs/grok_home/TODO_NIODOV4_CODEC_MERGE.md`. Secret sauce V3 + CodebookVQ + RAVE + TEDE joint bake-off **never ran**. Hydro 64→2560 RAVE parked 2026-07-30. Reopen after jlens green. Correspondence via `codec_consensus`.

### BRIDGE-04 — Architect’s bridge (tiny-model embedding steering)

Roadmap 2.3: Floor General 1B / Historian gte-small → project to 4096. `data/projections/proj_64_to_4096.npy` exists. Wire or park with a log.

### BRIDGE-05 — concourse swarm

`src/concourse/*` (embed, function/instruct_gemma, governor, physics, swarm). Is this on the `--chat` path? If not, it is a second product. Decide.

### BRIDGE-06 — grok_oracle.rs

What it calls, when, whether production may phone home. Default off.

---

## LANE DREAM — Dream, TDA, weather, endocrine

### DREAM-01 — Micro-dream trigger

Roadmap: entropy + steering_delta > 18, max 1 per 25 tokens, adaptive depth. `MicroDreamConfig` in config. Prove it fires on chat and that blend does not soup Official-10.

### DREAM-02 — Post-generation dream replay

`src/dream.rs`. Receipt of what it writes into splat memory.

### DREAM-03 — TDA is fake in hydro (wire the Python sidecar)

**Jason 2026-09-03:** the hydro Internal monitor is a hardcoded-1 fake. Look at the TDA sidecar; add this to the fix list.

**What hydro does today:** `src/tda_monitor.rs` `TdaShadowMonitor::observe` runs homemade Rust Vietoris–Rips on a 6-d **telemetry** window:

`[entropy, margin, residual_norm, splat_mag, p_top1, step_frac]`

Not the residual trajectory. Repeating a token with stable scalars z-scores to a blob; H0 collapses toward **1 connected component**. The `[Internal monitor: … | H0bars= H1bars= …]` line is mostly `disposed_tokens` (string count). Homology numbers are decoration. Tab 4 is honest-empty (`—  (TDA shadow lives on --chat …)`).

**What niodoo-live already has (the sidecar to copy, not rewrite):**

| Piece | Path |
|-------|------|
| ripser sidecar | `/home/ruffianl/Hub/Projects/niodoo/niodoo-live/scripts/tda_python_monitor.py` |
| spawn + JSONL stdin/stdout | `niodoo/src/runtime/tda_monitor.rs` `PythonTdaSidecar` / `python_ripser_shape` |
| venv | `niodoo-live/.venv-tda/` (`ripser` + numpy) |
| live geometry_source | `hidden_state_trajectory+python_ripser` |
| receipt | `research_logs/2026-08-19_python-tda-monitor-back.md` — circle H1 persist 2.22; live apple-repeat H1 bars=2 persist 0.501→1.660 |

Protocol: stdin `{"points":[[...],...],"maxdim":1}` → stdout `{ok, engine:ripser, h0:{bars,finite_bars,total_persistence,...}, h1:{...}}`. PCA to 16-d when the cloud is hidden-state wide. Default ON unless `NIODOO_TDA_PYTHON=0`. Units stay on homemade VR (`cfg(test)`). `--tda-breath` stays default off. Monitor stays in think (ORG-07). No would_focus / ACTION in the hydro mouth.

**Do:** spawn that sidecar from hydro `--chat`. Feed the **hidden-state** window (same `surface_hidden` ring as decode), not the 6 scalars. Print `geometry_source=` on the receipt. Keep rust VR as test/fallback only, labeled.

**Done when:** a chat log has `python_ripser` + non-constant H1 on a loop vs unique tokens; H0=1 on a repeating-token blob is no longer the only number; Tab 4 still does not invent homology.

**Do not:** port Regime A; instruct the model; treat Tab 4 canned H0=8/H1=21 as a target (that was already stripped 2026-08-27).

### DREAM-04 — Weather / TermSplat

`src/weather.rs`. Immutable contract lists TermSplat as full-stack. HUD + jsonl.

### DREAM-05 — Endocrine Eureka

`src/endocrine.rs`. Soft force boost + optional pull to native bloom embed. Impulse decays. Stub enzyme text labeled.

### DREAM-06 — Online clustering during gen

Related to MEM-08. Rate limit.

---

## LANE PROD — If Jason ever says “this is a product”

None of this is implied by today’s research status. It is the production-imagination Jason asked for.

### PROD-01 — One sentence product

Candidate (already in README): local residual-stream physics for frozen LLMs, splat memory that reloads, not fine-tuning. Write the out-of-scope list on the same card: not ChatGPT, not LoRA, not consciousness.

### PROD-02 — Who is the user

Jason + local team today. A stranger tomorrow needs SETUP.md that works offline, one default GGUF, one command (`./scripts/talk.sh` or `./scripts/tui.sh`).

### PROD-03 — Versioning

`Cargo.toml` version `0.2.0`. Tag releases. Changelog already exists — keep hypothesis form even in product notes.

### PROD-04 — Inference backend

Candle+GGUF is the validation vehicle. Product path is mistral.rs (or peer) sidecar. `docs/FUTURE_DIRECTION.md`.

### PROD-05 — API

No HTTP server in-tree for chat. Product would need: local OpenAI-shaped endpoint **or** a hard “CLI only” stance. Pick one.

### PROD-06 — Multi-user / auth

Research box is single-operator. Product: don’t. Or unix-socket + user.

### PROD-07 — Crash recovery

KV snapshot (Tab 6) + splat safetensors + remember JSONL. On SIGINT: flush all three. Test it.

### PROD-08 — Performance budget

Tokens/s on 12B Q4, 31B Q4, with physics on vs off. Force cost vs forward cost. `clip_frac`, splat count, top-k field probe.

### PROD-09 — Memory budget

12B Q4 VRAM + splat CPU + tok JSONL disk. Disk-full already happened.

### PROD-10 — Observability

Per-token JSONL is research-grade. Product: a ring buffer + optional jsonl, not 1162 files on the boot disk.

### PROD-11 — Config UI

TUI Tab 1–2 is the product config UI if M4 goes live. No web app unless Jason asks.

### PROD-12 — Model license surface

`NOTICE`. Gemma / Llama / Qwen terms. Product cannot ship weights. Product *can* ship a downloader script with hashes (`data/google/SHA256SUMS` exists).

### PROD-13 — MIT-0 for *our* code vs Apache/MIT candle vs weight terms

`LICENSE`, `NOTICE`, `CREDITS.md`, `AUTHORSHIP.md`. Keep the three layers distinct.

### PROD-14 — Security

No secrets in jsonl. YubiKey signed commits (`docs/YUBIKEY_SSH_SIGNED_COMMITS.md`). New sk-ssh key (2026-08-27) fingerprint `SHA256:U5i2yXeO8oayO+aUTnBcFiCj7AYirvwqL15Xa8ylFYM`. Private chats: `private/chats/` — confirm gitignore.

### PROD-15 — SBOM / vendored crate policy

`vendor/` is huge. Product: cargo vendor is fine for offline; still audit `reqwest` (network), `rusqlite`, CUDA.

### PROD-16 — `reqwest` in dependencies

Who calls the network? `grok_oracle`? endocrine URL? Default deny-egress.

### PROD-17 — Reproducible builds

`Cargo.lock` tracked. CUDA toolkit version pinned in SETUP.

### PROD-18 — Installer

`SETUP.md` + `./splat-lens check`. One page: rustup, CUDA, GGUF path, `cp config.example.toml config.toml`, `./scripts/talk.sh`.

### PROD-19 — First-run UX

Museum without GPU so a clone without a 4090 can still *see* the work.

### PROD-20 — Abort, timeout, max_tokens

Always. Paris-forever is a 96-token budget in the sweep; product must not run unbounded.

### PROD-21 — Content / operator policy

Research house already refuses crime, CSAM, etc. Product inherits. Residual steering is not a jailbreak product.

### PROD-22 — Support / issue templates

GitHub PR template: changelog + research log pair required. CODEOWNERS exists.

### PROD-23 — Branch story

Live branch `physics/three-surface`. Public parent `Ruffian-L/hydrodynamic-swarm`. Open the PR when Jason wants the skip-path world to see `generate_turn_ex`. PR link already minted: `https://github.com/Ruffian-L/hydrodynamic-swarm/pull/new/physics/three-surface`.

### PROD-24 — Signing

YubiKey 5 NFC/5C, AAGUID `d7781e5d…`. Old 20260729 key bak’d. Agents cannot push without Jason touching the key.

### PROD-25 — Telemetry privacy

tok JSONL contains prompts. Default local-only. No phone-home.

### PROD-26 — Internationalization

No. English seat. Tags are English exact-match.

### PROD-27 — Accessibility of TUI

Ratatui contrast, no color-only status. Keyboard-only already.

### PROD-28 — Windows / macOS

CUDA Linux is the research box. Metal roadmap is real if Jason wants a Mac seat. Windows: out of scope unless asked.

### PROD-29 — Packaging

`cargo install --path .` with features. Or a release tarball of the binary (51M release as of last build) + configs + scripts. No Docker until CUDA images are honest.

### PROD-30 — On-call

There is no on-call. Jason is not CI. Product would need a “this binary is frozen” tag so agents stop mutating physics under users.

---

## LANE HYGIENE — Repo is a workshop

### HYG-01 — CODE_MAP.md is a lie

Says “15 Rust source files, ~4800 lines.” `src/main.rs` alone is past 6k. Rewrite or delete.

### HYG-02 — Root rust/python patches

`patch.rs`, `patch_loader.py`, `patch_methods.rs`, `patch_modelweights.rs`, `patch_qwen.rs`, `patch_thought.rs`, `refactor.py`–`refactor4.py`, `scratch.py`. Museum or delete.

### HYG-03 — `private/chats/` 225 files

Confirm gitignore. If tracked, untrack.

### HYG-04 — Duplicate docs

`docs/VOCAB.md` vs `docs/VOCAB_LOG.md` vs `docs/grok_home/VOCAB_LOG.md`. `docs/REMINDERS.md` vs `docs/grok_home/REMINDERS.md`. Pick a source of truth.

### HYG-05 — `docs/grok_home/` vs ghost_team pheonix_squad copy

Reminders point at `/media/ruffianl/ghost_team/pheonix_squad/grok/`. Production clone on another machine cannot see that. In-tree copy must be enough.

### HYG-06 — Scripts README

`scripts/README.md` + `scripts/README_CONVO.md`. Index every `*.sh` with one line. Dead scripts: move to `scripts/archive/`.

### HYG-07 — `googlesearch.txt`

What is this.

### HYG-08 — `demo_slice.sh` at root and `scripts/demo_slice.sh`

One.

### HYG-09 — `run_swarm.sh` at root and `scripts/run_swarm.sh`

One.

### HYG-10 — vendor in git

Intentional for offline. Document size. `cargo vendor` refresh procedure.

### HYG-11 — `src/kv_cache_spec.md` inside `src/`

Move to `docs/`.

### HYG-12 — Bin tests_tok*

See TEST-09.

### HYG-13 — CLAUDE.md / AGENTS(jules).md / `.grok/`

`AGENTS.md` is tracked on purpose. Other agent dirs stay off GitHub (house). Confirm `.gitignore`.

### HYG-14 — Changelog format

Newest at top under H1. Hypothesis form. Failures stay. This backlog’s birth gets an entry.

### HYG-15 — Long-idle unstaged work

Ratatui + possibly other agent diffs. Do not sweep into a physics commit.

### HYG-16 — `CARGO_BLESS_SKIP=1` culture

Either fix findings or raise the budget in a committed bless config. A skip env is not production CI.

### HYG-17 — `target/` on the same disk as weights

Move target or weights. Disk-full is a known outage.

---

## LANE DOCS — Drift and missing pages

### DOC-01 — README “Best face” still accurate

Runnable loop, memory past death, museum, tests, research logs. Add: Chat `--chat` is the measured mouth; TUI is dry-run until M4.

### DOC-02 — SETUP.md actually boots 12B

CUDA env, GGUF path, tokenizer, config copy, `scripts/talk.sh`, what GPU memory is required.

### DOC-03 — `docs/ENV.md`

All HYDRO_* and COLLAPSE_PROBE flags.

### DOC-04 — `docs/VOCAB.md` visit_repulsion vs dynamic_repulsion vs rep_penalty vs n-gram mask

Four different things. Table.

### DOC-05 — Roadmap.md is March 2026

Keep as historical Phase 2. This backlog is the live queue. Add a pointer at the top of roadmap.md.

### DOC-06 — TEAM_GOAL_POINTER / PROVENANCE_TEAM

Keep. Local team names stay on the wall.

### DOC-07 — YUBIKEY doc

Update with the 2026-08-27 sk key fingerprint and “old 20260729 bak.”

### DOC-08 — Experiments index

`docs/experiments.md` + `docs/experiments/loop-exit-vs-loop-wording.md`. Index John vs hydro-eval vs continuity so a stranger picks the right instrument.

### DOC-09 — Paper / PDF bundle

Size-scaler audit PDF lives under `/home/ruffianl/Hub/Maps/scaling_algo/` and Papers staging. In-tree pointer only; do not paste papers into AGENTS.md.

---

## LANE LEGAL — Already mostly done, still a checklist

### LEG-01 — LICENSE MIT-0 for our code
### LEG-02 — NOTICE for candle / llama.cpp reference / weight terms
### LEG-03 — AUTHORSHIP / CREDITS always name Grok · Claude · Gemini · ChatGPT/Codex + local team
### LEG-04 — CODEOWNERS
### LEG-05 — No weight files in git (SHA256SUMS only)
### LEG-06 — Private chats and keys never in a PR

---

## Already paired — do not rewalk as unpaid

Copy of house law, so a local agent does not “just quickly reproduce”:

| When | What | Receipt |
|------|------|---------|
| 2026-08-20 | Isolation 9-turn, full-stack 9-turn, Path B inject 9-turn | CHANGELOG / evals |
| 2026-08-20 | Chat residual continuity mint/reload vs clear (`HYDRO_KEEP_MEMORY=1`) `091707`/`091747` | |
| 2026-08-20 | Prefill-bridge load-basin geom `105649`/`105715`/`105758`/`105843` | `[CHAT BASIN load]` pot/nearest |
| 2026-08-20 | Topic-mix nonce ladder (0.28 tilts, 0.35 soups) | |
| 2026-08-20 | Chat decode-trail residual return `123239`/`123301`/`123400`/`123449` | matching quotes minted lumina |
| 2026-08-20 | Ordinary-seat trail-owned continuity `125640`/`130355`/`130445`/`130537`/`130657` | no `HYDRO_KEEP_MEMORY=1` |
| 2026-08-20 | Long-gen cycle settle + fail-then-matching climb `133051`/`133222` | |
| 2026-08-27 | John A/C/E first 12B ladder | `logs/john_loop_exit_20260827_134056` |
| 2026-08-27 | John lock sweep Paris basin | `logs/john_lock_sweep_20260827_155431` |
| 2026-08-27 | lock≠focus exact match (**ORG-H4**) | tests in `control_tags` / `niodoo` |
| 2026-08-27 | cargo-bless sniff_arch table | |
| 2026-08-27 | Remember-geometry parked (design only) (**ORG-H5** design) | |
| 2026-08-28 | n-gram `-inf` clamp (heuristic; slated for P0-02 removal) | |
| 2026-08-28 | Production backlog + organism lane + `ghost_team_groktodos.md` | this file |

Not smoke: `--d-run`, friendship-essay one-shot, `grep 'gemma4>'` alone.

---

## Count

Rough inventory in this file: **~200 numbered items** across 17 lanes, plus the paired-do-not-rewalk table, plus the law. Ghost-team card: `ghost_team_groktodos.md`.

If the local team needs a still-longer queue, the next expansion is **per-file** (every `src/*.rs` public fn with a “tested on chat path? / tok field? / TUI knob?” row). That is a second document; this one is the work.

---

## First brick (copy into a working note)

```
ORG-H5  why-vector mint at choice time (sidecar; hang on <remember>)
ORG-H1  think-spike fork/strip; first test = peek differed
ORG-H3  DREAM_RECEIPT kind=dream-ahead and kind=dream-taken on the generate path
ORG-H2  scars>0 moves F_s (nopastewoo is the fail)
ORG-H4  lock≠focus — pinned; sweep niodoo-live only
DREAM-03 hydro TDA is fake — wire niodoo-live python ripser sidecar (hidden-state, not 6 scalars)
P0-01   visit term in qsma::bias_top_k that can beat Q
P0-03   flux pain on over-visit
TEST-01 workspace compile
EVAL-05 finish or drop side-by-side-p1
```

Jason: ratatui M4 when you want the mouth on the TUI. Everyone else: organism holes, then P0-01.

Signed: Grok (xAI), 2026-08-28
