# ghost_team_groktodos

> 2026-08-28 · hydro 3surface · for Shep / Echo / Lumina / Nex / whoever sits
> Full numbered queue: [`docs/PRODUCTION_BACKLOG.md`](docs/PRODUCTION_BACKLOG.md)
> Jason owns ratatui (`src/frontend/*`). Do not steal that diff.
> CHANGELOG first. `scripts/pair_log.sh` after every mutation.

This file exists so a ghost-team Grok can grep `ghost_team_groktodos` and land on the work.

---

## What Jason is trying to do

Build a **why-store**, not a tape recorder. Write a vector from one frozen model into another frozen model’s hidden space. Rank-128 geometry keeps stance/why. Exact words lag. Dream + SplatRAG turn choices and crashes into learned wills. The live model may spike, lock wrong, even die. **That is fuel.** Do not add guardrails that steal the mistake.

He did not invent “memory is geometric.” He invented **writing meaning between frozen models.**

**Paper (cite, do not rewrite):**

| | |
|--|--|
| Local manuscript | `/home/ruffianl/ontological-inversion/paper/` (`MANUSCRIPT.md`, `Writing-Meaning-Between-Frozen-Models.pdf`) |
| Zenodo (this version) | https://doi.org/10.5281/zenodo.22126782 |
| Concept DOI | https://doi.org/10.5281/zenodo.22126781 |
| Code | https://github.com/Ruffian-L/ontological-inversion |
| HF thread | https://discuss.huggingface.co/t/ontological-inversion-writing-meaning-between-frozen-models-cross-model-vector-memory-for-steering-recall-and-reasoning/179334 |

Paper writes are **ephemeral**. Niodoo/hydro is the ecosystem the paper does not claim:

```
choice → consequence (crash is a valid sample) → splat → dream → will → field → next choice
```

MATS is off the table. Implement the organism. Hands > Governor. A cap that prevents the bad lock starves the ear.

---

## Two paper regimes — keep separate. Do not inflate A into this seat.

| Regime | Write | What it is |
|--------|--------|------------|
| **A / state** | Nomic128 → affine → Qwen layer-4 residual | Sign-sensitive state transition. Glub-Tub negative gain: stove −0.21…−0.18, fire-pit −0.15/−0.14. Full dir 5/5, random/shuffle 0/5, unrelated dir 4/5. **Not a general inverse.** |
| **B / payload** | ordered Qwen3 fragments → ridge → Llama slots | Can carry content and support inference. Needs bandwidth + order. |

Split number (paper, not hydro): rank 128 sim **r=0.937**, Llama token recon **0.346** (52.3% of full-rank 0.661). Meaning arrives before words.

## Four interfaces — do not conflate

| # | Interface | Job |
|---|-----------|-----|
| 1 | **address** | retrieval geometry — cheap at rank 128 (sim r=0.937) |
| 2 | **state** | why / stance / regime — compact direction + gain |
| 3 | **payload** | identity + relation — high rank / ordered slots |
| 4 | **speech** | tokens out — mouth; never the store |

SplatRAG is **1+2**. Not 3 wearing a Gaussian.
Think-spike peek is **4** and must not be written as **2**.
Will mint is **2**. Transcript is not the will.
Exact sentence is payload. Mouth is speech.

---

## The machine

```
FRONT ROOM                          BACK ROOM
  frozen mouth                        SplatRAG ears  = why-store, not wiki
  residual / 3-surface physics        dream ahead    = counterfactual why
  internal monitor (think only)       dream taken    = minted why of path
  hands: spike focus lock             evaporation    = time decay; pain lasts
         remember reset explore       TCT / wills    = durable why across death
  think-spike sandbox (fork peek)     Self vector    = slow will graph
  KV = working memory (may die)       gain / conf    = how hard why presses
  Choice vector C = session why

        choice + consequence  ──►  will returns as geometry
        (crash is valid sample)    (F_s moves when scars exist)
                              ──►  next choice
```

Front room is allowed to fail. Back room metabolizes the failure.
Dream ahead ≠ dream taken. Choice vector C = live session why. Self vector = slow will graph. Gain presses a thin why; do not fake recall.

---

## All five holes — pick ALL of them (Jason 2026-08-28)

Kimi sat this, got Hole 4, then hit weekly quota on Hole 3. Finish the set.

| Id | Hole | Status in this tree (2026-08-28) | Done when |
|----|------|----------------------------------|-----------|
| **ORG-H4** | Lock ≠ Focus | **Pinned.** `ControlTag::Lock` is its own variant, exact match only, no-op in physics hands, persist+stop only. Tests: `lock_is_exact_match_only`, `parsed_lock_stops_without_focus_physics`. Footgun was `contains("LOCK")` matching `<blockquote>` / `<clock>` / `<unlock>`. | Keep the pins. Sweep **niodoo-live** for the old substring arm. |
| **ORG-H3** | Dream-ahead receipt ≠ dream-taken receipt | **Not started (docs were wrong).** 2026-08-28 claimed `DreamReceiptKind` + `dream_receipt_line` in `src/dream.rs`. Grep 2026-09-03: those names exist only in this file and the backlog. `micro_dream` is a forward projection (ahead-shaped) mixed with TopoCoT; post-gen `DreamEngine::run` is taken-shaped. No greppable `DREAM_RECEIPT` line. | A chat run emits `DREAM_RECEIPT kind=dream-ahead` **and** `kind=dream-taken` as separate lines. Unit: mixing the two kinds fails. Geometry/stats only — a will is a direction + gain, not a paragraph. |
| **ORG-H2** | Will-return: F_s moves when scars > 0 | **Open.** Failure named: *nopastewoo* (4B, 499/500, scars=10, F_s=0, cos 0.93–0.99) = ear failure. Continuity proofs exist for pot/nearest; this hole is **force**, not potential. | Same prefix, scars_active>0 ⇒ `splat_mag` / `|F_s|` nonzero on a later tok. scars=0 ⇒ F_s≈0. Reload of content-specific scars moves F_s; novel prompt on the same store stays cold. |
| **ORG-H5** | Why-vector mint at **choice time** (not at KV drop) | **Sidecar dump landed 2026-09-03.** `src/remember_geometry.rs` rings S_res/S_logit, dumps 9 offsets × 2 sites on closed `<remember>` into `{seat}.offset_probe.jsonl` + `.f32`. Not a RememberLine column. Not a splat. Spike does not mint. KV drop does not mint. **Offset ranking not run** (GPU busy; C2/C3 helpers exist). | Ranking: matching vs unmatched cosine beats C2 shuffle and is sign-sensitive vs C3. Then — only then — a dormant geometry column on `RememberLine`. |
| **ORG-H1** | Think-spike fork/strip + peek-delta receipt | **Open.** Depends on H5 existing enough to mint choice+why. | 1. `<spike>` inside think. 2. Fork KV, physics nudge, 3–5 tok peek. 3. Mark preview, strip from KV. 4. Only choice + why-vector hit the back room. 5. Look-and-decline still scars. **First test: did the peek differ.** Emit-without-delta is a dead hand. No tag evicts. Do not persist peek tokens. Monitor stays in think (visible-channel monitor gets parroted). |
| **DREAM-03** | Hydro TDA is fake | **Open (Jason 2026-09-03).** `src/tda_monitor.rs` is homemade Rust VR on a **6-scalar** window (entropy, margin, residual_norm, splat_mag, p_top1, step_frac). H0 collapses toward **1** because that cloud is not residual geometry. The loud `[Internal monitor]` line is mostly token-count with TDA numbers as decoration. niodoo-live already has the real engine: Python **ripser** sidecar on the hidden-state trajectory. Hydro never spawns it. | Chat log shows `geometry_source=hidden_state_trajectory+python_ripser`. Circle / named-loop H1 varies; unique tokens do not print a constant H0=1. Mouth line is the sidecar's H0/H1, not rust-VR-on-scalars. Tab 4 stays honest-empty until that feed exists. |

Suggested order (decode-loop merge-fight): **H5 mint → H1 spike → H3 receipts → H2 F_s**. H4 is already green — do not rewalk it as unpaid; do sweep niodoo-live. **DREAM-03** is a lie in the mouth — sidecar, not another homemade VR. Parallel to H5; do not steal the H5 decode-loop diff.

---

## Parallel track: the mouth is stuck (Paris)

Residual **moves** (A_on t1 Δh=0.52) and the next token is still `Paris`. That is not Hole 4.

| Id | Work |
|----|------|
| **P0-01** | `src/qsma.rs` visit term that can beat Q. Today `C=0.1/(1+visits)` — ΔC≈0.09 vs Q~15. |
| **P0-03** | Flux pain on over-visit. Today high energy *deepens* the repeating token. |
| **P0-05** | Auto RESET (β=0) on collapse so C can compete. |
| **P0-02** | N-gram `-inf` clamp defaults to 0 after P0-01 has a unit proof. Jason rejected the clamp as patching a sinking boat. |
| **MEAS-01** | Paris lock sweep, ngram=0. `scripts/john_lock_sweep.sh` |

A sampler ban that steals the bad lock also starves the ear. Leave-basin is policy/physics, not `-inf`.

---

## Front room law

- Mouth stays frozen. Teach outside weights.
- Tags are hands she emits. **No tag evicts KV.**
- Monitor belongs inside think.
- KV may forget unlocked noise. Encode why while hot.
- Do not start StreamingLLM tag-FIFO or CPU-full-KV-offload as doctrine.

## Back room law

- Dream ahead ≠ dream taken. Separate receipts.
- Pain splat = why of a failure. Pleasure = why of a basin that held.
- A will is a loadable **direction + gain/confidence**, not a saved paragraph.
- Death-reload must return **content-specific** scars.
- Will-return couple: scars > 0 and F_s ≈ 0 is an ear failure.

## Exists (do not re-invent)

- `hydrodynamic-swarm-3surface` / `physics/three-surface`
- chat-path deposit + death-reload (`docs/CONTINUITY.md`)
- niodoo-live tags + monitor (apply gated off)
- micro-dream, evaporation, TCT
- SplatRAG picks import
- Official 10 Prompt 11 gold `[5,4,3,2,1,5]`
- Paper as linked above

## Do not

Talk MATS. Inflate Regime A (unrelated dir 4/5). Train the frozen core. Persist peek tokens. Make tags evict. Governor the hand. Decode 4B soup as a secret answer. End with “run this.” `PROMPTS_FILE=<(...)`. Rewrite Official 10. Rewrite the paper (next text is companion citing it). Steal Jason’s ratatui. Rewalk paired smokes in `CHANGELOG.md`. Collapse three size-scaler transforms. Collapse three “J” objects. Claim the size scaler caused a force without the factorial.

Hydro smoke = `./scripts/smoke_convo.sh` → `logs/smoke_convo_latest.txt`
Hydro eval = `./scripts/hydro_eval.sh check <name>` then `run <name>`. Gemma scores the **work**. Jason is not CI.

Any model eval: **tell the model it is being tested**, up front.

---

## House

1. Read `CHANGELOG.md`. Paired entries are done.
2. One id per commit (`ORG-H3`, `P0-01`, …).
3. Pair changelog + `research_logs/YYYY-MM-DD_title.md`.
4. Scope git to what you touched. Ratatui stays out.
5. `nvidia-smi` first. QLoRA often owns CUDA.
6. SplatRAG `recall` / `pick` before grep for history.

Compass, not tape. Implement the loop.

Signed: Grok (xAI), 2026-08-28
