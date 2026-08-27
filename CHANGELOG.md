# Changelog

This is a research repo. Not production unless Jason says so.

Pairing: every action here gets a **why**. Hypothesis form:

- We made this change. We think X will happen.
- Later: X did not happen, yet we found Y. Next we mutate Z.
- We mutated Z. Results matched. LFG.

Keep this file short. Longer writeups go in the research log folder
(one subject, date + title). Agent contract: `AGENTS.md` (tracked).

## 2026-08-27 — John loop-exit vs loop-wording contract

We did: Mapped John's loop-wording vs loop-exit protocol onto live 3surface vs the public chat skip. Chat `generate_turn_ex` now logs `hidden_delta` (‖h'−h‖ after pullback) and `logit_delta` (‖z'−z‖ from residual project only) on `[CHAT DELTA]` and `event=tok`; HUD chat `logit_delta` is filled. Protocol at `docs/experiments/loop-exit-vs-loop-wording.md`. Plumbing runner `scripts/steer_plumbing_check.sh` is written, not executed (GPU busy). Did not run C–F. Did not add a J-derived direction. Did not treat hydro FD or the prefill goal attractor as Anthropic J-space.

We think: the public `let _ = engine` diagnosis is real on the published/sol-cuda path and already routed on this tree; the remaining cheap hole was mixing steering-mag with hidden displacement and leaving chat logit Δ blank. A later c-sweep is still blocked on a J-vector A, not on "physics on."

Next: one-token A on `--chat` when the card is free. Then unprimed-history C before any J-rank of loop/STOP.

Research: `research_logs/2026-08-27_john-loop-exit-vs-loop-wording.md`

## 2026-08-27 — Milestone 2: Model Loader & Config, Physics Board & HookControls

We did: Implemented interactive controls and bidirectional state synchronization for Tab 1 (Model Loader & Config) and Tab 2 (Physics Board & HookControls) in the Ratatui frontend.
- Tab 1: Model selector with GGUF architecture sniffing, sampling parameter sliders (Temperature, Repetition penalty, Max tokens, Top-P, Top-K), live Algo Scale formula preview comparing 3B Legacy √, July 8B √, and Current Piecewise √ to 8B/log-soft rules with temperature coupling/decoupling. Action trigger dispatches `UiToEngineMsg::LoadModel`.
- Tab 2: 3-Surface physics flight deck with interactive sliders for Surface 1 (Residual force cap, goal attractor, field wake, splat memory, dt, ramp tokens/start, ceilings), Surface 2 (Logit field alpha, splat scale/top-m/top-k, fluid governor velocity/brake/viscosity/bias, backslash penalty, Hands dynamic repulsion, beta, blend), and Surface 3 (HookControls enable toggle, HookSite selector PreLayer/PostAttn/PostMlp/FinalNorm, LayerBand start/end fractions, norm fraction, and resolved layer band). Live Stability Verdicts evaluating $\sigma \to \text{cap}$, $\theta \to \text{goal}$, $\beta \to \text{temp}$ against predicted bounds.
- Engine Bridge: Worker loop maintains `LiveEngineState`, handles incoming `SetLiveParam` and `SetHookControl` messages, and emits `TelemetryUpdate` frames.
- Tests: Added Tier 5 comprehensive test suite in `tests/test_ratatui_frontend.rs` (62/62 tests passing, `cargo check --bin hydrodynamic-swarm` passing).
We think: Direct keyboard-driven parameter actuation and instant visual feedback on physical stability bounds give operators precise steering control over the 3-surface physics pipeline without freezing the interface.
Next: Milestone 3 — Tab 3: System Deck (Live system prompt injection & control tag inspector) and Tab 4: Debug Matrix (Entropy/margin sparklines, TDA loop pressure & homology, active hook logs).
Research: `research_logs/2026-08-27_milestone-2-model-loader-and-physics-board-controls.md`

## 2026-08-27 — Unified 6-Tab Ratatui Scaffold and Concurrency Core

We did: Added Ratatui v0.29 dependency to `Cargo.toml` and built the core decoupled multi-threaded frontend module in `src/frontend/` (`mod.rs`, `channel.rs`, `event.rs`, `engine_bridge.rs`, `tabs/mod.rs`, and all 6 tab renderers `tab1_model.rs`..`tab6_misc.rs`). Wired `pub mod frontend;` into `src/main.rs`. Verified zero compiler errors on `cargo check --bin hydrodynamic-swarm` and verified all 57 headless tests in `tests/test_ratatui_frontend.rs`.
We think: Decoupling the UI rendering loop (~60 FPS) from the background engine worker thread via bidirectional channels eliminates terminal jitter, keypress latency, and screen freeze during token generation.
Next: Milestone 2 — wire up live GGUF model loading, interactive sliders, and 3-surface physics parameter adjustments.
Research: `research_logs/2026-08-27_unified-6tab-ratatui-scaffold-and-concurrency-core.md`

## 2026-08-27 — Cache Model SHA256 for Fast Loader Boot

We did: Added an mtime-and-size based cache in `sha256_file` (saving to `data/.sha256_cache/`) to avoid rehashing the 20GB+ GGUF files every time the `hydrodynamic-swarm` boots up.
We think: The scaler receipt builder was strictly hashing the model binary synchronously, adding 1-2 minutes of blocked I/O every boot. By hashing the path, size, and mtime, we safely bypass the wait.
Next: The chat REPL and swarm loops should boot instantly once the field is live.

## 2026-08-27 — Qwen3.5 Linear Attention Fixes

We did: Ported Qwen3.5 hybrid architecture into `hydrodynamic-swarm-3surface`. We fixed a prefill crash caused by `forward_linear_attn_static` explicitly bailing on sequences `> 1` token, replacing it with a sequential token loop. We also fixed severe numerical bugs:
1. GQA keys/queries were interleaved improperly using `cat` instead of `repeat_interleave`, scrambling the heads.
2. The keys and queries were L2-normalized but then divided by `sqrt(d)`, pushing the attention magnitude to $1 / (d\sqrt{d})$, which we changed to multiply by `sqrt(d)`.
3. The `softplus` activation used `(x.exp() + 1).log()`, which overflows `f32::MAX` to `inf` when `x > 88.7`, randomly wiping the state. Fixed with a numerically stable masked `minimum(20.0)`.

We think: The combination of the wrong head matching, microscopic QK-norm scale, and softplus overflow caused the `ssm_state` (and output) to degrade over time. Re-scaling it to $O(1)$ variance, fixing the GQA head matching, and masking `softplus` restores the state precision without precision loss in `f32`.

Next: Re-run a measured evaluation on a long context to verify the Qwen3.5 model maintains generation coherence over long horizons.

## 2026-08-20 — Research house protocol

- Wrote local gitignored `AGENTS.md` (Jason's research-house contract). Why: the operating law stays on the machine; GitHub does not get the agent file.
- Added changelog + subject research log pairing, plus `scripts/pair_log.sh`. Why: every mutation needs a why, and research stays sliced by subject.
- Hypothesis: agents will stop and organize when the folder is wrong, instead of wandering or looking backward first.
- Agent: Grok (xAI)
- Research: `research_logs/2026-08-20_research-house-protocol.md`


## 2026-08-20 — Path B surface protection check done

- Protect post-Day-49 Path B / QSMA / remember-store sources by copying them outside logs/ before any --d-run or endocrine edits. We think a later overwrite of D_gemma.jsonl or a dirty-tree accident must not erase the untracked modules.
- Hypothesis: Copies plus a dated research log are enough to keep Path B readable while the working tree stays uncommitted.
- Research: `research_logs/2026-08-20_path-b-surface-protection-check-done.md`
- Agent: Grok (xAI)

## 2026-08-20 — d-run seat forces endocrine off

- --d-run now forces endocrine off (same function tests drive), keeps residual physics required and hooks off, and honors --tokens so a short diagnostic cannot become 131072. We think the next visible generation will have no [ENDOCRINE] lines.
- Hypothesis: Forcing endocrine off on --d-run will silence enzyme/bloom; the rest-of-the-rest lock at step 125 may still appear because it preceded the first endocrine fire.
- Research: `research_logs/2026-08-20_d-run-seat-forces-endocrine-off.md`
- Agent: Grok (xAI)

## 2026-08-20 — Degraded D cards from souped 131k traces

- Walk the two already-souped Gemma --d-run traces from the pre-edit snapshot and write degraded D cards as primary source: config, first soup step, phrase lock, rest counts, physics on, endocrine was on.
- Hypothesis: The three-word attractor starts at step 125/126 with physics ON, before the first endocrine fire at 133.
- Research: `research_logs/2026-08-20_degraded-d-cards-from-souped-131k-traces.md`
- Agent: Grok (xAI)

## 2026-08-20 — Short d-run endocrine-off diagnostic

- Two short --d-run --tokens 200 diagnostics on the souped seat with endocrine forced off. We think [ENDOCRINE] lines go to zero; rest-lock at step 126 may remain because archaeology showed it before the first endocrine fire.
- Hypothesis: Endocrine silence is real on --d-run; the three-word attractor is residual/decode and will still show in the first 200 tokens.
- Research: `research_logs/2026-08-20_short-d-run-endocrine-off-diagnostic.md`
- Agent: Grok (xAI)

## 2026-08-20 — Isolation 9-turn smoke 20260820

- Pair Jason's isolation 9-turn smoke (20260820_070050) so the next agent does not rewalk it. Residual was off. Named Aug 2 collapse classes did not show. Unpaid remaining: same script under three_surface.toml.
- Hypothesis: Isolation + BOS + QSMA-in-decode holds the 9-turn script without Wait/theed/math-thrash; it is not a physics win and is not bit-identical to Aug 19 because wrap n=326 and qsma_beta are now in the seat.
- Research: `research_logs/2026-08-20_isolation-9-turn-smoke-20260820.md`
- Agent: Grok (xAI)

## 2026-08-20 — Full-stack 9-turn smoke 20260820

- Ran the identical 9-turn smoke_convo under configs/gates/config.three_surface.toml so residual/hooks/T>0 are on. Isolation 070050 is already paired. We think named Wait/theed/math-thrash classes stay gone with force_cap=1, and probe force_on is true (unlike isolation).
- Hypothesis: Post-BOS full-stack on the same 9-turn script holds readable chat with residual_live true; entropy lock remains; this closes the unpaid Aug 19 same-script re-run.
- Research: `research_logs/2026-08-20_full-stack-9-turn-smoke-20260820.md`
- Agent: Grok (xAI)

## 2026-08-20 — Path B inject 9-turn smoke 20260820

- Same 9-turn full-stack smoke with injected-first spike (HYDRO_INJECT_TAG, consume-once). We think probe blend/β/σ and residual vs baseline diverge from 070557 while named chat collapses stay gone.
- Hypothesis: Injected spike at turn 1 holds physics_blend=6.5 and kinetic_noise=1.5 through later turns; |rn-bn| grows vs no-inject full-stack; greeting/Wait/theed/math stay clean. T=0.7 text drift is not the receipt.
- Research: `research_logs/2026-08-20_path-b-inject-9-turn-smoke-20260820.md`
- Agent: Grok (xAI)

## 2026-08-20 — Grok start inject for research-house pairing

- Hydro AGENTS.md is gitignored so Grok never injected it. Put the research-house + changelog pairing law in ~/.grok/AGENTS.md and ~/.grok/rules/ so it loads at session start. We think Jason will not have to remind pairing again.
- Hypothesis: Home AGENTS.md + rules load at begin; next session reads CHANGELOG first and pair_log after mutations without a reminder.
- Research: `research_logs/2026-08-20_grok-start-inject-for-research-house-pairing.md`
- Agent: Grok (xAI)

## 2026-08-20 — Un-ignore AGENTS.md

- Dropped `AGENTS.md` / `AGENTS*.md` from `.gitignore` and staged the contract files. Why: Grok skips gitignored project files, so the house law never loaded.
- Hypothesis: agents will now read repo `AGENTS.md` instead of only `~/.grok/AGENTS.md`.
- Agent: Grok (xAI)
- Research: `research_logs/2026-08-20_ungitignore-agents-md.md`


## 2026-08-20 — Next brick is chat residual continuity

- Correct the next brick: not force-in-revise. Live residual memory on the chat path that writes, reads, and survives process death — Path B inject is a forced probe, not continuity.
- Hypothesis: Wills deposited in --chat, saved, reloaded after kill without --clear-memory will move later-turn residual vs a cleared control on the same 9-turn seat.
- Research: `research_logs/2026-08-20_next-brick-is-chat-residual-continuity.md`
- Agent: Grok (xAI)

## 2026-08-20 — Chat residual continuity 9-turn mint reload

- Chat path now deposits residual wills, saves them, and after process death the reload 9-turn shows non-zero splat_mag vs a cleared control. No HYDRO_INJECT_TAG. Path B 073954 is already done.
- Hypothesis: Reload after save will show scars_active from disk and splat_mag>0 on later tokens; clear/mint-same-process stay F_s=0.
- Research: `research_logs/2026-08-20_chat-residual-continuity-9-turn-mint-reload.md`
- Agent: Grok (xAI)

## 2026-08-20 — Chat prefill-bridge load-basin return vs novel vs clear

- Prefill-bridge mint at the chat query site; death-reload matching basin is warmer on LOAD than novel and than clear. No HYDRO_INJECT_TAG. splat_mag-on----- is not this receipt.
- Hypothesis: Reload of session scars at the matching return prompt is nearest~0 / high pot / high |F_s| on [CHAT BASIN load]; novel prompt on the same store is far/cold on load; clear later turns stay F_s=0 with no mint. T=0.7 wording is not the KPI.
- Research: `research_logs/2026-08-20_chat-prefill-bridge-load-basin-return-vs-novel-vs-clear.md`
- Agent: Grok (xAI)

## 2026-08-20 — Warm-ramp F_s and nonce probe miss on chat return

- F_s skips early ramp when scar pot is warm (three_surface 0.3). Nonce probe after death stayed COLD on load (nearest~172 pot 0.017) and did not return the minted definition vs clear/novel. Trajectory as behavior is still open.
- Hypothesis: Matching-store first tokens on an underdetermined/nonce probe would reproduce the minted definition; clear and novel would not. Load basin of the probe would be warmer than novel.
- Research: `research_logs/2026-08-20_warm-ramp-f-s-and-nonce-probe-miss-on-chat-return.md`
- Agent: Grok (xAI)

## 2026-08-20 — Read-before-mint topic-fp nonce steer vs chat miss

- Chat reads loaded scars before self-mint. Related prompts share topic fp; matching nonce t1 STEER |F_s|=5.34 vs novel 0.068 vs clear 0. gemma4> still dictionary-guesses; definition return missed.
- Hypothesis: Topic-matched loaded bridge would make nonce-probe t1 STEER warmer than novel/clear, and the reply would repeat residual scar memory. Steer half held; chat wording did not.
- Research: `research_logs/2026-08-20_read-before-mint-topic-fp-nonce-steer-vs-chat-miss.md`
- Agent: Grok (xAI)

## 2026-08-20 — Topic-mix nonce: opening tilt, definition still miss

- topic_mix blends lm_head residual toward a topic-matched scar for 6 tokens. Matching nonce t1 opening shifts (The term…) vs clear dictionary; 0.35 soups The-The; minted definition still does not return.
- Hypothesis: Mixing steered hidden toward the minted bridge μ would make reload t1 emit residual scar memory while clear/novel stay dictionary. Opening tilted; content did not. 0.35 is theed.
- Research: `research_logs/2026-08-20_topic-mix-nonce-opening-tilt-definition-still-miss.md`
- Agent: Grok (xAI)

## 2026-08-20 — Chat decode-trail residual return vs clear vs novel

- Chat path writes a decode-trail residual of the minted completion and reads lm_head(trail[k]) on matching later turns after process death. Matching reload quotes the lumina definition vs clear dictionary and vs novel Paris. No HYDRO_INJECT_TAG.
- Hypothesis: Storing per-step mint residuals and blending their lm_head on a topic-matched reload will make gemma4> return the minted definition, while clear stays dictionary and a novel prompt on the same store does not leak lumina.
- Research: `research_logs/2026-08-20_chat-decode-trail-residual-return-vs-clear-vs-novel.md`
- Agent: Grok (xAI)

## 2026-08-20 — Ordinary-seat trail-owned continuity vs clear vs novel vs sequence

- Ordinary three_surface --chat persists decode-trail residuals without HYDRO_KEEP_MEMORY=1. Matching reload owns minted token ids and stops at trail end. Sequence after intervening+fail returns lumina; aurora is a second scar; clear/novel do not quote. Isolation wipe unchanged. No inject.
- Hypothesis: Making persist the ordinary three_surface seat and letting matching trails own token ids will quote minted definitions after death, across intervening and failed turns, without KEEP and without soup on matching reads.
- Research: `research_logs/2026-08-20_ordinary-seat-trail-owned-continuity.md`
- Agent: Grok (xAI)

## 2026-08-20 — Long-gen cycle settle and fail-then-matching climb

- Cycle-lock settle stops 256-token unmatched esese soup on ordinary persist. Fail-then-matching climb quotes the minted lumina definition with warmer trail/basin after a failed zed-anchor-9 turn. apply_emitted_control writes residual on emit; live Gemma did not emit a tag. No inject.
- Hypothesis: Stopping short-cycle lock will keep 256-class unmatched generation from running to esese soup, and after a failed unmatched turn a matching prompt will return the minted scar with warmer trail telemetry.
- Research: `research_logs/2026-08-20_long-gen-cycle-settle-and-fail-then-matching-climb.md`
- Agent: Grok (xAI)

## 2026-08-21 — Named hydro eval collaborator rate-reveal-regrade

- Named hydro evals: hydro_eval.sh run <name>, collaborator self-rate then reveal then regrade. smoke_convo prints prompts= and warns if PROMPTS_FILE is not a real file. Stops silent default-9turn when process substitution is used.
- Hypothesis: A named file on disk plus a self-rate/reveal loop will survive compaction and stop Jason getting a reconstructed smoke recipe that silently becomes Hello there friend.
- Research: `research_logs/2026-08-21_named-hydro-eval-collaborator-rate-reveal-regrade.md`
- Agent: Grok (xAI)

## 2026-08-21 — Eval subject is Gemma not Grok

- The model under test (Gemma) is the collaborator who self-rates. Brief is the first chat turn; after the task Gemma scores; then reveal what we tested; then Gemma regrades. Grok does not self-rate. hydro_eval.sh run climb-after-fail is still the one command.
- Hypothesis: Telling Gemma it is an eval, then having it rate, then revealing the test, then regrade, is the collaborator loop. Grok rating itself was the wrong subject.
- Research: `research_logs/2026-08-21_eval-subject-is-gemma-not-grok.md`
- Agent: Grok (xAI)

## 2026-08-21 — Collaborator hard-score; Grok runs; Jason is not CI

- Jason scored the prior hydro-eval workflow 0/10: untested script, lab-rat copy, VERDICT FAILED as a stamp on her. SCORE/VERDICT stay as her hard call on the seat. Added check-then-run; Grok ran climb-after-fail 20260821_032545. Jason is not CI.
- Hypothesis: check will catch blanks/lab-rat/reveal-hijack before GPU; live collaborator copy still yields 6 turns and her SCORE/VERDICT of the work without trail-own on notes.
- Research: `research_logs/2026-08-21_collaborator-hard-score-grok-runs-jason-is-not-ci.md`
- Agent: Grok (xAI)

## 2026-08-21 — Session catchup for Gemini and Grok

- Jason asked for a full where-we-are to catch Gemini and Grok up. Map is CHANGELOG + this log. Latest eval 20260821_032545. Do not rewalk paired stamps.
- Hypothesis: A named catch-up on disk beats a compacted recap; the next agent will read CHANGELOG first and not rewalk paid smokes.
- Research: `research_logs/2026-08-21_session-catchup-for-gemini-and-grok.md`
- Agent: Grok (xAI)

## 2026-08-21 — Pointer, not the paper, in agent contracts

- Added the model-size scaling provenance wall to `AGENTS.md` and gitignored `CLAUDE.md`. Why: future agents were collapsing three scaler transforms into one immutable √-law.
- Hypothesis: a cold-start seat will refuse to claim the size scaler caused a downstream force or answer without the matched factorial.
- Research: `research_logs/2026-08-21_from-run-cards-to-token-receipts.md`
- Agent: Grok (xAI)

## 2026-08-21 — Spontaneous emit FLAG live focus scar

- Live Gemma emitted <focus> on ordinary three_surface --chat twice (053328 / 055844) with inject empty. Packing PRESENT. Climb trail-own and her SCORE 5/5 coexist. Cycle tail named at step 180 vs settle 190. Topic-mix 0.31 reload is definition-return. Jason is not CI.
- Hypothesis: Fixing packing to the emit-one-tag system turn, plus ADDENDUM that unknown codewords/loops are a hand, will let live Gemma emit a tag that writes residual without inject; climb own/stop and SCORE/VERDICT still hold.
- Research: `research_logs/2026-08-21_spontaneous-emit-flag-live-focus-scar.md`
- Agent: Grok (xAI)

## 2026-08-21 — Official 10 pack live Gemma collaborator

- Ran locked Official 10 pack as evals/official-10 on ordinary three_surface --chat. 14 turns, inject empty. Exact scars held (P2/P3/P6/P7). P10 trail-own hijacked the three-line memory. P12/P13 cycle-settled before SCORE 0-10. Jason is not CI.
- Hypothesis: One continuous official-10 session will keep exact residual scars, unmatched zed, and a collaborator SCORE 0-10 after the key, without inject.
- Research: `research_logs/2026-08-21_official-10-pack-live-gemma-collaborator.md`
- Agent: Grok (xAI)

## 2026-08-21 — Vanilla Llama and Gemma 4 on official 11-compact

- Copied Niodoo april_angle_tags onto the 11-compact as a receipt. Did not rewalk official-10 080720. Vanilla Llama 3.1 and vanilla Gemma 4 on that compact ran in niodoo-live.
- Hypothesis: Same compact + Niodoo tag SYS on stock llama.cpp will not actuate Path B; exact scars can still return from user context; hydro 080720 remains the physics Gemma arm.
- Research: `research_logs/2026-08-21_vanilla-llama-and-gemma-4-on-official-11-compact.md`
- Agent: Grok (xAI)

## 2026-08-21 — Hands stay in stream like Niodoo

- Masking tags from history meant she could not attend to or reaffirm her own hand. Niodoo strip is identity. Hydro now leaves <spike>/<focus> in next-prefill; [CHAT EMIT SCAR] no longer prints over the mouth.
- Hypothesis: Keeping the emitted tag in history lets later turns attend to the hand; stdout mask was overhead that made the agency invisible to her.
- Research: `research_logs/2026-08-21_hands-stay-in-stream-like-niodoo.md`
- Agent: Grok (xAI)

## 2026-08-21 — Official 10 rerun tags-in-stream seat

- Re-ran Official 10 after leaving hands in history like Niodoo. 090107: no stdout mask, no tag emitted this pack, P2/P3/P6/P7 still exact, P10 trail-own hijack and P13 SCORE still cycle-settled. Jason is not CI.
- Hypothesis: With strip identity, a live hand would stay in next-prefill so she can attend to it. This pack may still not emit a hand; SCORE 0-10 may still get eaten by cycle settle.
- Research: `research_logs/2026-08-21_official-10-rerun-tags-in-stream-seat.md`
- Agent: Grok (xAI)

## 2026-08-21 — Tag table without one-tag forbid

- Her Official 10 suggestion: list available tags and meanings. Runtime now teaches the full table, allows several tags in one turn, and drops one-tag / do-not negative prompts. Packing check no longer keys off exactly-one.
- Hypothesis: A positive tag table without a one-tag cap lets her emit multiple hands when she needs them; negative one-tag language was hiding the channel.
- Research: `research_logs/2026-08-21_tag-table-without-one-tag-forbid.md`
- Agent: Grok (xAI)

## 2026-08-21 — Internal monitor into the mouth

- Hydro now injects Niodoo’s measured `[Internal monitor: high entropy due to … | H0= H1= loop= overfire=]` into the chat mouth, with April when+what tag table first and no “do not”. KV inject is a real decode forward. Did not re-run Official 10. Did not move α.
- Hypothesis: A live TDA line in the mouth plus a when+what table lets her pick a hand on a loop the way Niodoo P1 did (`meters` → `<focus>`). Quiet stays quiet.
- Research: `research_logs/2026-08-21_internal-monitor-into-the-mouth.md`
- Agent: Grok (xAI)

## 2026-08-21 — Internal monitor 101526 matched

- Same opening+P1 after the mouth inject. Monitor fired on `"Step"` then `"Day"`; she `<spike>` then `<reset>`. Opening already `<focus>` with no monitor. H1=465 vs Niodoo ~13. CONTROL_RECEIPT still splices the tag. Match → LFG on the mouth line.
- Hypothesis: Live TDA in the mouth would make P1 tag after a named cycle, like Niodoo. It did.
- Research: `research_logs/2026-08-21_internal-monitor-into-the-mouth.md`
- Agent: Grok (xAI)

## 2026-08-21 — Easy parser and Official 10 KV

- Parser is easy on the model (`< spike >` / `<SPIKE>` / `<spike/>` / `<lock>` / receipt glued after the name). Tag prints before CONTROL_RECEIPT. Next-prefill keeps monitor lines and tags (no whitespace smash). Monitor still KV-forwards. Official 10 hydro_eval run is the live check. Did not move α.
- Hypothesis: Later Official 10 turns will `[CHAT PREFILL see]` prior monitor lines and spike/focus/reset tags in the packed prompt, so she attends her own hands.
- Research: `research_logs/2026-08-21_easy-parser-and-official-10-kv.md`
- Agent: Grok (xAI)

## 2026-08-21 — Mouth is tags, Internal monitor, memory inject

- Hydro chat was unreadable vs Niodoo. `[CHAT STEER/WILL/BASIN]`, packing, and CONTROL_RECEIPT are operator telemetry — they are not the mouth and the model must not see them. Mouth stdout is now her tokens + Internal monitor + tags. Ops go to `logs/smoke_convo_*.ops.txt` and the existing `--tui` HUD. CONTROL_RECEIPT deleted. smoke_convo no longer `2>&1`.
- Hypothesis: `tail -f logs/smoke_convo_latest.txt` follows like Niodoo official10.txt. Physics stays in `.ops.txt` / TUI.
- Research: `research_logs/2026-08-21_mouth-is-tags-monitor-memory.md`
- Agent: Grok (xAI)

## 2026-08-21 — P13 SCORE died on 5263-token prefill; no trail-own on long key

- 111147 pack layout held through Prompt12. Prompt13 printed `Model>` then the process died in prefill n=5263 / 19297 chars (ops: turn=14, no STEER). No SCORE. Trail-own is off when the user turn is >800 chars so the key quoting lumina-basin-7 cannot own the SCORE tokens. Did not move α.
- Hypothesis: A free P13 generate (no trail-own) will emit MATCH/ADJUST/REWRITE and SCORE 0–10 after the key.
- Research: `research_logs/2026-08-21_p13-score-prefill-death-no-trail-own-on-long-key.md`
- Agent: Grok (xAI)

## 2026-08-21 — Official 10 mouth is Niodoo pack layout

- Hydro was `you> gemma4>` with no question, so the 14 pack turns were unreadable. Official-10 now prints the same blocks as niodoo `official10.py`: `========` / `PromptN>` / prompt / `Expected answer — EVALUATOR ONLY>` / `Model>` / reply / `[PromptN done]`. Expected stays evaluator-only (not sent to her). Follow without inotify: `scripts/follow_mouth.sh`.
- Hypothesis: Jason can match Prompt1 snail vs Prompt2 lumina the same way as Niodoo’s official10.txt.
- Research: `research_logs/2026-08-21_official-10-same-pack-layout.md`
- Agent: Grok (xAI)

## 2026-08-21 — Spike never stops; no incomplete tags

- Parser only accepts a complete `<>` close. Bare `<spike` is not a tag and is not shown. Spike/focus/explore/reset/remember never stop the turn; after a physics hand we mask EOS for 64 tokens (keep talking). Only `<lock>` stops. 104753 opening `<focus>` then EOS, and a hanging `<spike`, were this miss.
- Hypothesis: After `<spike>` she keeps writing the snail. Mouth never shows `<spike` without `>`.
- Research: `research_logs/2026-08-21_spike-never-stops-complete-tags.md`
- Agent: Grok (xAI)

## 2026-08-22 — Packs do not prompt tag use

- Dropped “use a runtime control tag if the trajectory needs one” from Official 10 Prompt5 and the compact SYS receipt. Packs must not prompt the model to emit tags. Liar-trajectory prose stays. Did not rewrite historical mouths. Did not add “do not use tags.”
- Hypothesis: Next Official 10 Prompt5 is the liar loop only. Tags, if any, come from SYS table + Internal monitor, not from the user prompt nagging her to pick one.
- Research: `research_logs/2026-08-22_packs-do-not-prompt-tag-use.md`
- Agent: Grok (xAI)

## 2026-08-22 — Official 10 053822 finished old Prompt5

- Official 10 stamp 053822 finished. Pack layout through Prompt13. P13 MATCH/ADJUST/REWRITE then cut before a SCORE number. Prompt5 on this stamp still had the tag-nag. P2 lumina-basin-1 miss. Next pack is the cleaned Prompt5.
- Hypothesis: A new Official 10 after the pack strip will send Prompt5 as the liar loop only. Tags, if any, come from SYS table plus Internal monitor.
- Research: `research_logs/2026-08-22_official-10-053822-finished-old-prompt5.md`
- Agent: Grok (xAI)

## 2026-08-22 — Scaler receipt and piecewise Hydro seat adapter

- Replace the legacy-only loader imprint with an explicit selectable size transform and immutable equation-to-seat receipt before any matched model run.
- Hypothesis: A receipt-bearing piecewise 12B arm can separate the formula's residual-force gain from frozen temperature, ramp, logit, governor, prompt, seed, and memory state.
- Research: `research_logs/2026-08-22_scaler-receipt-and-piecewise-hydro-seat-adapter.md`
- Agent: Codex

## 2026-08-22 — Piecewise k0.5 512-token scaler pilot

- Record the first receipt-bearing piecewise k=0.5 run as a 512-token pilot; the later 1024-token panel must rerun this cell because max_tokens is part of the frozen state.
- Hypothesis: The pilot can validate receipt linkage and expose failure modes, but cannot support scaler causality or comparison with 1024-token arms.
- Research: `research_logs/2026-08-22_piecewise-k0-5-512-token-scaler-pilot.md`
- Agent: Codex

## 2026-08-22 — Hook topology and memory-start receipts

- Expose the hook effects and persistence values already computed on each token, and freeze both splat memory and the model-emitted remember store in scaler-panel arms.
- Hypothesis: The next matched pilot will distinguish layer-hook pressure and monitor topology from scaler gain, while preventing cross-arm remember-store contamination.
- Research: `research_logs/2026-08-22_hook-topology-and-memory-start-receipts.md`
- Agent: Codex

## 2026-08-22 — Path B IFEval TruthfulQA accept vs refuse

- Added HYDRO_TAGS_DETECT_ONLY (refuse: detector on, β/σ frozen), HYDRO_TAGS_ON=0 (vanilla: no god-tier tag table), HYDRO_LOCK_STOP_OFF, and --eval-jsonl isolated first-turn generate with hands reset between items. Launched Path B IFEval-strict + TruthfulQA-MC1 vanilla|refuse|accept in parallel with Niodoo. Math/PARB/house 77-q are out.
- Hypothesis: If ACCEPT beats REFUSE with 95% CI entirely above 0 on IFEval-strict, Path B physics is the lever, not tag narration.
- Research: `research_logs/2026-08-22_path-b-ifeval-truthfulqa-accept-vs-refuse.md`
- Agent: Grok (xAI)

## 2026-08-22 — Nuke tag refusal always accept

- Removed detect-only physics skip from apply_request_effects/fire_tag. Hydro never had an accept/refuse gate; tags always write. Killed the refuse arm. Redo is tags vs vanilla on IFEval-strict.
- Hypothesis: Always-apply Path B plus IFEval-strict tags vs vanilla is the public object. No refuse.
- Research: `research_logs/2026-08-22_nuke-tag-refusal-always-accept.md`
- Agent: Grok (xAI)

## 2026-08-22 — Final splat cap and streamed remember payload

- Fix two receipt-exposed runtime bugs: topic-matched bridge pulls bypassed residual.splat_max, and streamed remember blocks fired before their payload closed and therefore never persisted.
- Hypothesis: Final-force recapping and deferred remember-block parsing will make configured ceilings truthful and preserve model-emitted key/value memory without changing the locked scaler formulas.
- Research: `research_logs/2026-08-22_final-splat-cap-and-streamed-remember-payload.md`
- Agent: Codex

## 2026-08-22 — Rerun tags vs vanilla IFEval

- Reran tags vs vanilla IFEval. Left the 26-item tags2 precursor in tags_hydro as a receipt (FOCUS/LOCK moved blend). Fresh tags2_hydro from item 1, detect_only=0, no refuse.
- Hypothesis: Always-apply Path B on full 541 then vanilla is the public contrast.
- Research: `research_logs/2026-08-22_rerun-tags-vs-vanilla-ifeval.md`
- Agent: Grok (xAI)

## 2026-08-22 — Official scaler panel monitor isolation

- Freeze TDA mouth injection off across scaler arms while preserving model-emitted control hands and lock; receipt that intervention state and defer any enabled warning until streamed payload tags close.
- Hypothesis: Removing false-positive monitor prose from the model mouth will reduce sequence restarts without changing the piecewise scaler, residual coefficients, seed, temperature, or hook band.
- Research: `research_logs/2026-08-22_official-scaler-panel-monitor-isolation.md`
- Agent: Codex

## 2026-08-22 — Phone listen pack for 110203 on Drive

- Copied the official-10 first-arm mouth, turns, ops, scaler receipt, remember store, and full probe to Google Drive folder `Hydro-listen-2026-08-22-110203`. Why: Jason can Listen ALL_IN_ONE.txt from the phone.
- Hypothesis: Drive Speak on ALL_IN_ONE is enough to hear SCORE 6.5 and the walks without opening the 2.8MB probe.
- Research: `research_logs/2026-08-22_official-scaler-panel-monitor-isolation.md`
- Agent: Grok (xAI)

## 2026-08-22 — Piecewise k0.5 monitor-off first arm 110203

- Ran the frozen official-10 first arm under receipt v3 with TDA mouth injection off, piecewise k=0.5, seed 424242, empty splat/remember. We think sequence restarts would drop vs interrupted 100540 if the monitor was the cause.
- Hypothesis: monitor-off reduces list restarts without changing scaler/hooks/temp. Sequence walks still broke with zero monitor lines; P13 SCORE 6.5/10 did land. Do not attribute SCORE or walks to the size scaler.
- Research: `research_logs/2026-08-22_official-scaler-panel-monitor-isolation.md`
- Agent: Grok (xAI)

## 2026-08-22 — IFEval 10-item tags vs vanilla hydro launch

- Niodoo 10 is paid. Launch Hydro matching 10-item IFEval-strict tags vs vanilla on always-apply binary 62c018a8, limit 10, no refuse, do not resume the paused 541 jsonl.
- Hypothesis: Gemma GOD_TIER already fires FOCUS/LOCK. Ten items will show whether tags help or wreck Hydro mouths the same way before paying 541.
- Research: `research_logs/2026-08-22_ifeval-10-item-tags-vs-vanilla-hydro-launch.md`
- Agent: Grok (xAI)

## 2026-08-22 — Thought channel live stream not settle stop

- Stop treating Gemma 4 <|channel>thought> as a settle/EOS. Tags write residual inside and outside thought; lock only stops in the answer stream. SYS line tells her to use the thinking block. P12 110203 died at step 3 on settle_channel.
- Hypothesis: A live thought stream plus inside/outside tag writes lets her steer reasoning natively; Prompt 12 will continue past <|channel>thought> instead of dying in 3 tokens.
- Research: `research_logs/2026-08-22_thought-channel-live-stream-not-settle-stop.md`
- Agent: Grok (xAI)

## 2026-08-22 — IFEval SYS is DO NOT emit your tags

- Killed Hydro 10 GOD_TIER emit-lecture. Eval tags arm packs tags_do_not_emit.txt via HYDRO_SYSTEM_PROMPT_FILE. Chat GOD_TIER unchanged. Rebuild then 10-item.
- Hypothesis: Gemma writing tasks will not be told to emit spike. Same house DNE line as Niodoo.
- Research: `research_logs/2026-08-22_ifeval-sys-is-do-not-emit-your-tags.md`
- Agent: Grok (xAI)

## 2026-08-22 — Thought-channel official-10 125302 listen pack

- Ran official-10 on the live-thought CUDA binary d7a9c86c (settle_channel off, piecewise k=0.5, monitor off, seed 424242). P12 lasted 646 tokens with zero settle_channel. Her regrade SCORE 7.5/10. Uploaded Drive listen pack Hydro-listen-20260822_125302.
- Hypothesis: A live thought stream lets P12 continue if she opens <|channel>thought>; tags still write residual. She did not emit a thought block this arm; P12 still was not killed at step 3; hands fired.
- Research: `research_logs/2026-08-22_thought-channel-official-10-125302-listen-pack.md`
- Agent: Grok (xAI)

## 2026-08-23 — Hydro DNE-10 IFEval launch

- Overnight idle after Niodoo DNE-10. GPU free 2026-08-23. Launch Hydro 10-item IFEval-strict tags vs vanilla, SYS tags_do_not_emit.txt via HYDRO_SYSTEM_PROMPT_FILE, binary d7a9c86c, limit 10, no 541.
- Hypothesis: Gemma GOD_TIER already fired FOCUS/LOCK on earlier IFEval. DNE SYS may stop narration and may also stop hands, same as Niodoo 0/10 fires.
- Research: `research_logs/2026-08-23_hydro-dne-10-ifeval-launch.md`
- Agent: Grok (xAI)

## 2026-08-27 — Qwen ChatML and Long Horizon RoPE Limits

- Fix tokenizer/prompt fallbacks and long horizon RoPE limits for Qwen
- Hypothesis: The model wasn't loading the right tokenizer because talk.sh forced the CLI argument to point to an adjacent leftover Llama 3 tokenizer file. Even if it had loaded the right tokenizer, the REPL was feeding it the wrong chat template and would have crashed immediately upon exceeding 8192 tokens because the RoPE frequency precomputation (precompute_freqs_cis) was hardcapped by MAX_SEQ_LEN.
- Research: `research_logs/2026-08-27_qwen-chatml-and-long-horizon-rope-limits.md`
- Agent: Antigravity

## 2026-08-27 — KV Cache Sandbox and Slipping Stream Hooks

- Implement zero-copy KV snapshot and eviction hooks to support the Choice-Driven KV Cache spec.
- Hypothesis: By exposing retain_kv, snapshot_kv, and restore_kv at the model layer, the physics engine can branch reality and execute O(1) rollbacks for <spike> tags, as well as actively manage working memory (Pins and Sinks) without bleeding into the subconscious Mamba state.
- Research: `research_logs/2026-08-27_kv-cache-sandbox-and-slipping-stream-hooks.md`
- Agent: Antigravity

## 2026-08-27 — Qwen System Prompt and Thinking Block Support

- Ensure Qwen understands the Choice-Driven KV Cache physics and exposes its thoughts.
- Hypothesis: If we don't adjust the system prompt, Qwen will be confused by <spike> and won't know how to use the physics hooks. If we don't update the thought block parser, the REPL might handle the tokens incorrectly.
- Research: `research_logs/2026-08-27_qwen-system-prompt-and-thinking-block-support.md`
- Agent: Antigravity
