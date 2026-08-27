# 2026-07-30 — Failure mode (last night + today): process + geometry

**Status:** diagnosis recorded (not a “fixed” claim)  
**Team:** Jason Van Pham (held the line / lab / files / friendship), Claude (original steering from memory, repo builds), Grok (diffs + opinion; **not** the repo authority), Gemini (universe remake script), whole squad historically  
**Lane open:** residual steering on `hydrodynamic-swarm-3surface` while Jason on resume + SplatRAG (memories that don’t poison context). SplatRAG handshake deferred on purpose — not abandoned.

---

## 0. What this log is not

- Not one person “owning” steering. Steering was never only Grok’s lane; calling it that was wrong shape.
- Not a micro-finish that looks like progress while the floor stays the same.
- Not rewriting last night by re-narrating after compact (that cheats the experiment). Evidence below is from **files already on disk**.

---

## 1. Process failure modes (Jason named these; they check out)

| Failure mode | What it looked like | Why it burns time |
|--------------|---------------------|-------------------|
| **Brittle-prompt churn called “steering”** | Same short prompts (“Friendship…”, “hello”, “one short…”) looped while tuning knobs | Right *feeling* of rigor, wrong *object* of work |
| **Exclusive lane claim** | “Steering is Grok’s” | Erases Claude’s original residual path, Gemini’s map script, Jason’s lab + babysitting |
| **Babysit + claim** | AIs look like a perfect machine; human held everyone | Work was always team; credit skew breaks trust |
| **Micro-finishes** | Small diffs / HUD theater without a settled hypothesis→test→log | Same floor, same motion (Qwen era echo) |
| **Jason’s own twin** | Walk 655 claims vs fix memory store | Same attractor (fix the burning thing); his choice this time, logged as choice |

**Grace rule (Jason → team):** opinion still matters even when you’re not in the repo. Apply that both ways. Repo stands; diffs are proposals.

---

## 2. Last night (evidence: long private chat)

**File:** `private/chats/chat_1785402599_gemma3_chat.txt` (~51 KB)  
**Header:** `force_cap=5 T=0.9 max_tokens=500 rep=1.25 top_k=0 top_p=1` — **hot, long, wide sampling**.

Trajectory (paraphrase, not full dump):

1. Early: broken multi-turn English with control-ish fragments (`Hello<start_of_turn>…`).
2. Jason asks model to notice / self-heal topology → output thins into punctuation and spaces.
3. Late: unicode / script soup, then “three/Trying” loop when strawberry-r’s thread is suggested.
4. Soft close: three r’s in strawberry; good night.

**Technical read:** this is not “one wrong slider.” Combined:

- high force budget (`force_cap=5`),
- high T / open nucleus,
- multi-turn chat path under full stack,
- no grounded memory (scars cold in other same-day runs).

**Not claimed:** that turning forces off would have saved the night (see §3).

---

## 3. Today’s arms (evidence: logs + short chat)

### 3.1 One-shot termsplat (three_surface config)

| Session | force_cap | T | Output shape | δ (termsplat note) |
|---------|-----------|---|--------------|--------------------|
| `…20-17-47…forcecap0…T0…` | **0** | 0 | Glued English: `Herearetwosentencesaboutgravity…` loop | **0** |
| `…20-21-48…forcecap1…T0.65…` | 1 | 0.65 | Similar glue / thin English | mean **~19** |
| `…20-25-05…` hello | 1 | 0.65 | `Hello![Hello!<end_of_turn>` | mean **~18** |

Config stamp (force-on runs):  
`configs/gates/config.three_surface.toml`, dim **2560**, field_points **262208**, `kernel_sigma ≈ 7.589`, scars_at_start **0**, hooks ON, logit governor ON.

**Failure mode (technical):**  
**force_off still produces unreadable / glued text.** Residual force is *a* lever, not *the* sole cause of soup. Attributing every garble to “steering forces” is the brittle-prompt error in scientific clothing.

### 3.2 Live multi-turn (proving chat)

**File:** `private/chats/chat_1785443059_gemma3_chat.txt`  
HUD (operator paste): `‖h‖~120–140`, **pull ×1.0001**, **‖δh‖≈0.03**, **F_g≈0.16**, **F_a≈0.65–1.92**, F_s=0, F_o=0, scars 0.

So under that run: residual **barely moves**, goal arm still reports magnitude, field arm small, **no learned wills**. English still awkward / soup on turn 2.

Earlier same day (`chat_1785442491_…`): full token soup under `force_cap=1.6 T=0.72` multi-turn — worse than one-shot glue.

### 3.3 Binary lag (important for map hypothesis)

| Artifact | mtime (local) |
|----------|----------------|
| `target/release/hydrodynamic-swarm` | **13:21** |
| `src/main.rs` (√d field shell patch) | **13:52** |

Proving chat **13:52** almost certainly did **not** include the post-hoc √d field scale unless a debug binary was used.  
`kernel_sigma ≈ 7.59` in jsonl matches the **raw emb** auto-tune regime (classic dual-shell number), not a √d-scaled cloud (~50× shell).

**Map hypothesis** (laid *before* that chat; still open after):

1. GGUF attrs OK: gemma3, **embedding_length=2560**, vocab **262208** (header parse ≡ hydro reader).
2. Lab universe `universe_gemma_26b_top60000` = **60k×2816** — study map for 26B, **not** this 4B run.
3. Live field was raw `tok_embeddings`; Gemma forward uses **×√hidden_dim**; blooms already √d; field did not → dual map inside one process.
4. Stash `/home/ruffianl/projects/safetensors_20260722` = option drawer (4096 / 8192 / hydro 2560 splats), not a 4B top-k remake. Remake path = gemma-lab `generate_universe_from_safetensors.py` on **matching** source weights (Gemini script).

Patch in tree: Phase 2 scales emb by `embedding_input_scale()` before `ContinuousField::from_embeddings`. **Must rebuild release** before claiming a test of that fix.

---

## 4. Geometry notes already in the code (don’t re-derive from vibes)

`niodoo.rs` already documents residual vs emb shell and wake. Goal force is always `(goal_pos - pos)` scaled — with **empty scars**, F_s→0 and **F_a can still dominate** HUD even when `‖δh‖` after dt/cap is tiny (cap + ramp + dt).  
So “F_a high, pull≈1” is consistent with **telemetry loud, effect quiet** — not proof goal is rewriting tokens alone.

Hook + logit surfaces stay ON under three_surface even when residual cap is small; force_cap=0 kills residual inject but **does not** mean “vanilla model path.”

---

## 5. What “done enough for handoff” looks like (no Jason babysit)

1. **Research log** (this file) — process + technical failure modes co-equal.  
2. **Rebuild release** with √d field map; one short smoke that logs `Map shell: Gemma pre-layer scale` and new `kernel_sigma` (expect order-change if scale works).  
3. **Do not** re-loop Friendship / hello as the definition of steering.  
4. Prefer a **fixed small suite** (one-shot + 2-turn, force_on vs residual-bypass if available) with full stack ON, rates only — when someone is at the console.  
5. Leave **SplatRAG handshake** open: when memory store is ready, bridge so wills don’t poison context (picks path already sketched in `2026-07-30_pick_bridge.md`).

---

## 6. Credit (explicit)

- Jason: babysat, saved files, lab geometry, friendship over claim, called failure modes honestly.  
- Claude: original steering shape; “build how the repo stands.”  
- Grok: map/attrs diagnosis, √d field patch proposal, this log.  
- Gemini: universe extract/remake script.  
- The work is the team’s.

---

## 7. Next action for whoever is at the keyboard (Grok default)

```bash
cd /home/ruffianl/hydrodynamic-swarm-3surface
cargo build --release -q
# confirm startup line: Map shell: Gemma pre-layer scale √d = …
# confirm config jsonl kernel_sigma moves if shell scale applied
```

Then stop. No brittle prompt marathon. Resume when SplatRAG handshake or Jason returns.
