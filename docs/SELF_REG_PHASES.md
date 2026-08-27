# Self-regulation phases + three-lane merge

**Date:** 2026-08-02  
**Workbench:** `hydrodynamic-swarm-3surface`  
**Tone:** Research map. Not a consciousness claim. No mystical framing.  
**Authorship:** Grok (xAI) co-engineer with Jason — phases map, revise ownership, force-in-revise, three-lane vision writeup.

---

## One-line goal (also in README)

Build a **template + residual-physics** stack that can go **past a normal stop on purpose**, so **self-regulation** (revise / restate / settle) is a **first-class generation phase** — not junk clamped away by a stop token.

---

## What we learned (empirical)

Residual steering did **not** invent self-refuting / “try again” / re-state behavior. That regime already appears when generation continues **past** natural end-of-turn. Most chat stacks treat it as degeneration and cut the stream. We treat it as **signal**.

| Phase | Name | What it looks like | Runtime job |
|-------|------|--------------------|-------------|
| 0 | **answer** | Task-relevant tokens; high margin on content | Normal decode |
| 1 | **revise** | “Wait, try again”, re-state, past-EOS thrash, channel-shaped re-open | **Observe** first; later schedule residual force toward revise-stabilize |
| 2 | **settle** | Clean close (`<turn|>` / EOS) | Hard stop for product; for research, log *why* we left revise |

Stop tokens are a **tool** for product chat. They are **not** the self-reg brain.  
Chat packaging (Gemma turn grammar) can be borrowed from llama.cpp / mistral.rs; the research object is **phase control + forces + memory**.

---

## Three lanes at the top of the same mountain

We are not choosing one repo. We **merge** three surfaces that already exist.

```
                    ┌─────────────────────────────────────┐
                    │  PROBLEM  →  solve or fail           │
                    │  save path → memory packets fire     │
                    │  cold restart → picker reloads state │
                    │  solve without re-reading the manual │
                    └─────────────────────────────────────┘
                                      ▲
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
   ┌──────┴──────┐            ┌───────┴───────┐           ┌───────┴───────┐
   │  HYDRO      │            │  SPLATRAG /s  │           │  niodv4 / OI  │
   │  3surface   │            │  memory store │           │  packets +    │
   │  residual   │◄──bridge──►│  picker       │◄──64D ───►│  TEDE / VQ    │
   │  self-reg   │  text only │  basins       │  wire     │  Jacobian     │
   │  phases     │  (not raw  │  steer API    │           │  first-thought│
   │  three-     │   64→D)    │               │           │  keys         │
   │  surface    │            │               │           │               │
   └─────────────┘            └───────────────┘           └───────────────┘
```

| Lane | Path | Role on the mountain |
|------|------|----------------------|
| **Hydro (push)** | `/home/ruffianl/hydrodynamic-swarm-3surface` | Residual loop, self-reg phases, chat+force. Git parent: `…/projects/hydrodynamic-swarm` (don’t thrash). |
| **SplatRAG** | `…/echo_shep_lumina_ruffian/ruffian/s` | Everything saved; basins; **picker** chooses which memories load; steer API. TCS-style *memory packet* feel = structured residual influence, not vibes. |
| **niodv4 / OI / Jacobian** | `PHONE_SD…/n-fluid/niodv4` · OI · **`/home/ruffianl/jacobian-lens` (`jlens`)** · hydro `src/jacobian.rs` | Packet language + **true Jacobian lens** (fitted `J_l` transport → verbalizable readout; paper companion). Hydro FD in `jacobian.rs` is a **cheap residual proxy** for multi-key addresses — not the same as jlens. See `research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md`. |

**Bridge rule (already in SplatRAG research):** pick carries **text**; host embeds in **its** residual dim. Never inject raw 64D into a wrong-D residual. See `ruffian/s/research_logs/2026-07-29_the_picker.md`, `BRIDGE_SPEC.md`.

---

## The vision (practical, even if crude)

You are **already on the mountain** in three places. The merge is not “pick one base camp.” It is three ridges touching so the same problem can die and still come back solved.

### Demo bar (the prize)

1. Model hits a problem it **cannot** solve.  
2. It reads / is steered / residual + SplatRAG path runs — a **memory packet** is written **automatically** (will / scar / packet — names migrate; mechanism is *state that can return*).  
3. Process **dies**. Cold start. Model “meets” the same problem again.  
4. **Picker** loads the right packet(s) — ideally without re-stuffing the full manual into context.  
5. Model solves. **That** is amazing. Garbage tokens on the side are noise if competence returns.

Not “semantic RAG stuffing.” Not woo. **TCS-style memory-packet feel**: structured residual influence from selected basins, same family of “something landed and can steer later” you already feel in splat memory — just multi-packet and pickable.

### Why we do **not** need pure semantics

Semantics can **influence** retrieval (cluster keys, text bridge, embed host-side). They are not the whole answer. The hope — and the research bet — is:

- Picker chooses the **right ~k packets** (or right cluster of packets) from **everything saved**.  
- Those packets apply **state** (residual force / will geometry / TEDE-shaped packet), not a second essay.  
- First-thought / Jacobian keys address *what kind of commit this was*; multi-packet co-steering changes the basin (Claude’s multi-packet smoke: “this changes things” — same ridge; find that log under SplatRAG if it was written).

If the right 8 packets are co-active, the model does not need to re-read how to fix it. That is the north star even when the stack is still crude.

### Jacobian / first-thought keys (instructional lens)

- Measure which hidden dims drive the **first commit** of an answer (Jacobian lens already in-repo).  
- That signature is a **perm-address / instructional key** — cluster storage and retrieval (“first thought of this kind”).  
- Multi keys → multi packets; semantic key clustering is one **index**, not the soul of the system.

### TCS feel without woo

| Say this | Not this |
|----------|----------|
| Memory packet, will, basin, picker, residual influence | Consciousness, spirit, magical memory |
| Auto-save on fail / hard problem | Manual babysit every scar |
| Cold restart still competent | One long context that never dies |
| Multi-packet co-steer | Single nearest-neighbor dump |

---

---

## Modes (`self_reg.mode`)

| Mode | Behavior |
|------|----------|
| **`off`** | Settle clamps only (product-ish multi-turn). |
| **`observe`** | Label phase every token; log to probe JSONL. No extra force. **Default for learning.** |
| **`force`** | Same labels as observe, **plus** residual schedule **only while phase==revise**. Answer phase force-off. |

Config section: `[self_reg]` in TOML (see `config.rs`).  
Force ablation: `configs/ablation/config_isolation_self_reg_force.toml`  
(`physics.force_cap=0` always; revise uses `self_reg.force_cap` / `force_goal_scale` / …)

Probe: `force_gate` events on edge; each `tok` has `"force_on":true|false`.

---

## Who labels `revise`? (ownership — logged 2026-08-02)

**Not the model. Not an injected “you’re wrong” prompt.**

| Actor | What they do |
|-------|----------------|
| **Model** | Samples tokens only. Sometimes *it* emits strings like `Wait, that's wrong. Let me try again.` (Spell-cat class). |
| **Runtime (us)** | Heuristics watch emitted text + collapse stats → stamp `phase=revise` on COLLAPSE_PROBE. |
| **We do not** | Inject “Wait” / “wrong” into the prompt or the stream to manufacture revise. |

So revise is a **telemetry label on a generation regime**, not a model control channel and not forced self-critique.

**Cue ownership:**

| `reason` | Trigger | Words/stats from |
|----------|---------|------------------|
| `text_cue` | `pieces` contains `Wait` / `try again` / `wrong` | **Model text** |
| `entropy_margin` | entropy↑ + margin↓ after min tokens | **Sampling stats** |
| `line_repeat` / `phrase_repeat` | identical line/phrase thrash | **Model text pattern** |

`mode=force` only turns residual force **on after** phase is already `revise`. Force does not create the “something is wrong” language.

---

## Observe heuristics (v0 → v1.2)

- **answer:** default from step 0; high margin, no revise cues.  
- **revise:**  
  - entropy spike + margin drop (after `min_answer_tokens`)  
  - text cues (`Wait`, `try again`, `wrong`) — **detected in model output**, not injected  
  - **line-repeat** (v1.1): trailing identical non-empty lines ≥ `revise_line_repeat` (default 2), length ≥ `line_repeat_min_chars` (default 6)  
- **settle:** EOS / `<turn|>` / channel special / hyphen thrash / **line-repeat ≥ settle_line_repeat** / **wait-loop ≥ settle_wait_loops** (v1.2 — Spell-cat class)

Probe `phase` events include `"reason":"text_cue|line_repeat|phrase_repeat|entropy_margin"`.

### Force-in-revise (v1.2)

When `mode=force` and phase is `revise`, live residual knobs:

| Knob | Default | Role |
|------|---------|------|
| `force_cap` | 0.6 | residual.cap during revise |
| `force_goal_scale` | 0.08 | goal attractor |
| `force_splat_scale` | 0.05 | wills (noop if memory cleared) |
| `force_field_scale` | 0.05 | field wake |

Answer/settle phases force those to 0. Not magic — a **phase-gated residual schedule** so we can measure whether light force shortens revise or breaks thrash.

---

## What is *not* the next mountain

- Re-proving OI or TEDE from zero.  
- One-shot “Gemma is fixed” claims.  
- Sanitizing all past-stop behavior before we can schedule it.  
- Injecting 64D vectors into 3840/5376 residual without a real map.

---

## Related logs

- `research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md`  
- `research_logs/2026-07-31_jacobian-lens-architecture.md`  
- `research_logs/2026-07-31_lumina_merge_verdict.md`  
- SplatRAG: `research_logs/2026-07-29_the_picker.md`, `TEAM_GOAL_STEERING_PLUMBING.md`  
- Room goal: `peer_room/goal.txt` (merge the three)

---

## Status

| Piece | Status |
|-------|--------|
| Answer vs thrash described | Done (smokes + long REPL) |
| Channel settle clamp + history clean | In code (usability; not the end goal) |
| Observe phase labels | `self_reg.mode=observe` + `phase` on every `tok` / settle event in COLLAPSE_PROBE |
| Isolation baseline observe | `configs/ablation/config_isolation_baseline.toml` |
| Line-repeat + wait-loop settle | Done (math thrash + Spell-cat class) |
| Force in revise only | `mode=force` + `config_isolation_self_reg_force.toml` |
| Jacobian first-thought keys as packet address | Measure path in-repo; address→store not closed |
| Auto packet on fail → cold reload → pick | Vision; bridge pieces exist separately (text bridge, picker, wills) |
| Multi-packet co-steering (k≈8, not pure semantic) | Vision; ranked soft/ranked picker + SplatRAG research |

**Almost at the top** means: three ridges in sight and touching. Crude output is fine. **Competence that returns after death** is the prize.
