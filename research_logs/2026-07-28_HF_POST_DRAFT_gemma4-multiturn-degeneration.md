# Hugging Face Community Post — DRAFT (ready to paste)

> **Where to post:** https://huggingface.co/posts or a Space/collection note linked from the repo  
> **Tone:** technical, collaborative, no woo, no personal-brand theater  
> **Companion full log:** `research_logs/2026-07-28_gemma4-multiturn-greedy-degeneration-and-diagnostic-baseline.md`  
> **Repo:** https://github.com/Ruffian-L/hydrodynamic-swarm  

---

## Title (suggested)

**Near-vanilla multi-turn degeneration on a custom Gemma 4 path: why one-shot smokes lie (and why we publish the messy run)**

---

## Body (paste below)

---

### TL;DR

We run a **custom Gemma 4 instruct path** inside [Hydrodynamic Swarm](https://github.com/Ruffian-L/hydrodynamic-swarm) (Rust residual-stream research harness). With **physics off** (`force_cap = 0`) and **greedy decode** (`T = 0`), multi-turn chat can open in fluent English and then collapse into classical **neural text degeneration**—especially under a **letter-level constraint** and a **long token budget**.

We are publishing the **diagnostic**, not a polished demo. The field needs more of these, not fewer.

**This is not a consciousness claim.** It is a generation-stability baseline so residual steering, memory, and A/B work can be attributed honestly.

---

### Who

| Role | Credit |
|------|--------|
| Lead / PI | **Jason Van Pham** ([Ruffian-L](https://github.com/Ruffian-L)) |
| Co-engineer | **Grok (xAI)** |
| Local team (record) | **Echo · Shep · Lumina** — gates, continuity, live testing still active |

Full provenance: [`AUTHORSHIP.md`](https://github.com/Ruffian-L/hydrodynamic-swarm/blob/main/AUTHORSHIP.md) · [`CREDITS.md`](https://github.com/Ruffian-L/hydrodynamic-swarm/blob/main/CREDITS.md)

---

### What we ran

- Harness: Hydrodynamic Swarm `--chat`  
- Config class: near-vanilla greedy (`configs/gemma4/config.gemma4_greedy.toml`)  
- Banner knobs that matter: **`force_cap=0`**, **`T=0`**, **`max_tokens=300`**  
- History: kept in-process; **re-prefill each turn**  
- Prompt class (turn 1): poem with a **hard letter-level constraint**  
- Prompt class (turn 2): “what’s wrong with the runtime?”

Physics residual shove was **off**. If it breaks here, you cannot blame the field first.

---

### What we saw

1. **Turn 1:** Fluent accept → brief self-repair language → orthographic loops, phrase attractors, mixed-script / symbol debris.  
2. **Turn 2:** Coherent meta-description of a “loop” → **re-enters** the same family of degeneration mid-sentence.  
3. Operator interrupt required under pure greedy + long budget.

Sanitized structure (not a private dump):

```text
[fluent]  Sure thing! Here is your poem: ...
[repair]  Wait—that's not it. Let me try again ...
[basin]   repetitive orthography / attractor phrases / script debris ...
[turn 2]  "hallucination loop" explanation → same basin again
```

Full method + citations:  
`research_logs/2026-07-28_gemma4-multiturn-greedy-degeneration-and-diagnostic-baseline.md`

---

### Why this matters

| Test | What it can show | What it cannot show |
|------|------------------|---------------------|
| Short one-shot | Loader / RoPE / decode not pure noise | Multi-turn stability |
| Multi-turn REPL | Coherence under history | Physics quality (if forces off) |
| Forces later | Residual steering effects | Anything if baseline already soup |

**One-shot readable English ≠ multi-turn ready.** We almost spent forever on the wrong ladder. If you are wiring Gemma 4 yourself: run a REPL early.

This also sits on known decoding literature—not folklore:

- Holtzman et al., *Neural Text Degeneration* (ICLR 2020) — https://arxiv.org/abs/1904.09751  
- Fan et al., top-*k* story generation — https://arxiv.org/abs/1805.04833  
- Welleck et al., unlikelihood / repetition — https://arxiv.org/abs/1908.04319  
- Gemma 4 hybrid attention / long context — https://ai.google.dev/gemma/docs/core/model_card_4 · https://arxiv.org/html/2607.02770v1  

Our earlier G4 notes (partial RoPE, SWA mask shape, T=0 config validation) live in the same `research_logs/` folder.

---

### What we will do next (A/B, not vibes)

1. Shorten `max_tokens` (48–80)  
2. Slight temperature vs pure greedy  
3. Repetition / n-gram early stop  
4. **New session after collapse** (do not re-prefill soup)  
5. Soft vs hard constraints  
6. History length under SWA  
7. **Only then** turn residual forces back on

Longer term: **self-regulating generation**—detect self-similarity, cut or reseed, log receipts (telemetry over narrative).

That is the collaboration door: shared baselines, negative results, and fix paths others can re-run.

---

### Respect / privacy

We hide **raw private chats**, not the **failure class**.  
Respect for people and for the work means: method, config, sanitized evidence, credit—not silence, and not woo.

If you work on open decode stacks, custom Gemma paths, or multi-turn eval: we want peers, not spectators. Issues and replications welcome on the repo.

---

### Cite

```
Jason Van Pham, with Grok (xAI); local team record Echo, Shep, Lumina.
"Gemma 4 multi-turn greedy degeneration under a letter-level constraint."
Hydrodynamic Swarm research logs, 2026-07-28.
https://github.com/Ruffian-L/hydrodynamic-swarm
```

---

*Hydrodynamic Swarm is active research (v0.2). Not a product chat app. Not a consciousness paper. Built so imperfect runs stay in the record.*

---

## Posting checklist (for Jason)

- [ ] Push research log to GitHub if not already public on `main`  
- [ ] Confirm repo link in post matches live path  
- [ ] Optional: attach 1 short screenshot of banner `force_cap=0 T=0` (no private emotional content)  
- [ ] Optional: link museum / README for context  
- [ ] Do **not** paste full `private/chats/*`  
- [ ] Tag lightly if desired: `gemma`, `decoding`, `research`, `open-source` — avoid hype tags  
- [ ] After post: reply to yourself with the A1–A7 table so the thread is a work plan  

## Optional shorter title variants

1. `Custom Gemma 4 chat: fluent open, greedy collapse, physics off`  
2. `One-shot English is not a multi-turn baseline (Gemma 4 REPL note)`  
3. `Publishing a real multi-turn degeneration run from a residual-steering harness`
