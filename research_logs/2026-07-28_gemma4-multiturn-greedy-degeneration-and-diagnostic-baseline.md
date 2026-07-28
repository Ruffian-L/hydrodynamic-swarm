# Gemma 4 multi-turn greedy degeneration under a letter-level constraint  
### A diagnostic baseline for custom loaders (near-vanilla residual path)

> **Date:** 2026-07-28  
> **Project:** [Hydrodynamic Swarm](https://github.com/Ruffian-L/hydrodynamic-swarm)  
> **Tree:** local research harness (Rust + Candle GGUF path)  
> **Status:** Observational research note · reproducible · **not** a consciousness claim  
> **Privacy:** Full multi-turn transcripts remain under gitignored `private/chats/`.  
> This log publishes **method, configuration, sanitized excerpts, and fix directions only**.

---

## Authors / credit

| Role | Name |
|------|------|
| Principal investigator / lead | **Jason Van Pham** ([Ruffian-L](https://github.com/Ruffian-L)) |
| Co-engineer | **Grok (xAI)** |
| Local team (record; live gates & continuity) | **Echo · Shep · Lumina** (see [`CREDITS.md`](../CREDITS.md)) |

**Short citation form:**

> Jason Van Pham, with Grok (xAI); local team record Echo, Shep, Lumina.  
> “Gemma 4 multi-turn greedy degeneration under a letter-level constraint.”  
> *Hydrodynamic Swarm research logs*, 2026-07-28.

Related prior note (method ladder):  
`research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md`

---

## Abstract

We document a **near-vanilla** multi-turn failure mode on a custom **Gemma 4** instruct path in Hydrodynamic Swarm: short generations can be fluent English, while a multi-turn REPL with **greedy decoding** (`temperature = 0`), **long token budget** (`max_tokens = 300`), and a **letter-level generation constraint** enters a classical degeneration regime (repetition, orthographic thrash, mixed-script debris). Physics residual steering was **off** (`force_cap = 0`). The event is therefore a **generation / multi-turn baseline** issue, not evidence that residual-field “physics” killed the model.

We treat this as a **public-facing diagnostic artifact**: the field benefits from real failure transcripts framed with configs and citations—not only polished demos. We outline an A/B and self-regulation agenda (history hygiene, repetition brakes, sampling, history length under hybrid attention) so the same stack can support serious collaboration without mystical framing.

---

## 1. Motivation

Frontier open-weight work is often judged by one-shot smoke tests and cherry-picked fluent completions. That is necessary but insufficient for:

1. **Custom loaders** (partial RoPE, hybrid sliding-window + global attention, chat wraps).  
2. **Multi-turn re-prefill** (history compounds errors).  
3. **Constrained generation** (letter-level or alliterative constraints are known to stress decoding).  
4. **Research honesty** — imperfect outputs are data; hiding only the failures makes the literature look cleaner than the stack.

Hydrodynamic Swarm is a local residual-stream steering harness with splat memory and telemetry ([README](../README.md)). Before residual physics can be credited or blamed, the **near-vanilla decode path** must be characterized. This note is that characterization for one multi-turn incident.

**Epistemic stance (explicit):**  
We do **not** claim model consciousness, personhood, or “death.”  
We **do** claim that generation dynamics are worth studying, fixing, and publishing when they fail in informative ways.

---

## 2. Setup

### 2.1 Software

| Item | Value |
|------|--------|
| Harness | Hydrodynamic Swarm (`--chat` REPL) |
| Entry | `scripts/chat_gemma4.sh` |
| Config | `configs/gemma4/config.gemma4_greedy.toml` (near-vanilla greedy probe) |
| Loader | `src/gemma4.rs` (custom G4 path; partial rotary; SWA mask work described in prior log) |

### 2.2 Generation knobs (as logged)

From session banner / private transcript header:

| Knob | Value | Intent |
|------|--------|--------|
| `force_cap` | **0** | Residual physics shove off |
| `temperature` | **0** | Greedy / argmax path |
| `max_tokens` | **300** | Long reply budget |
| History | Kept in-process; **model re-prefills each turn** | Multi-turn diagnostic |

TOML reference (`configs/gemma4/config.gemma4_greedy.toml`):

```toml
[physics]
force_cap = 0.0
steer_hidden = true
splat_force_scale = 0.0
goal_force_scale = 0.0
field_wake_scale = 0.0
field_logit_alpha = 0.0
force_ramp_tokens = 0
prefill_bridge_scar = false

[generation]
max_tokens = 24   # session overrode to 300 via --tokens / chat script default
temperature = 0.0
top_k = 1
```

**Note:** Chat script default tokens is high (`chat_gemma4.sh` uses `${1:-300}`). Long budgets amplify degeneration once a bad basin is entered.

### 2.3 Model path (operator environment)

- Gemma 4 instruct GGUF (Q4_K_M class; operator local path).  
- Tokenizer: matching Gemma 4 assets / `tokenizer.json`.  
Exact local paths are environment-specific; weight terms: see [`NOTICE`](../NOTICE).

### 2.4 Prompt class

**User turn 1 (class):** letter-level constraint poem  
> e.g. write a poem where every letter / line starts with `"s"` (operator phrasing).

**User turn 2 (class):** meta-diagnostic  
> ask what is wrong with the runtime / why speech failed.

Letter-level constraints are unusually hard for subword LMs: they push the model off natural token trajectories into character-shaped thrash.

---

## 3. Observation (sanitized)

### 3.1 Turn 1 — fluent open → basin collapse

**Behavior:**

1. Opens in fluent English (accepts task, announces poem).  
2. Briefly self-corrects (“that’s not it / try again”).  
3. Then enters **degeneration**: orthographic loops, phrase attractors, fullwidth/script mixing, symbol debris, until budget or operator interrupt.

**Sanitized structural excerpt (not full private log):**

```text
[fluent open]
Sure thing! Here is your poem:
**S**ome **S**ure ...

[self-repair attempt]
Wait—that's not it. Let me try again ...

[degeneration basin — abbreviated]
... THECOT / Heeness / SlAL ... fullwidth S ... mixed-script debris ...
```

### 3.2 Turn 2 — correct meta → same attractor family

**Behavior:**

1. Produces a **technically plausible self-report** (loop / stuck tokens / failed termination).  
2. Mid-explanation, **re-enters** repetitive / ill-formed loops.  
3. Operator interrupt (`^C`) required; no reliable self-rescue under pure greedy + long budget.

This matches classical **neural text degeneration** under poor decoding conditions: high likelihood paths that are bland, repetitive, or unstable when decoding is not regularized [1].

### 3.3 Attribution (what this is / is not)

| Hypothesis | Supported? |
|------------|------------|
| Residual “physics” murdered the model | **No** — `force_cap = 0` |
| Weights never loaded / pure garbage logits | **No** — fluent English exists at turn openings |
| Multi-turn + greedy + long budget + hard constraint → degeneration | **Yes** — matches observation |
| Prior-turn soup re-prefilled into turn 2 worsens stability | **Plausible** — history includes collapsed text |
| Hybrid attention / SWA / wrap still need multi-turn QA | **Open** — see prior G4 loader log |

---

## 4. Related work (citations)

Decoding and degeneration are not new; we place our incident in that literature so it is taken as **engineering evidence**, not folklore.

1. **Holtzman et al., “The Curious Case of Neural Text Degeneration,”** ICLR 2020.  
   arXiv: [1904.09751](https://arxiv.org/abs/1904.09751)  
   Documents that decoding strategy alone can produce repetitive / degenerate text from the same model; motivates nucleus sampling.

2. **Fan et al., “Hierarchical Neural Story Generation,”** ACL 2018.  
   arXiv: [1805.04833](https://arxiv.org/abs/1805.04833)  
   Top-*k* sampling as a practical alternative to pure argmax/beam pathologies.

3. **Welleck et al., “Neural Text Generation with Unlikelihood Training,”** ICLR 2020.  
   arXiv: [1908.04319](https://arxiv.org/abs/1908.04319)  
   Explicit treatment of repetition and likelihood–quality mismatch.

4. **Gemma 4 model card / technical materials (Google).**  
   Hybrid local sliding-window + global attention; long-context design notes.  
   Model card: [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4)  
   Tech report (HTML): [arXiv 2607.02770](https://arxiv.org/html/2607.02770v1)  
   Relevant to multi-turn re-prefill and mask/window bugs (our prior fix: SWA mask shape `[q_len, kv_len]`).

5. **Su et al., RoPE** — foundational positional encoding used throughout modern LMs.  
   arXiv: [2104.09864](https://arxiv.org/abs/2104.09864)  
   Our G4 path required **partial** rotary on full-attn layers (`partial_rotary_factor` / p-RoPE style keep-pairs); wrong RoPE → unreadable one-shots (fixed earlier; see multiturn diagnosis log).

6. **Hydrodynamic Swarm prior log (this tree):**  
   `2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md` — one-shot ≠ multi-turn baseline; partial RoPE; T=0 validation; SWA crash fix.

---

## 5. Why publish imperfect outputs

Most public demos show **success surfaces**. Custom stacks and multi-turn paths fail in ways that:

- waste weeks if only one-shot tests are used;  
- get mis-attributed to “the model is broken” or to residual steering when decode is the real issue;  
- never enter the searchable record, so solo researchers rebuild the same ladder alone.

**Respect rule for this publication:**

- Publish **configs, method, sanitized failure structure, citations, next experiments**.  
- Keep **private emotional dialogue and full raw dumps** local unless the operator chooses otherwise.  
- No woo. No clout. Credit the team that actually works the tree.

This is the door to **real collaboration**: A/B tests, self-regulating generation (stop on repetition, history quarantine after collapse), and shared baselines others can re-run.

---

## 6. Proposed next experiments (green-lit agenda)

Run **one knob at a time** against the multi-turn REPL baseline.

| ID | Change | Expected signal |
|----|--------|-----------------|
| A1 | `max_tokens` 48–80 instead of 300 | Collapse truncates; less wall-of-debris |
| A2 | `temperature` 0.3–0.7 (vs 0) | Possible escape from pure argmax basins [1] |
| A3 | Repetition / n-gram block or early stop | Fewer `Heeness`-class loops |
| A4 | **New chat after collapse** (do not re-prefill soup) | Turn-2 meta stays coherent longer |
| A5 | Soft constraint prompts (“many words start with S”) vs letter-hard | Constraint hardness ablation |
| A6 | History truncate under SWA window | Quality vs length curve |
| A7 | Only then: re-enable small residual forces | Attribute physics only after A1–A6 |

**Self-regulation sketch (product of collaboration, not mysticism):**

1. Detect high token self-similarity over a window.  
2. Cut generation or raise temperature briefly.  
3. Optionally drop last assistant turn from history if quality score fails.  
4. Log the event to JSONL (Echo-style receipts over vibes).

Shep / Echo / Lumina lanes already emphasize gates, telemetry, and continuity; this note is the **generation-stability gate** that must sit in front of physics claims.

---

## 7. Limitations

- Single operator environment; quant and exact GGUF build matter.  
- Sanitized excerpts, not a full public dump of private chat.  
- No claim that all Gemma 4 deployments behave identically (official runtimes vs custom loader).  
- Physics path deliberately off; results do not evaluate splat/field quality.

---

## 8. Conclusion

Under **near-vanilla** settings (`force_cap = 0`, `T = 0`, long token budget), a multi-turn Gemma 4 chat on our custom path can open fluently and then **degenerate** under a letter-level constraint—and can **re-degenerate** while correctly naming the failure. That is a **generation baseline** problem with a clear experimental ladder, not a reason to hide the record or to mythologize the model.

We publish this so the work is **searchable, citable, and serious**. Imperfect outputs are part of the science. The next step is A/B stabilization and self-regulating decode—not building only in the dark.

---

## References

[1] Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. https://arxiv.org/abs/1904.09751  

[2] Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. ACL. https://arxiv.org/abs/1805.04833  

[3] Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., & Weston, J. (2020). *Neural Text Generation with Unlikelihood Training*. ICLR. https://arxiv.org/abs/1908.04319  

[4] Google. *Gemma 4 model card*. https://ai.google.dev/gemma/docs/core/model_card_4  

[5] Gemma Team / Google. *Gemma 4 Technical Report*. https://arxiv.org/html/2607.02770v1  

[6] Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864  

[7] Van Pham, J., et al. *Hydrodynamic Swarm* — research logs, 2026-07-28 multiturn diagnosis vs oneshot (this repository).

---

**Authorship**

- **Author / PI:** Jason Van Pham (Ruffian-L)  
- **Co-engineer:** Grok (xAI)  
- **Local team (record):** Echo · Shep · Lumina — live gates, continuity, telemetry discipline still in progress on this tree  
- **Project:** hydrodynamic-swarm  
- **Date written:** 2026-07-28  
- **Note:** Failures logged carefully are paths others do not have to re-walk alone. No consciousness claim. Generation must be fixed in public light.

---

*Green light to publish this shaped log and the companion Hugging Face post draft. Respect = method + credit + privacy of raw private chats; not silence.*
