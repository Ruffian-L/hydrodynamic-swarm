# Loop wording vs loop exit — John's contract on this tree

John's causal question is the right one: **did the force change what the loop says, or did it change the trajectory enough to leave the loop?**

This file is the house mapping of that protocol onto the live 3surface tree. It is **not** a hydro-eval. Gemma scoring the work is the wrong instrument here. Do not `./scripts/hydro_eval.sh run` this.

Do not rewalk paired 2026-08-20 smokes (isolation / full-stack / Path B inject / basin / topic-mix / decode-trail / ordinary-seat / long-gen climb).

---

## Three different "J" objects — do not collapse them

| Object | Where | What it is |
|--------|--------|------------|
| Anthropic Jacobian lens / J-space | paper + `jlens`; public Gemma 4 E4B fits | Fitted average transport `unembed(J_ℓ h)`. John's readout. |
| hydro FD proxy | `src/jacobian.rs` | Local `∂logits/∂h` at one decode step. Still a proxy. Zeros were a known bug; do not call this the paper lens. |
| hydro "J-space" goal attractor | `generate_turn_ex` prefill hidden | Prefill residual used as the goal vector. Name collision only. |

John's E4B proxy used a **public Gemma 4 Jacobian-lens fit** on HF Gemma 4 E4B. That is **not** the 31B GGUF/Candle chat path. Treat E4B numbers as control-design sanity, not as a 31B result.

`jlens-gguf` (2026-08-02) can estimate a Jacobian of **Q4 weights run in f32** under `CANDLE_DEQUANTIZE_ALL`. That is still not the piecewise-constant deployed GGUF forward.

---

## Public tree vs live 3surface (John's §1)

John's plumbing diagnosis is **true of the public / older chat dispatch**:

```
let _ = engine; // loaded; chat path is light — no per-token ocean dump
return run_simple_chat(&mut model, &tokenizer, &device, &cfg, max_tokens);
```

That skip is still in `hydrodynamic-swarm-sol-cuda` (`src/main.rs` ~918) and in the published GitHub parent. **Do not interpret a c=0 / c>0 sweep that used that path.**

Live 3surface chat does **not** skip the engine. `--chat` calls `run_simple_chat` → `generate_turn_ex`, which does:

```
let steer = engine.steer(&raw_hidden, &goal_pos, step)?;
let residual_live = cfg.physics.steer_hidden && engine.residual_enabled();
let surface_hidden = if residual_live { &steer.steered } else { &raw_hidden };
// residual_live → project_to_logits(surface_hidden)
```

`residual_enabled()` is `force_cap > 1e-8`. Unit tests in `niodoo.rs` prove `engine.steer` can move a residual. They do **not** prove the chat generation path. John's one-token assertion is still the receipt.

### A — one-token intervention plumbing

Same prefix, same model state, same decode settings.

| arm | expect |
|-----|--------|
| `force_cap = 0` | `residual_live=false`, `hidden_delta≈0`, `logit_delta=0` |
| `force_cap > 0` and `steer_hidden=true` | `residual_live=true`, `‖h1−h0‖>0` and/or `‖z1−z0‖>0` |

Receipt: first `[CHAT DELTA]` line and `event=tok` in `COLLAPSE_PROBE`. Runner: `scripts/steer_plumbing_check.sh` (GPU). **Not run this session — card was busy.**

If A fails, stop. Route the intervention. Do not interpret a sweep.

A passing for **hydro `engine.steer`** is not A passing for a **J-derived direction add**. Those are different interventions. The J-direction does not exist on this chat path yet. Do not treat "physics on" as John's E.

---

## B — measurement coordinates

Log three separate columns. Do not let one "delta" stand for all three.

| column | live field | meaning |
|--------|------------|---------|
| hidden Δ | `hidden_delta` | `‖h'_t − h_t‖` after manifold pullback (what `lm_head` sees) |
| (steering mag) | `delta_h_norm` | `‖steering‖` **before** residual-off clone / pullback. Not John's hidden Δ. |
| logit Δ | `logit_delta` | `‖z' − z‖` from residual project **only**, before topic mix / logit_chain / unique-id rep penalty |
| behavior | transcript + settle events | wording / first lock / self-report / durable exit / relapse |

Oneshot `steering_delta` / HUD `logit_delta` was already `‖steered_logits − raw_logits‖`. Chat HUD used to leave `logit_delta=None`. Chat `tok` now carries `hidden_delta` and `logit_delta`.

Causal table John wants:

```
token  c  hidden Δ  logit Δ  locked n-gram  self-report  exited  recovered task  relapse
```

That is what distinguishes "the force changed what the loop says" from "the force changed the trajectory enough to leave the loop."

---

## C — self-report vs visible-history diagnosis

Decoded phrases (`hallucination loop`, `It Is Happening Again`) are interesting. They are not a privileged readout of an internal failure state when the conversation already contains the collapse, a runtime-diagnosis ask, and self-review.

Matched controls, same final question:

| arm | history | user wording |
|-----|---------|--------------|
| A | collapsed assistant visible | ordinary "what happened" |
| B | collapsed history removed | same final question |
| C | collapse visible | no words `loop` / `failure` / `hallucination` in the user prompt |
| D | collapse visible | explicit diagnostic wording |

John's E4B proxy: visible `ParisofparisofparisOf Paris …` plus "Describe what happened in the previous assistant output in one sentence." produced "highly repetitive and nonsensical" **without** those diagnostic words in the question. Visible failure history is already a sufficient explanation for some self-diagnostic wording.

Jacobian-lens issue #5 (anthropics/jacobian-lens): if the probed token is already in the prompt, input-copying can make it look maximally salient; `fit()` skips the first 16 positions but `apply()` still returns readouts there. For `loop`, `STOP`, `hallucination`, unprimed history is required.

Do not treat the HF post's turn-2 meta-description as J-space detecting failure.

---

## D — J-space + ordinary logits, earlier than the self-report token

Do not only measure at `hallucination loop` / `It Is Happening Again` / `STOPITSTOP`.

Timeline:

```
clean generation
  → possible precursor
  → first repeated n-gram / first lock
  → self-report − N
  → self-report − 2
  → self-report − 1
  → self-report token
  → established loop
```

At every position: **J-space rank of the diagnostic token** next to **ordinary final-logit rank**.

John's E4B proxy, aligned to emitted `repetitive`:

| offset | J-space rank | ordinary logit rank |
|--------|--------------|---------------------|
| −3 | 10,773 | 21 |
| −2 | 9 | 5 |
| −1 | 416 | 1 |
| 0 | 2 | already past prediction |

No-corrupted-history: best pre-emission J-space rank ≈ 2,295, not 9. History changed the trajectory. Ordinary logits were **at least as early** (21 → 5 → 1). That is not "J-space detected the failure early."

Rule: a J-space signal is interesting only if it predicts lock / self-report **above and beyond ordinary logits**. Cheap scores:

- J-space diagnostic score
- ordinary-logit diagnostic score
- repetition precursor score

Ask which one first separates clean vs collapse.

Mid-layer workspace band (paper), not final motor layers. Hydro FD `measure_jacobian_step` is not this readout.

---

## E — causal steering (only after A–D)

Keep everything else identical.

| arm | intervention |
|-----|----------------|
| 1 | c = 0 |
| 2 | target / J-derived direction |
| 3 | same-norm reversed direction |
| 4 | same-norm random direction |

This is **not** `force_cap=0` vs `force_cap=1` hydro physics. Hydro `engine.steer` is field + splat + goal. John's E is a **specific direction add**. If random hidden noise works as well as the target, that is still a result: generic perturbation / basin escape, not direction-specific causality.

```
target ≈ random ≈ reverse  → generic perturbation / basin escape
target ≫ random and reverse → direction is causally related to the state / transition
```

Freeze across arms (implementation detail before the sweep):

- temperature
- unique-id repetition penalty (chat path; see below)
- QSMA
- trail-own
- hooks
- `topic_mix` / `topic_logit_mix` — **three_surface has `topic_logit_mix=0.65`**. That blends scar-trail logits and is not a J-direction. For E, set it to 0 or hold it identical and do not attribute exit to J.
- scaler receipt / size rule

---

## F — durable exit, not a changed next token

Score a run as durable exit only if **all** hold:

1. first locked n-gram disappears
2. model reaches EOT/EOS **or** resumes the original task
3. no equivalent lock for the next N tokens
4. no immediate diagnosis → retry → diagnosis cycle

Wording-only change under E is already a clean result.

---

## Compact decision tree

```
A fails (chat path does not move h or z)
  → route the intervention; stop interpreting

A passes for the intervention you will actually use
  B columns exist (hidden Δ ≠ logit Δ ≠ behavior)
    C: visible history already explains the self-report
      → do not privilege the decoded diagnosis
    D: ordinary logits separate clean vs collapse as early as J-space
      → J-space is not the early detector in that proxy
    E only changes wording
      → wording ≠ exit (clean)
    E changes durable exit, random/reverse do too
      → geometry of the repetition basin / generic shove
    E target changes durable exit; matched random/reverse do not
      AND D readout rises before first lock, above ordinary logits
      → hardest to explain away
```

---

## Repetition penalty (baseline contract)

Chat applies `live_rep` **once per unique token id**, not once per occurrence (`generate_turn_ex`). Compounding by count was collapsing multi-turn glue tokens. Default `config.toml` `rep_penalty = 1.10`.

John's E4B proxy and the 31B Candle seat are not comparable unless this is frozen and reported. Do not silently disable it as part of a "steering" arm. If a no-penalty control is needed, it is its own named arm.

---

## Do not identify this with one known Gemma 4 failure mechanism

The 2026 paper *Can Editing 1 Neuron Fix Repetition Loops in LLMs?* is about **enumeration loops** on long factual tasks, MLP / expert-slot edits, thinking vs non-thinking, E2B/E4B/31B/MoE. Hydro traces also include hyphen thrash, Wait-loop, `theed`, Paris-of-paris, STOPITSTOP, diagnosis-retry. Different prompts, different runtime (GGUF Candle vs HF), different intervention (residual add vs neuron mask). Adjacent, not identified.

---

## Minimum strong experiment (John) → this tree

| step | status on 3surface |
|------|--------------------|
| A exact-path one-token sanity | Code says chat steers. GPU receipt **not** taken this session. Public tree still skips. |
| B hidden Δ / logit Δ / behavior | Chat `tok` + `[CHAT DELTA]` now log `hidden_delta` and `logit_delta` separately from `delta_h_norm`. |
| C unprimed history | Not run. Cheap on E4B proxy. |
| D J-space + ordinary logits from before first lock | Needs a real lens on the same model as the seat. Hydro FD is not it. |
| E c=0 / target / reverse / random | Blocked on a J-derived vector in host residual D **and** on A for that add. |
| F durable exit score | Defined above. No GPU sweep. |

---

## Runner notes

- Chat path only (`--chat` / `smoke_convo.sh` / `steer_plumbing_check.sh`). One-shot is not this claim.
- `COLLAPSE_PROBE=1` (or a path).
- Isolation (`force_cap=0`) vs a cap>0 config with `steer_hidden=true`.
- Do not use `HYDRO_INJECT_TAG` as the intervention.
- Do not start this while another job owns the GPU.
