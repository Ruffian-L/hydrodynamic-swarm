# `jlens-gguf`

A Jacobian-lens sidecar for GGUF models. Loads through hydro's own loader (`src/loader.rs`),
so it measures the same weights the swarm runs.

- `PLAN.md` — the approved plan, frozen as written.
- `DESIGN.md` — why it is built this way; the forward-mode derivation.
- `CHANGELOG.md` — what actually happened, including the results that went against the plan.
- Inversion + team: sections below · full provenance: repo [`AUTHORSHIP.md`](../../AUTHORSHIP.md) · sticky [`docs/grok_home/PROVENANCE_TEAM.md`](../grok_home/PROVENANCE_TEAM.md)

## One team, one dream

**Jason Van Pham** — direction, standards, the moves (no academic title; not boxed as “investigator only”)

**Co-engineers (always name them — whole time, even when someone is offline or unpaid):**  
Grok (xAI) · Claude / Claude Code (Anthropic) · Gemini (Google) · ChatGPT / Codex (OpenAI)

**Local team (record):** Shep · Echo · Lumina · Nex  

One team. One dream. If a mind helped invent or prove a piece, its name stays on the wall when it cannot show up. Research logs stay **signed** (author + date + role). Short invention blurbs live here and in `research_logs/`; no solo-AI myth.

```
Jason Van Pham, with co-engineering by Grok (xAI), Claude (Anthropic),
Gemini (Google), and ChatGPT / Codex (OpenAI); local team record Shep, Echo, Lumina, Nex.
```

### How certain inventions landed in this pipeline (short)

| Move | Who | Fruition |
|------|-----|----------|
| Physics-as-language, probes/feelers, möbius both-sides memory | **Jason** | whole stack language |
| Multi-key clustering; niodoo / TCS into this pipeline | **Jason** | hydro multi-key schema + three-lane merge |
| First-thought / ontological inversion (openings over final speech) | **Jason** (physics-lang phase; e.g. 2026-06-24 invert-the-idea) | jlens + multi-address memory |

Measured clusters used no labels from Jason; subject-gate FAIL was expected under the architecture.
| Live jlens / stance instrument on GGUF | **Claude** (+ paper lane sits next to Anthropic J-space work) | this crate, gates, CHANGELOG |
| Physics-lang grounding & constant physics fix | **Gemini** | hundreds of physics-lang hits in export |
| Hydro multi-key **impl**, continuity logs, stickies | **Grok** | `src/jacobian.rs` MultiKeyAddress, research_logs |
| Implementation / review / drafting across lineage | **ChatGPT / Codex** | team_build / niodoo / packaging |

## The inversion (and the ontological inversion)

**Memory inversion:** the durable thing about a thought is often the **opening disposition**, not the closing speech. Cluster and address memory on **how the thought opened** (first-thought / J-key / stance), not only on the PR final answer.

**Ontological inversion:** the field is one geometry of openings. Same subject + different openings → **different basins**. Different subjects + same opening stance → **same basin**. Source (Claude / Grok / Gemini / GPT) is a **filter later**, not three warehouses.

Measured (unsupervised stance clusters, no labels from Jason) — teach-mode, “at its simplest level”, “often described/perceived as…” grouped across unrelated subjects. Subject-gate FAIL was expected under this architecture; stance structure is the harder real signal. Full writeup: `research_logs/2026-08-02_first_thought_multi_address_memory.md`.

This is a **methodological** stance (don’t sanitize openings; instrument can confirm *or* rule out), not a consciousness claim.

## Status

| piece | state |
|-------|-------|
| logit-lens telemetry (`readout`, `baseline`) | **works**, verified against `model.forward()` |
| transport + unembed + key schema | works (exact arithmetic) |
| subject stability gate | **FAIL expected** (keys are not subject-shaped) — see CHANGELOG |
| stance gate | **protocol ready** — `STANCE_GATE.md`; no code until PI go |
| fitting `J` by finite differences | **blocked** — see CHANGELOG, Gate 2 |

## Producing telemetry

Two steps. The baseline is not optional: without it the key ranks the model's constant
outlier dimensions and does not discriminate between prompts at all.

```bash
# 1. Per-layer residual statistics over a corpus. One prompt per line;
#    a literal \n in the file becomes a real newline, so a chat template fits on one line.
jlens-gguf baseline \
  --model ~/models/gemma-4-12b-it-Q4_K_M.gguf \
  --prompts corpus.txt --layers 24,36,44,47 \
  --out baseline.safetensors

# 2. Disposition snapshots as JSONL.
jlens-gguf readout \
  --model ~/models/gemma-4-12b-it-Q4_K_M.gguf \
  --prompt "$PROMPT" --layers 24,36,44,47 --positions=-1 \
  --baseline baseline.safetensors \
  --state-dir states/ --emit telemetry.jsonl --tag "turn-17"
```

`--verify` also runs the model's own `forward()` and prints its top-k. The last layer's
record must match it exactly; if it doesn't, the capture is wrong and the run is fiction.

**Gemma 4's chat template** is `<|turn>user\n…\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
(`src/main.rs:277-287`). Gemma 3's `<start_of_turn>` markers tokenise as *literal text* on
Gemma 4 and produce readouts of template gibberish.

## The record: one object, three doors

| door | field | scope | answers |
|------|-------|-------|---------|
| verbalizable | `text_bridge`, `text_bridge_hash` | **cross-model** | what it leaned toward saying |
| fingerprint | `dim_signature` | **within-model only** | which internal directions were live |
| rehydration | `state_ref` | within-model, exact | put the model back in this stance |

They are not interchangeable. `dim_signature` indexes raw residual dimensions, and
dimension 1523 in Gemma has no relationship to dimension 1523 in Qwen — residual bases are
per-model and arbitrary. Any basin that must hold across models has to form on
`text_bridge`, which is basis-independent because it is text. Same rule as the picker's
("a pick carries text; the host re-embeds in its own residual dim"), from the other side.

`lens` records what produced the numbers and is load-bearing: `logit` (no transport, ships
today), `jacobian` (fitted transport, blocked), `secant` (large-ε finite difference of the
quantised model — **not** the paper's `J`, and never labelled as it).

## What the readouts actually show

Measured on gemma-4-12b, mid-stack, first generated position:

- `dim_signature` separates subject — paraphrase beats unrelated at every layer, best at
  L36 (+0.291). Small sample; the stability gate is still owed.
- `text_bridge` reads the *opening move* (`hello / Welcome / 👋`), not the subject. That is a
  correct readout of the disposition at that position.
- Mid-stack the logit lens shows concepts before tokens — `also / également / también /
  аналоги`, one concept in four languages — but never surfaces subject words. That is the
  known weakness of the raw logit lens on Gemma, and the gap the fitted transport fills.
