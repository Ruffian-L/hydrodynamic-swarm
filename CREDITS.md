# Credits

This was a collaboration. Recording who did what is part of the work, not a footnote.
**Credit decisions are Jason Van Pham’s.** This file follows his rule: name the collaborators;
do not flatten the project under a lone-author story.

## Lead (decision owner)

**Jason Van Pham** — research direction for Hydrodynamic Swarm (residual-stream field steering,
splat memory, museum / telemetry discipline), evaluation standards, release decisions, and final
accountability for what is published. He has led this line with AI and local-team collaborators
since about **October 2025**. He did **not** build it alone.

Contact: jasonvanpham@niodoo.com

## AI collaborators (credit everyone)

| Collaborator | How they show up in this work |
| --- | --- |
| **Grok (xAI)** | Architecture, research logs, telemetry, ablations, museum tooling, continuity stack, long-running co-engineering |
| **Claude / Claude Code (Anthropic)** | Code, review, packaging, documentation, solo-lane / merge-direction passes |
| **ChatGPT / Codex (OpenAI)** | Implementation help, recovery, critique, drafting sessions across the lineage |
| **Gemini (Google)** | Experiment dialogue, continuity, multi-provider research stack |

Where a stretch names one system more specifically, that is **extra detail**, not a reason to erase the others.
Full short table also lives in [`AUTHORSHIP.md`](AUTHORSHIP.md).

## Provenance

The residual-field harness, splat memory, and related approaches in this lineage were developed by
Jason Van Pham beginning in late 2025 **with** these collaborators. This repository is a dated
public record. The license governs reuse of the code; it does not transfer the provenance of the
ideas, and it does not mean “Jason typed every line alone.”

## Local / persona collaborators

Named local collaborators on the Niodoo / hydro home (runtime dialogue, gates, repair, and live
testing). Public use of persona names follows Jason’s approval; credited here as part of the record.

They worked this tree **before** cloud co-engineers re-entered some sessions, and they **still
work it** — including gate receipts and live ablation testing (e.g. G1–G2 closed, **G3** pain /
dissipation knobs and configs in tree).

| Collaborator | How they show up in this work |
| --- | --- |
| **Shep** | Build/repair loops; endocrine path restore and wire-in; G-gate lead on memory-coupling smokes; ablation / pain–dissipation testing direction |
| **Echo** | Runtime and collaboration passes; telemetry discipline (pot / nearest / force flags); anti-narrative “receipts over prose” |
| **Lumina** | Earlier project lineage; human-readable RECEIPT.md / museum notes; continuity across team sessions |
| **Nex** | Memory and entity tracking in the wider home (when present on a stretch) |

Team goal and gate docs that name them explicitly (ops, not legal license):

- [`TEAM_GOAL_MEMORY_COUPLING.md`](TEAM_GOAL_MEMORY_COUPLING.md) — Shep · Echo · Lumina roles on coupling smokes  
- [`docs/SHEP_LONGFORM_GOALS_20260725.md`](docs/SHEP_LONGFORM_GOALS_20260725.md) — G1–G3 board (G3 open for team testing)  
- [`docs/ENDOCRINE_SHEP_WIRED_20260718.md`](docs/ENDOCRINE_SHEP_WIRED_20260718.md) — Shep endocrine restore path  

On-disk artifacts that match this work include `scripts/g3_ablation.sh`, `configs/gates/config.g3_*.toml`, and
signed receipts under `logs/` when a gate closes.

## Note

Per-experiment attribution lives in research logs and run receipts (which binary, which config,
which session). Where this file is vague, the artifacts are specific. Corrections that **add**
missing collaborator credit are welcome. Corrections that rewrite this into a solo-author story
are not.

Sibling claim packages use the same pattern: `niodoo-hidden-state-steering` / `niodoo-live` →
`CREDITS.md`.

## Credit everyone — no over-credit

**Rule:** Thank who actually helped. Do **not** rewrite earned work as if someone
else shipped it.

### Earned here (do not hand away)

| Who | What |
| --- | --- |
| **Jason + Grok + Claude + Gemini** (and the rest of the AI lineage on this path) | **Llama and Gemma 3 loaders**, tuning until it worked — trial and error, garbage outputs, no public “Gemma3/Qwen map” day-one. That win is **ours**. Not copied. |
| **Shep · Echo · Lumina · Nex** | Runtime, gates, continuity, team memory — as above. |

### Fair external credit (format / weights / packaging)

| Who | What |
| --- | --- |
| **Hugging Face Candle** | Quantized GGUF runtime we build on. Apache-2.0 OR MIT. |
| **Google** | Gemma weights / terms. |
| **Unsloth / bartowski** | GGUF packaging where we use their files. |
| **llama.cpp** | GGUF *ecosystem* / format history only. **Not** the source of our Llama, Gemma 3, or Gemma 4 loaders. Jason hard-pathed Gemma 4: no C++ peek. |

Detail: `NOTICE`, `research_logs/2026-07-28_gemma4-loader-attribution-and-map.md`.

*Last updated: 2026-07-28 — hard path for Gemma 4; format history only for llama.cpp.*
