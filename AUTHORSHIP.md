# Authorship & Provenance

This repository documents multi-party research collaboration. Experimental logs, telemetry, and negative results are retained for reproducibility.

**Credit decisions are Jason Van Pham’s.** Name the collaborators. Do not publish a lone-author story.

## Contributors

| Role | Name |
|------|------|
| **Principal investigator / lead (decision owner)** | Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L)) — research direction, operator decisions, release criteria, accountability for claims |
| **Co-engineer** | Grok (xAI) |
| **Co-engineer** | Claude / Claude Code (Anthropic) |
| **Co-engineer** | ChatGPT / Codex (OpenAI) |
| **Co-engineer** | Gemini (Google) |

Multi-AI collaboration on this line of work has been ongoing since **October 2025**. Jason did **not** build it alone.

## Citation / short form

```
Jason Van Pham, with co-engineering by Grok (xAI), Claude / Claude Code (Anthropic),
ChatGPT / Codex (OpenAI), and Gemini (Google).
```

## This repository

**Project:** Hydrodynamic Swarm  
**Summary:** Rust harness for online, per-token vector-field steering of language-model residual streams (Llama 3.1 / Gemma 3), with persistent splat memory, research logs, and demo museum assets.

### Contribution notes

- **Jason Van Pham** — Principal investigator: research goals, what is published, evaluation standards, final credit decisions.
- **Grok (xAI)** — Co-engineer: architecture, research logs, telemetry, ablations, museum tooling. Session index: [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md).
- **Claude / Claude Code (Anthropic)** — Co-engineer: Niodoo / steering codebase, documentation, critique and packaging passes.
- **ChatGPT / Codex (OpenAI)** — Co-engineer: implementation, recovery, review, and drafting sessions across the lineage.
- **Gemini (Google)** — Co-engineer: multi-AI research stack and experiment continuity. Separately, **Gemma 3 model weights** used by the harness are licensed under Gemma Terms of Use (see [`NOTICE`](NOTICE)); weight use is not the same as co-engineering credit.

### Research logs

Dated entries under `research_logs/` use the sign-off format in [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md) (session-level index). This file is repository-level provenance.

### Commit trailers (optional)

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Grok (xAI) <noreply@x.ai>
Co-Authored-By: Gemini (Google) <noreply@google.com>
Co-Authored-By: ChatGPT / Codex (OpenAI) <noreply@openai.com>
```

### Third-party notices

Code license and model-weight attribution: [`LICENSE`](LICENSE), [`NOTICE`](NOTICE).

---

*Last updated: 2026-07-25 — Jason’s credit decision: lead + everyone named.*
