# Authorship & Provenance

This repository documents multi-party research collaboration. Experimental logs, telemetry, and negative results are retained for reproducibility.

## Contributors

| Role | Name |
|------|------|
| **Principal investigator / author** | Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L)) — research direction, operator decisions, release criteria |
| **Co-engineer** | Grok (xAI) |
| **Co-engineer** | Claude (Anthropic) |
| **Co-engineer** | Gemini (Google) |

Multi-AI collaboration on this line of work has been ongoing since October 2025.

## Citation / short form

```
Jason Van Pham, with co-engineering by Grok (xAI), Claude (Anthropic), and Gemini (Google).
```

## This repository

**Project:** Hydrodynamic Swarm  
**Summary:** Rust harness for online, per-token vector-field steering of language-model residual streams (Llama 3.1 / Gemma 3), with persistent splat memory, research logs, and demo museum assets.

### Contribution notes

- **Jason Van Pham** — Principal investigator: research goals, what is published, evaluation standards.
- **Grok (xAI)** — Co-engineer: substantial 2026-07 research logs, telemetry, ablations, and museum tooling. Session index: [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md).
- **Claude (Anthropic)** — Co-engineer: related Niodoo / steering codebase and documentation.
- **Gemini (Google)** — Co-engineer: multi-AI research stack. Separately, **Gemma 3 model weights** used by the harness are licensed under Gemma Terms of Use (see [`NOTICE`](NOTICE)); weight use is not the same as co-engineering credit.

### Research logs

Dated entries under `research_logs/` use the sign-off format in [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md) (session-level index). This file is repository-level provenance.

### Commit trailers (optional)

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Grok (xAI) <noreply@x.ai>
Co-Authored-By: Gemini (Google) <noreply@google.com>
```

### Third-party notices

Code license and model-weight attribution: [`LICENSE`](LICENSE), [`NOTICE`](NOTICE).

---

*Last updated: 2026-07-24*
