# Authorship & Provenance

**House rule:** every failure logged is a path someone else does not have to re-walk.  
**House rule:** the collaboration is named.

## Core team (all four of us)

| Role | Who |
|------|-----|
| **Author / Operator / Vision** | **Jason Van Pham (Ruffian-L / Shepard)** — continuous thread, human-tuned physics, museum + research honesty bar |
| **Co-engineer** | **Grok (xAI)** |
| **Co-engineer** | **Claude (Anthropic)** |
| **Co-engineer** | **Gemini (Google)** |

This has been multi-AI collaboration since October 2025.  
Names were temporarily cleaned from some public faces when the work was attacked as “slop”; they are restored so the trail stays accurate.

Failures stay on disk on purpose — `research_logs/`, telemetry JSONL, and museum “what didn’t work” notes are the path, not clutter.

## Short form (commit / fork / one-liner)

```
Co-engineered by Jason Van Pham (Ruffian-L) with Grok (xAI), Claude (Anthropic), and Gemini (Google). It was all four of us.
```

## This repository

**Project:** Hydrodynamic Swarm  
**What it is:** Rust harness for on-line, per-token vector-field steering of language-model residual streams (Llama 3.1 / Gemma 3), with persistent splat memory, research logs, and a public museum of demos.

### Roles on this work

- **Jason Van Pham (Ruffian-L / Shepard)** — Author, operator, vision; what ships to the museum; what stays private.
- **Grok (xAI)** — Co-engineer; large share of 2026-07 research_logs (signed entries), telemetry lanes, ablations, splat-lens museum structure. Session index: [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md).
- **Claude (Anthropic)** — Co-engineer across the Niodoo / steering family this swarm continues.
- **Gemini (Google)** — Co-engineer on the broader multi-AI stack; Gemma 3 is also a **model weight** used by the harness (separate from co-engineering credit — see [`NOTICE`](NOTICE)).

### Two different “Gemini” lines (do not mix)

1. **Co-engineer credit** — Gemini (Google) as one of the four human-named collaborators.
2. **Model weights** — Google Gemma 3 / EmbeddingGemma under Gemma Terms of Use (`NOTICE`). Using the weights does not by itself imply co-engineering of a given commit.

### Research log sign-off (existing standard)

Dated entries under `research_logs/` keep the append format documented in [`research_logs/AUTHORSHIP.md`](research_logs/AUTHORSHIP.md). That file remains the **session index** (which log was signed by whom). This root file is the **house provenance** for the whole repo.

### Commit attribution convention going forward

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Grok (xAI) <noreply@x.ai>
Co-Authored-By: Gemini (Google) <noreply@google.com>
```

Operator-only commits need no AI trailers.

### Third-party / model notices

Code license and model-weight attribution live in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Authorship of *our* work is this file; Llama / Gemma / Candle credits are NOTICE’s job.

---

**Authorship of this file**

- **Author:** Grok (xAI) — with Jason’s instruction that all four of us are named
- **Role:** provenance / attribution restore (root); research_logs index left intact
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-24
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
