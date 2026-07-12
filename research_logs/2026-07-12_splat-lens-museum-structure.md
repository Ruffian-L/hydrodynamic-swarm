# SplatLens museum structure — keep this for every mini-milestone

**Date:** 2026-07-12  
**Authors:** Jason (vision / co-engineer) · Grok (xAI) (slice demo + museum door)

---

## Rule

When a **mini-milestone** lands (stable short-paragraph, force fix, size port, etc.), **retain the same museum path** so forks can watch without CUDA:

```text
./splat-lens                 # interactive door (view-only by default)
./splat-lens museum          # open museum immediately
./splat-lens generate        # optional: record new run (needs GPU + GGUF)
./demo_slice.sh              # generate B4d-q slice then open museum
```

### Layout (do not rename casually)

| Path | Role |
|------|------|
| `splat-lens` | One friendly CLI door |
| `tools/museum/index.html` | Museum UI |
| `tools/museum/catalog.json` | Milestone list + worked / didn’t / ongoing |
| `tools/museum/demos/*.viz.json` | **Committed** public checkpoints |
| `tools/splatlens_slice.html` | Slim loader (drop/fetch `.viz.json`) |
| `tools/latest_demo.viz.json` | Local only (gitignored) |
| `tools/museum/demos/custom-latest.viz.json` | Local only (gitignored) |
| `demo_slice.sh` | Generate + open museum |

### Adding the next mini-milestone

1. Run with `--viz` (or `./demo_slice.sh` / `./splat-lens generate`).
2. Copy `logs/*.viz.json` → `tools/museum/demos/<id>.viz.json` (keep ~50–100KB if possible).
3. Add an entry to `tools/museum/catalog.json` (`worked` / `didnt_work` / `ongoing` + `research_logs`).
4. Sign a research log with **Authorship** (Grok + Jason).
5. Commit **code + museum demos + research logs** — not `logs/**` runtime noise.

---

## Current public checkpoints

- **b4d-q-friendship-65** — locked short-paragraph 4B slice (recommended)
- **early-v1-friendship-50** — historical SplatLens recording

Research ledger index: `research_logs/AUTHORSHIP.md`

---
**Authorship**
- **Author:** Grok (xAI) — session co-engineer with Jason / Shepard
- **Operator / vision:** Jason (co-engineer)
- **Role:** museum structure lock + git hygiene for mini-milestones
- **Project:** hydrodynamic-swarm
- **Date written:** 2026-07-12
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends. Keep `./splat-lens` as the stable demo door across milestones.
---
