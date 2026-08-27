# Path B surface protection check done

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Protect post-Day-49 Path B / QSMA / remember-store sources by copying them outside logs/ before any --d-run or endocrine edits. We think a later overwrite of D_gemma.jsonl or a dirty-tree accident must not erase the untracked modules.

## Hypothesis

Copies plus a dated research log are enough to keep Path B readable while the working tree stays uncommitted.

## What changed

**Check: DONE before any endocrine / --d-run code changes.**

Live tree: `/home/ruffianl/Hub/Projects/hydro/hydrodynamic-swarm-3surface`
Branch: `physics/three-surface` HEAD `15fb7338071bd6c41a4207f50c1907f48de600d3` (Day 49).

### Protection method

Copied (not committed; 3surface working tree is 6k+ dirty and vendor-noisy) to a durable sibling **outside `logs/`**:

`/home/ruffianl/Hub/Projects/hydro/path_b_protect_20260820/`

Scratch copy for the measurement seat: `/tmp/grok-goal-9902ae0e9d81/implementer/path_b/` plus `/tmp/grok-goal-9902ae0e9d81/implementer/path_b_protect.txt`.

### Human-checkable files (present, non-empty)

| path | bytes | sha256 | git |
|------|------:|--------|-----|
| `src/qsma.rs` | 4876 | `85f456a8d97c255470657d4bdfe512ce9e794a9d674f0b15d04d743143e0f803` | untracked |
| `src/control_tags.rs` | 13316 | `8034a2c5124c443e62c58551ef4b361f154c4e7f47cee436251f6092645068be` | untracked |
| `src/remember_store.rs` | 6060 | `35e32ca091d8b4987d363856c4b0b69219f222e441d7f34eefc5dbb1098c40d7` | untracked |
| `src/niodoo.rs` (wiring) | 54353 | `7209db798592b2e5f2a7648cca86d1ffaa08c156c5da6039c6e8e7198ff7f186` | modified |
| `src/main.rs` (wiring) | 176939 | `d7036bafcc36fabfae8412ff9aae1ad05ecc7a1a81f01a06a784f42ae96ded7a` | modified |
| `src/lib.rs` | 1259 | `40ada91c7a2f3d3d4a457f4d76dfac5ad58161c10d0771245ba1f366ca3a56a6` | untracked |
| `src/endocrine.rs` | 10345 | `e041a1d980b837827af7cdaa3409146903112070d8b5c024ba249521200593f6` | modified |

Live copies still sit at those `src/` paths (not moved). Durable copies are named without the `src/` prefix in the protect dir.

### Archaeology snapshot (before any new --d-run)

Copied into `/tmp/grok-goal-9902ae0e9d81/implementer/archaeology/` without truncating originals:

- `logs/2026-08-19_18-39-48_gemma4_v3-forcecap0_T0_75_s20_a0_d18.jsonl` 4634796 B (pid 340257)
- `logs/2026-08-20_04-08-12_gemma4_v3-forcecap0_T0_75_s20_a0_d18.jsonl` 116550 B (pid 437249)
- `logs/D_gemma_stdout.log` 348775 B
- `logs/live.txt` 398563 B
- `logs/D_gemma.jsonl` 601 B
- `logs/LOOP_TRAIL.txt` 17491 B

Manifest: `/tmp/grok-goal-9902ae0e9d81/implementer/archaeology/MANIFEST.txt`.

No `--d-run` launched in this step. No 131072-token run started.

## Findings

**Protection check DONE.** All seven files are present, non-empty, readable at live `src/` paths and at the durable sibling. Archaeology jsonl + stdout + LOOP_TRAIL snapshotted before any new `--d-run`. No code diffs in this step.

## Next

Force `--d-run` endocrine off (residual physics on, hooks off). Allow `--tokens` to keep a short diagnostic from becoming 131072. Walk the snapshotted traces into degraded D cards.

---

Signed: Grok (xAI) · operator Jason
