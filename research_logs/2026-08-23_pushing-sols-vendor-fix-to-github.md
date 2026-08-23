# Getting Sol's vendor fix onto the GitHub remote

> Date: 2026-08-23
> Agent: Claude (Opus 5)
> Repo: Ruffian-L/hydrodynamic-swarm

## Context

Ask was "update hydro on the GitHub remote", with an open question about
whether a pull was needed first. Surveying the tree first turned out to
matter more than the push did.

## What the survey found

`Hub/Projects/hydro/` holds six hydro trees. Only three point at GitHub
(`Ruffian-L/hydrodynamic-swarm`): `sol-issues` is the real clone, and
`sol-cuda` / `sol-sign` are worktrees hanging off its `.git`. A fourth
(`gitlab-hydro`) points at GitLab and was left alone. Two more are broken
worktree pointers into `/media/ruffianl/ghost_team/...` — their `.git`
files reference paths that no longer resolve, so every git command in them
fails. Not touched; flagged below.

The clone had **never fetched** since it was made — no `FETCH_HEAD`, and the
only remote-tracking refs were `origin/master` and `origin/HEAD`. That is
why the branches looked untracked and the state looked ambiguous.

Checking the real remote instead of the stale local refs:

| branch | local | remote | verdict |
|---|---|---|---|
| `master` | 4d4a54fc | 4d4a54fc | identical |
| `fix/concourse-lock-recovery` | 0561cd2f | 0561cd2f | already pushed |
| `fix/cuda-feature-gating` | e387d30b | e387d30b | already pushed |
| `fix/vendor-checksum-ignore-collisions` | 15 uncommitted files | absent | the actual gap |

So: no pull was needed — nothing upstream to pull, `master` was already
level. Two of three branches were already on GitHub. The only real gap was
Sol's uncommitted vendor work from 2026-08-22.

## What was committed

Sol's 15 dirty paths were not stray junk, which is what they look like at a
glance (`vendor/**/target/`, `vendor/**/.cargo/`, `.vscode/`, `*.bin`).
They are vendored files that the crates' own `.cargo-checksum.json`
manifests require, which broad workspace ignore rules had been stripping —
the exact bug the branch name describes. Committing them was correct;
deleting them as noise would have re-broken fresh clones.

Sol's research log closed with "verification receipts will be added after
the new integrity checker ... run from the tracked file set" — i.e. the
checker was written but never run. Ran it before committing:

    $ python3 scripts/check_vendor_integrity.py
    Vendor integrity check passed: 19311 files verified

That is the receipt Sol left open. Commit `07b97b1e`, 21 files, +4459.

## The pre-commit gate

The machine-wide `cargo-bless` hook (`~/.config/git/hooks/pre-commit`)
blocked the first commit attempt. It fires on staged `*.rs`, which the
restored vendored sources are, then runs `cargo check --locked`, which
panics in `vendor/cudarc/build.rs:138`:

    Unsupported cuda toolkit version: `13.3`. Please raise a github issue.

This box has CUDA 13.3; vendored `cudarc` 0.19.3 does not know that version.
No staged file touches cudarc, so the gate failure is environmental and not
attributable to this change. Used the hook's own documented escape hatch,
`CARGO_BLESS_SKIP=1`, for that single commit, and said so in the commit
message. Nothing was disabled permanently.

Worth noting the direction: before Sol's fix the build died *earlier*, on
the missing `vendor/cc/src/target/generated.rs`. It now gets far enough in
to reach cudarc. The failure moved later, which is progress, not regression.

## Hypothesis

The `cargo-bless` gate will keep failing on this machine for any commit that
stages a `.rs` file, until either `cudarc` is bumped past 13.3 support or the
CUDA feature path is isolated. `fix/cuda-feature-gating` (already on the
remote) is aimed at exactly that, so it is the branch to test this against —
not a new one.

## Next

- Merge order to decide: all three `fix/*` branches sit on `master` with no
  PRs opened. None were opened here; that is Jason's call.
- Repair or retire the two dead ghost_team worktree pointers.
- Confirm whether `fix/cuda-feature-gating` actually clears the cargo-bless
  gate on CUDA 13.3. If it does, the bypass stops being needed.

## Not done

Did not open PRs. Did not merge anything. Did not touch `gitlab-hydro` or
push anything to GitLab. Did not fetch the remote's several hundred `bolt-*`
branches into local refs. Did not modify or disable the pre-commit hook.
