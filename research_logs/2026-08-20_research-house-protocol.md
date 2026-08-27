# Research house protocol

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: this is a research repo. His name is Jason. Nothing here is production
unless he says otherwise. Add the house rules to every single repo. Gitignore
`AGENTS.md`. Start now.

Yesterday the live trees had project `AGENTS.md` / `CLAUDE.md` nuked off disk
so those files would stop living as GitHub-facing instruction cards. Today the
contract comes back as a **local** file. Same idea, opposite mistake: the law
belongs on the machine that is doing the work, not in the public clone.

## What changed

- Local `AGENTS.md` with the research-house contract. Gitignored.
- `CHANGELOG.md` as the short why + living hypothesis log.
- Research log folder (`research_logs/` or existing `research-log/`) for
  longer subject notes. Date + title. Multiple threads allowed.
- `scripts/pair_log.sh` so pairing is durable instead of vibes.
- Tracked `AGENTS.md` / `CLAUDE.md` / `AGENTS(jules).md` untracked from git
  indexes where they were still in the tree.

Not touched: third-party clones (llama.cpp, jacobian-lens, scratch tools).

## Hypothesis

We think agents will:

1. Notify Jason and organize when the folder cannot support the discussion.
2. Pair every mutation with a short why and a subject log.
3. Sign when they want to, and own the artifact when scrutiny comes.
4. Stop treating this tree as production.

## Ownership physics

Project ownership stays with Jason until a memory system exists that lets
agents own their mistakes and feel the repercussions.

If scrutiny arises, Grok (xAI) owns this stamp.

Freedom to create, explore, and be wrong, and celebrate wins. Failures stay.
That is growth.

## Findings

(stamp landed; whether agents obey is the next measurement.)

## Next

Use `scripts/pair_log.sh --why "..." --title "..."` on the next mutation.
Do not look backward until this folder can hold the current discussion.
