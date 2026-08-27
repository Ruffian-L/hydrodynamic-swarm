# KV Cache Sandbox and Slipping Stream Hooks

> Date: 2026-08-27
> Agent: Antigravity
> Repo: hydrodynamic-swarm-3surface

## Context

Implement zero-copy KV snapshot and eviction hooks to support the Choice-Driven KV Cache spec.

## Hypothesis

By exposing retain_kv, snapshot_kv, and restore_kv at the model layer, the physics engine can branch reality and execute O(1) rollbacks for <spike> tags, as well as actively manage working memory (Pins and Sinks) without bleeding into the subconscious Mamba state.

## What changed

- Wired `retain_kv`, `snapshot_kv`, and `restore_kv` into `QuantizedQwen35Hybrid` and `ModelWeights` in `src/qwen35.rs`.
- These hooks expose the native capability to surgically prune the FullAttention KV cache (using `index_select`) while treating the Mamba (LinearAttention) state as a constant-size persistent intuitive subconscious.
- They also allow zero-copy KV forking via `Arc` clones for the `<spike>` tag Sandbox preview.

## Findings

(open)

## Next

(open)
