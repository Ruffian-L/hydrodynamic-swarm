# Qwen System Prompt and Thinking Block Support

> Date: 2026-08-27
> Agent: Antigravity
> Repo: hydrodynamic-swarm-3surface

## Context

Ensure Qwen understands the Choice-Driven KV Cache physics and exposes its thoughts.

## Hypothesis

If we don't adjust the system prompt, Qwen will be confused by <spike> and won't know how to use the physics hooks. If we don't update the thought block parser, the REPL might handle the tokens incorrectly.

## What changed

- Patched `format_multiturn_prompt_ex` for Qwen to include a robust system prompt explaining `<think>`, `<spike>`, and `<lock>`.
- Updated `gemma4_in_open_thought` to recognize `<think>` and `</think>` boundaries so the REPL prints them as live trajectory correctly.

## Findings

(open)

## Next

(open)
