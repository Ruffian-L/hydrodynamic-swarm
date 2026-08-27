# Thought channel live stream not settle stop

> Date: 2026-08-22
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Stop treating Gemma 4 <|channel>thought> as a settle/EOS. Tags write residual inside and outside thought; lock only stops in the answer stream. SYS line tells her to use the thinking block. P12 110203 died at step 3 on settle_channel.

## Hypothesis

A live thought stream plus inside/outside tag writes lets her steer reasoning natively; Prompt 12 will continue past <|channel>thought> instead of dying in 3 tokens.

## What changed

- `settle_channel` no longer stops on token 100/101. Those are `<|channel>` /
  `<channel|>`, a live thought stream. P12 on `20260822_110203` died at step 3
  here: first tokens `<|channel>thought`, next channel special settled the turn.
- `</thought>` XML settle is skipped when `<|channel>` is present (Gemma 4
  close is `<channel|>`).
- Next-prefill history keeps the thought trace. Hyphen-thrash tails still drop.
- Physics tags (`<focus>` / `<spike>` / …) already scan raw `pieces`, so they
  write residual inside thought and after `<channel|>`. `<lock>` only
  commit-and-stops once thought is closed.
- SYS table: use the thinking block; tags inside steer reasoning now; tags after
  `<channel|>` steer the final answer.

## Findings

- Unit tests pass: live-stream (no settle on 100/101), open-thought lock does
  not stop, closed-thought lock does, history keeps thought+answer, SYS contains
  the thinking line. No GPU receipt yet. Do not claim P12 is fixed until a
  matched arm on a rebuilt CUDA binary.

## Next

- Rebuild the CUDA binary (hash will change from `62c018a8…`). One matched
  official-10 or a Prompt-12-only chat is the check: mouth continues past
  `<|channel>thought`, tags inside thought fire `hand_fired` before
  `<channel|>`.
- Do not switch packing to `enable_thinking=true` in the same cell. Current
  gen-prefix is still empty-thought (thinking-off); she reopens the block. That
  packing change is a later one-variable mutation.
