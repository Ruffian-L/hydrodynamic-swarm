# Spike never stops; no incomplete tags

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason watching Official 10 104753: after Internal monitor she emitted `<spike>` and it looked stopped; later a bare `<spike` with no `>`. Parser must never treat spike as a stop. There must not be incomplete tags.

Probe: turn 1 `<focus>` then model EOS at step 145 (same family as Niodoo SPIKE-eot). Turn 2 spike at 306 did keep going; a later `<spike` had no close yet (tokenizer split).

## Hypothesis

Tight `<>` close (no spanning to math `>`), hold the mouth until `>`, strip leftover openers from history, and mask EOS for 64 tokens after a physics hand. Only LOCK stops.

## What changed

- `take_simple_hand`: `>` must sit on this tag. `<spike` is not a hit. `$\rightarrow$` cannot close it.
- `incomplete_open_hand`: do not stream `<spike` / `<foc` until `>`. Truncate leftovers off `pieces` at turn end.
- `ControlTag::stops_turn` is LOCK only. `fire_tag` uses that.
- After spike/focus/explore/reset/remember, EOS logits = −∞ for 64 tokens.
- Units: `spike_never_stops_and_incomplete_is_not_a_tag`.

## Findings

104753: opening focus→EOS is the stop miss. Hanging `<spike` is the incomplete miss.

## Next

Rebuild, restart Official 10 so Jason sees complete `<spike>` and keep talking.

Signed: Grok (xAI) · operator Jason
