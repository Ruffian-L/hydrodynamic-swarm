# Tag table without one-tag forbid

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Her Official 10 suggestion: list available tags and meanings. Runtime now teaches the full table, allows several tags in one turn, and drops one-tag / do-not negative prompts. Packing check no longer keys off exactly-one.

## Hypothesis

A positive tag table without a one-tag cap lets her emit multiple hands when she needs them; negative one-tag language was hiding the channel.

## What changed

- Gemma’s Official 10 runtime suggestion was a list of available tags and meanings. The system turn already had a table, then told her “exactly one tag” and a stack of do-nots.
- GOD_TIER_SYSTEM now leads with Available tags and meanings. Several tags in one turn are fine. No “one tag.” No “do not emit / do not narrate.” Packing PRESENT keys off the table + `<spike>`, not “exactly one tag.”
- Locked Official 10 user prompts unchanged. GPU-free packing tests green.

## Findings

Packed `--chat` now contains `Available tags and what they do` plus `<spike>` `<explore>` `<focus>` `<reset>` `<remember>` `<lock>`, and `Several tags in one turn are fine`. It does not contain `exactly one`, `at most one`, `do not emit`, or `do not narrate`. Units: `god_tier_system_is_system_turn_and_forbids_narration`, `gemma4_chat_packing_includes_tag_table_not_one_tag_forbid`. Official 10 user prompts stayed locked.

## Next

Do not rewalk `090107` as unpaid. Next live chat is where several tags can show up in one turn and stay in history.

Signed: Grok (xAI) · operator Jason

