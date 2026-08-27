# Easy parser and Official 10 KV (monitor + tags)

> Date: 2026-08-21
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason: run the Official 10 pack; Internal monitor and spike tags must enter KV so the model sees them; parser really easy on the model; give the tail command.

101526 already showed mouth monitor + `<spike>`/`<reset>`, but `[CONTROL_RECEIPT]` spliced between `<focus` and `>` because the hand fired before the completing token printed. History smash (`split_whitespace`) flattened monitor lines for the next prefill.

## Hypothesis

An easy `<>` parser (spaces, case, trailing slash, first word even if a receipt is glued on, bare `<lock>`) plus printing the tag before the receipt, plus keeping newlines in next-prefill history, plus a real KV forward of the monitor line, lets later Official 10 turns attend to both the mirror and the hands. `[CHAT PREFILL see]` is the receipt.

## What changed

- `first_hand_word`: `< spike >` `<SPIKE>` `<spike/>` `<focus\n>` `<focus [CONTROL_RECEIPT]…>` `<lock>` `<remember>` all parse. `</reset>` and `<turn|>` stay out.
- Fire hands **after** `on_token`, so the mouth shows `<focus>` whole, then the receipt.
- `gemma4_history_clean` no longer smashes to one line. Monitor lines and tags stay as-emitted for the next prefill KV.
- `[CHAT KV] internal_monitor tokens=N` on inject. `[CHAT PREFILL see] turn= monitor_lines= tags=` on every packed turn.
- `smoke_convo.sh` points `logs/smoke_convo_latest.txt` at the live file **at start**, so `tail -f` works during the run.
- Units: `parser_is_easy_on_the_model`, `ignores_htmlish`, `next_prefill_keeps_emitted_tag_for_attention`.
- Official 10: `./scripts/hydro_eval.sh check official-10` then `run`. Same locked pack. No inject. Wipe store. Did not move α.

## Findings

(check + live stamp filled after the run)

## Next

Watch `[CHAT PREFILL see]` on turn 2+: monitor_lines>0 and tags include spike/focus/reset if she used them. P12/P13 cycle-settle and P10 trail-own remain known risks. Do not move α.

Signed: Grok (xAI) · operator Jason
