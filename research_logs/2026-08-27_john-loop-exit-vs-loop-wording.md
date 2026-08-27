# John loop-exit vs loop-wording contract

> Date: 2026-08-27
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

John sent a stronger experiment for the Gemma 4 / Jacobian-lens loop traces. The causal question is loop **wording** vs loop **exit**. He ordered cheap contracts first so a steering sweep cannot stay ambiguous.

This log maps that protocol onto the live tree vs the public chat skip. It is not a GPU result.

## What changed

- Protocol: `docs/experiments/loop-exit-vs-loop-wording.md`
- Chat telemetry in `generate_turn_ex`: `hidden_delta = ‖h'−h‖` after pullback, `logit_delta = ‖z'−z‖` from residual project only. `[CHAT DELTA]` on step 0. `event=tok` JSONL carries both. HUD `logit_delta` is no longer `None` on chat.
- Runner (not executed): `scripts/steer_plumbing_check.sh`

Did not: run A on GPU (card at 96%). Did not add a J-derived direction. Did not run C–F. Did not merge public/sol-cuda. Did not call hydro FD or the prefill goal attractor "J-space." Did not identify this with the Gemma 4 one-neuron enumeration-loop paper. Did not commit — this tree already carries weeks of unrelated dirty work.

## Hypothesis

John's order is the right one. On **this** tree, the public `let _ = engine` chat skip is already gone: `--chat` goes through `engine.steer` and `project_to_logits` when `steer_hidden && force_cap>0`. The remaining cheap hole was B: chat logged `delta_h_norm` (steering mag before pullback) and left logit displacement blank, so a later c-sweep would still mix coordinates.

We think: once GPU is free, A on `generate_turn_ex` will pass for **hydro residual**. That still does not license a J-direction causal claim. E needs a vector in host residual D and a separate A for that add. John's E4B proxy already shows why D matters: ordinary logits ranked `repetitive` 21→5→1 while J-space was not cleaner.

## Findings

Code, not a run:

| Claim | Public / sol-cuda | Live 3surface |
|-------|-------------------|---------------|
| Chat dispatch skips `engine.steer` | yes (`let _ = engine`) | no (`generate_turn_ex` steers) |
| `delta_h_norm` | oneshot / engine | yes, but it is ‖steering‖ not ‖h'−h‖ after pullback |
| Chat `logit_delta` | n/a on light chat | was `None`; now residual-only ‖z'−z‖ |
| J-derived direction on chat | no | no |
| Real J-lens on GGUF decode | no | `jlens-gguf` is f32-dequant Jacobian of Q4 weights, not deployed piecewise-constant path |

John's E4B proxy (his numbers, not ours): visible collapsed history is sufficient for some self-diagnosis; J-space rank of `repetitive` moved with history; ordinary final logits were at least as early. Issue #5 input-copying still applies if `loop` / `STOP` / `hallucination` are in the prompt.

`three_surface.toml` `topic_logit_mix=0.65` is a scar-trail logit blend. It must be frozen (preferably 0) on any E sweep or exit cannot be attributed to a J-direction.

## Next

1. When the GPU is free: `./scripts/steer_plumbing_check.sh` (A for hydro residual, 12B default, 1–2 tokens, `--chat`).
2. C on an E4B / HF proxy is cheaper than 31B Candle. Unprimed history before any J-rank of `loop`.
3. D only with a real lens on the same model as the seat. Hydro `src/jacobian.rs` is not it.
4. E only after A passes **for the J-add**, with reverse/random same-norm controls, scoring F (durable exit) not next-token wording.

Signed: Grok (xAI)
