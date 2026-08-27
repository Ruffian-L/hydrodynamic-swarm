# d-run seat forces endocrine off

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

--d-run now forces endocrine off (same function tests drive), keeps residual physics required and hooks off, and honors --tokens so a short diagnostic cannot become 131072. We think the next visible generation will have no [ENDOCRINE] lines.

## Hypothesis

Forcing endocrine off on --d-run will silence enzyme/bloom; the rest-of-the-rest lock at step 125 may still appear because it preceded the first endocrine fire.

## What changed

`--d-run` now applies `d_run_seat_policy` (same function the tests call):

- endocrine **forced off** even without `--no-endocrine` (print: `Endocrine: OFF (--d-run)`)
- residual physics **required** (`seat.physics_required`)
- hooks **off** after `--set` so sliders cannot re-enable them
- `--tokens N` is honored (short diagnostic cannot be overwritten to 131072)
- bare `--d-run` still defaults to 131072
- EOS mask uses `seat.eos_masked`
- `logs/D_gemma.jsonl` `tokens_target` is the actual `max_tokens`, not hardcoded 131072
- Chat / `talk.sh` without `--d-run` still has endocrine ON (IMMUTABLE_RUN_CONTRACT)

Tests (shipped function, GPU-free):

```
cargo test --offline --bin hydrodynamic-swarm d_run_
  generation_tests::d_run_forces_endocrine_off_physics_required_hooks_off ... ok
  generation_tests::d_run_honors_short_token_budget ... ok
  generation_tests::chat_without_d_run_keeps_endocrine_on ... ok
cargo test --offline --bin hydrodynamic-swarm generation_tests::no_endocrine_flag_still_works_off_d_run
  ... ok
```

Capture: `/tmp/grok-goal-9902ae0e9d81/implementer/d_run_policy_test.txt`

Path B sources were protected *before* this diff (`research_logs/2026-08-20_path-b-surface-protection-check-done.md`).

Signed: Grok (xAI) · operator Jason

## Findings

Four GPU-free tests on the shipped `d_run_seat_policy` passed (`d_run_policy_test.txt`). `--d-run` without `--no-endocrine` has `endocrine_enabled=false`, `physics_required=true`, `hooks_enabled=false`. `--tokens 200` stays 200. Chat without `--d-run` keeps endocrine on.

## Next

Short visible `--d-run --tokens 200` if CUDA can start; otherwise mark live close blocked. Do not start 131072.

---

Signed: Grok (xAI) · operator Jason
