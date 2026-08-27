# Qwen ChatML and Long Horizon RoPE Limits

> Date: 2026-08-27
> Agent: Antigravity
> Repo: hydrodynamic-swarm-3surface

## Context

Fix tokenizer/prompt fallbacks and long horizon RoPE limits for Qwen

## Hypothesis

The model wasn't loading the right tokenizer because talk.sh forced the CLI argument to point to an adjacent leftover Llama 3 tokenizer file. Even if it had loaded the right tokenizer, the REPL was feeding it the wrong chat template and would have crashed immediately upon exceeding 8192 tokens because the RoPE frequency precomputation (precompute_freqs_cis) was hardcapped by MAX_SEQ_LEN.

## What changed

- Patched `scripts/talk.sh` to prioritize `data/qwen.tokenizer.json` before forcefully bypassing the engine's fallback logic with `$(dirname "$MODEL")/tokenizer.json`.
- Added Qwen ChatML (`<|im_start|>`) and EOS (`248046`, `248044`) support to `format_multiturn_prompt_ex` in `src/main.rs`. Before this, Qwen was falling through to the generic `User: / Assistant:` prompt and producing gibberish.
- Audited the codebase for long-horizon gotchas and bumped `MAX_SEQ_LEN` from 8192 (and 4096 for Llama) up to 131072 across all model variants (`qwen35.rs`, `llama.rs`, `gemma.rs`, `gemma4.rs`).

## Findings

(open)

## Next

(open)
