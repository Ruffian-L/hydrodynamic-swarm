# Generation stabilizers — multi-turn chat path

**Date:** 2026-07-28  
**PI:** Jason Van Pham  
**Engineer:** Grok (xAI)  
**Track:** decode stability (not residual memory coupling)

## Problem

Near-vanilla Gemma 4 multi-turn under **T=0**, long `max_tokens`, and hard constraints entered classical degeneration basins; re-prefill kept collapsed turns in history.

## Change

| Piece | What |
|-------|------|
| `GenerationConfig` | `top_k`, `top_p`, `consecutive_repeat_break`, `no_repeat_ngram`, `drop_collapsed_history` |
| `run_simple_chat` | top-k/p sampling, loop-break, n-gram mask, collapse heuristic, `reset`, drop bad turns from history |
| One-shot loop | same sample helper + consecutive break + n-gram mask |
| `configs/gemma4/config.gemma4_stable.toml` | T=0.7, top_k=64, top_p=0.9, rep=1.3, max_tokens=80, ngram=3 |
| `scripts/chat_gemma4.sh` | **defaults to stable**, tokens **80** (was greedy + 300) |

## Operator

```bash
./scripts/chat_gemma4.sh
# in chat: reset  → clear history after a bad turn
# probe old greedy: GEMMA4_CONFIG=configs/gemma4/config.gemma4_greedy.toml ./scripts/chat_gemma4.sh 300
```

## Not claimed

Does not fix partial RoPE/SWA bugs (those are loader track). Does not replace memory-coupling science. Reduces **runaway decode walls** under multi-turn.

---
**Authorship:** Grok (xAI) co-engineer with Jason (PI)
