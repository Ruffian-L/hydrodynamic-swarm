# Convo parity — only two scripts

| Who | Command | What |
|-----|---------|------|
| **Human (Jason)** | `./scripts/talk.sh` | Live multi-turn chat |
| **AI / team smoke** | `./scripts/smoke_convo.sh` | Multi-turn auto prompts → log |

**Shared settings:** `scripts/convo_defaults.sh`  
Same config, model, tokens, flags. Change once → both surfaces match.

## Why not one-shot

See `research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md`  
(we wrote this together): readable one-shot ≠ multi-turn ready.  
Smokes must be **multi-turn** (`--chat` with history). AIs can’t REPL like humans; **`smoke_convo.sh` is the durable stand-in** (stdin turns + `logs/smoke_convo_latest.txt`).

## Both look here after smoke

```bash
grep 'gemma4>' logs/smoke_convo_latest.txt
# full log:
less logs/smoke_convo_latest.txt
```

## Override (both scripts)

```bash
HYDRO_CONFIG=configs/gates/config.three_surface.toml \
HYDRO_MODEL=data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf \
HYDRO_TOKENS=80 \
./scripts/talk.sh

# same overrides for smoke:
HYDRO_CONFIG=... HYDRO_MODEL=... ./scripts/smoke_convo.sh
```

Default today: isolation baseline + Gemma 4 12B + 64 tokens (coherence / self-reg lane).

Other scripts in this folder are legacy/sweeps — **not** the parity path.
