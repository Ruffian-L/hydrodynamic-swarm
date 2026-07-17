# Live Gemma 3 baseline + residual TCT

**Date:** 2026-07-16  
**Status:** landed — load, chat template, physics hook, TCT dim-match smoke

## What shipped (niodoo-live)

| Piece | Detail |
| --- | --- |
| Backend | `niodoo/src/physics/gemma.rs` — Gemma 3 GGUF + `forward_physics` (post-attn_norm sacred hook) |
| Dispatch | `ModelArchArg::Gemma3`, auto-detect `gemma3`/`gemma2`/`gemma` |
| Chat | `ChatTemplateArg::Gemma3` (`<start_of_turn>user/model`, stop `<end_of_turn>`/`<eos>`) |
| Tokenizer | `tokenizers` **0.19 → 0.22** (required for Gemma tokenizer.json) |
| Assets | symlinks under `niodoo-live/model/gemma-3-4b-it-Q4_K_M.{gguf,tokenizer.json}` → hydro `data/google/` |
| Smoke script | `niodoo-live/scripts/gemma_tct_smoke.sh` |

## Verified smoke (CPU)

```
[LOADER] GGUF architecture: Gemma3
[Gemma3] heads=8 kv_heads=4 blocks=34 hidden=2560 head_dim=256 rope_seq=8192
[UNIVERSE] ... dim=2560 (Model hidden=2560)
[TCT] Loaded 28 residual scars (2 prefill_bridge) dim=2560 ... bridge_only=true
```

Telemetry: `tct_n_considered=2`, `tct_nearest_dist≈2.9e3`, force 0 on first-visit (LOCALITY COLD — expected until same residual basin as hydro deposit).

## How to run

```bash
cd niodoo-live/niodoo
RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features niodv4_bridge

# or: ./scripts/gemma_tct_smoke.sh
niodoo/target/release/niodoo \
  --model-path model/gemma-3-4b-it-Q4_K_M.gguf \
  --tokenizer-path model/gemma-3-4b-it-Q4_K_M.tokenizer.json \
  --model-arch gemma3 --chat-template gemma3 \
  --tct-splat-path ../hydrodynamic-swarm/data/splat_memory.tct \
  --tct-splat-bridge-only \
  --context-length 8192 --physics-end-layer 33 \
  --system-prompt-mode free --max-steps 24
```

## Still not “universal”

- Llama remains fully supported; Gemma is a second production backend.
- RAVE / 4096 packet codec is still Llama-shaped — use residual TCT on Gemma, not packet JSONL ported from Llama.
- Gemma 4 / gemma3n need separate loaders (not this path).
- Same-prompt continuity warm requires matching residual basin (revisit hydro bridge prompt on live).

## Non-claims

No feelings. Continuity = measurable residual geometry + load/apply KPIs.
