# 2026-07-30 — John A0: SWA prefill row-validity gate

**Source:** John6666 review on HF (Gemma 4 multi-turn diagnostic)  
**Workbench:** `hydrodynamic-swarm-3surface` · `src/gemma4.rs`  
**Status:** geometry fixed + unit gate green  

## Defect (static code, confirmed)

Local layers trimmed K/V to the last 1024 positions **before** building a
`[q_len, kv_len]` mask over the **full** query length. Arithmetic:

| Prefill | Retained K/V | Empty valid-key rows |
|--------|--------------|----------------------|
| 1023 | 1023 | 0 |
| 1024 | 1024 | 0 |
| 1025 | 1024 | **1** |
| 1039 | 1024 | **15** |
| 2048 | 1024 | **1024** |

Those rows softmax over empty support → NaN / pad-class risk (mlxcel #401 family).
Not proven as the sole cause of the published 31B transcript — but the geometry
was real and still present on master until this fix.

## Fix

| Phase | Policy |
|-------|--------|
| **Prefill** (`seq > 1`) | Keep full K/V. SWA via **mask only**. |
| **Decode** (`seq == 1`) | Trim rotating cache to window (memory). Single query at end → all retained keys valid. |

## Gate (no GPU)

```bash
cargo test --bin hydrodynamic-swarm a0_swa
```

- `john_table_legacy_trim_empty_rows` — documents the old arithmetic  
- `a0_fixed_prefill_no_empty_rows_at_boundary` — 1023/24/25/39/2048/4096  
- `valid_keys_self_consistent_with_window` — count = min(i+1, W)  

Helpers: `valid_keys_per_query`, `legacy_trim_prefill_valid_keys`,
`fixed_prefill_valid_keys`, `empty_valid_key_rows`.

## Live gate CLI

```bash
cd /home/ruffianl/hydrodynamic-swarm-3surface
./target/release/hydrodynamic-swarm \
  --model data/google/gemma-3-4b-it-Q4_K_M.gguf \
  --tokenizer data/google/tokenizer.json \
  --a0-swa-check
```

### Live result 2026-07-30 (Gemma 3 4B, CUDA GB10)

| Check | Result |
|-------|--------|
| A0a static legacy table | PASS (empty rows match John) |
| A0a static fixed geometry | PASS (0 empty at 1023–1039) |
| A0b logits finite @ 1023/24/25/39 | PASS (~0.43–0.47 s each) |
| A0b hidden finite | PASS |
| **Overall** | **PASS** |

### Live result 2026-07-30 (Gemma 4 **12B**, card SWA=1024)

Source: `/media/ruffianl/ghost_team/models/gemma-4-12b-it-Q4_K_M.gguf`  
Loader: `swa_win=1024` · hidden=3840 · 48 blocks

| Check | Result |
|-------|--------|
| A0a static | PASS |
| A0b finite @ 1023/24/25/39 | PASS (~1.6–1.7 s each) |
| **Overall** | **PASS** (card-faithful) |

```bash
./target/release/hydrodynamic-swarm \
  --model /media/ruffianl/ghost_team/models/gemma-4-12b-it-Q4_K_M.gguf \
  --tokenizer data/google/gemma4_assets/tokenizer.json \
  --a0-swa-check
```

Also on disk: `bart_google_gemma-4-31B-it-Q4_K_M.gguf`, `unsloth_gemma-4-31B-it-Q4_K_M.gguf` under the same models dir.

### Live result 2026-07-30 (Gemma 4 **31B Dense**, card SWA=1024)

Source: `/media/ruffianl/ghost_team/models/bart_google_gemma-4-31B-it-Q4_K_M.gguf`  
Loader: `swa_win=1024` · hidden=5376 · 60 blocks · heads=32

| Check | Result |
|-------|--------|
| A0a static | PASS |
| A0b finite @ 1023/24/25/39 | PASS (~4.1–4.5 s each) |
| **Overall** | **PASS** (incident-class size) |

This is the size class from the published multi-turn diagnostic. Geometry fix holds under full 31B prefill.

## Still open (John ladder)

- First-token logit parity vs Transformers or llama.cpp on identical token IDs  
- Template / token parity for multi-turn  
- A1–A6 one-knob decode controls after A0 green  
- Three-arm pick→hydro coupling (Prediction 3)  

## Team

Jason (lead) · John6666 (review) · Grok (wire) · Claude (HUD / picker elsewhere)
