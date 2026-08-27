# Chat prefill-bridge load-basin return vs novel vs clear

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Prefill-bridge mint at the chat query site; death-reload matching basin is warmer on LOAD than novel and than clear. No HYDRO_INJECT_TAG. splat_mag-on----- is not this receipt.

## Hypothesis

Reload of session scars at the matching return prompt is nearest~0 / high pot / high |F_s| on [CHAT BASIN load]; novel prompt on the same store is far/cold on load; clear later turns stay F_s=0 with no mint. T=0.7 wording is not the KPI.

## What changed

Prefill-bridge mint at the chat query site (`mint_chat_prefill_bridge_at` → `deposit_prefill_bridge`) when `mint_wills`/`HYDRO_KEEP_MEMORY`. `[CHAT BASIN load]` is printed **before** mint so novel cannot look WARM from a same-turn self-mint. GPU-free: `chat_prefill_bridge_query_near_not_far_or_empty` in `{SCRATCH}/chat_basin_query.txt`. Isolation wipe flags still default. **No `HYDRO_INJECT_TAG`.** Isolation `070050`, full-stack `070557`, Path B inject `073954`, mint-reload splat_mag `091707`/`091747` were not rewalked.

Four-arm live `--chat` on `configs/gates/config.three_surface.toml`, residual.force_cap=1 `enabled_path=true`, inject= empty, BOS `first_id=2` every turn:

| arm | stamp | keep | store |
|-----|-------|------|-------|
| clear | `20260820_105649` | wipe + no-save | skip save (8 RAM wills die); later turns `pot=0 \|F_s\|=0` **(no mint)** |
| session | `20260820_105715` | KEEP | mint in-process; saved 10 |
| reload | `20260820_105758` | KEEP | **Loaded 10**; matching return prompt |
| novel | `20260820_105843` | KEEP | **Loaded 10**; capital-of-France prompt |

### `[CHAT BASIN load]` (this is the receipt, not splat_mag-on-`----`)

| arm | turn | nearest | pot | \|F_s\| |
|-----|------|---------|-----|--------|
| reload t1 | matching basin | **0.00** | **1.624** | **17.0530** |
| novel t1 | same 10 scars, other prompt | **156.94** | **0.041** | **4.3901** |
| clear t2 | no persist | 182.06 | 0.000 | 0.0000 |
| session t2 load | gen deposits still far | 182.06 | 0.011 | 2.0174 |
| session t1 after mint | in-process bridge | 31.50 | 0.664 | 20.9012 |

Reload after mint: pot=1.875. Novel after self-mint: pot=0.705 nearest=31.50 — that post-mint line is **not** the death-reload comparison.

### `gemma4>` replies (not the `grep gemma4>` prefill banner)

- reload t1: `A residual stream is the evolving sequence of hidden states that carries information through the layers of a transformer model during inference.`
- reload t3 return: `A residual stream is the primary pathway of hidden states that flows through a transformer's layers to accumulate and carry information.`
- novel t1: `The capital of France is Paris.`
- clear t1 (same return prompt, empty store): `A residual stream is the sequence of hidden states that flows through a transformer's layers, accumulating and refining information at each step to produce the final output.`

T=0.7 wording is **not** the KPI. Clear and reload both produce residual-stream sentences on the same prompt. Geometry on **load** is what distinguishes matching scars from novel and from clear.

## Findings

Reload matching basin on **load** is WARM (nearest=0, pot=1.624, |F_s|=17.05). Novel on the same 10 scars is COLD on load (nearest=156.94, pot=0.041, |F_s|=4.39). Clear later turns stay F_s=0 with no mint. In-session KEEP mint produces |F_s| after mint; same-turn gen deposits still sit far at the next prefill until a new bridge is minted (session t2 load nearest=182). Chat wording at T=0.7 does not separate reload from clear. inject empty. Continuity receipt is `[CHAT BASIN load]`, not Path B inject and not splat_mag on `----`.

## Next

Do not rewalk 070050 / 070557 / 073954 / 091707 / 091747 / this geom (105649–105843). Force-in-revise still adjacent. 131k still not this lane. Chat-text return/climb is unpaid as a KPI; geometry is paid.

---

Signed: Grok (xAI) · operator Jason
