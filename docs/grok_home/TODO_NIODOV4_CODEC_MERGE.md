# STICKY TODO — niodv4 codec merge (do not forget)

**Pinned:** 2026-08-03  
**Owner:** Jason · Grok co-engineer on cold start  
**Status:** OPEN — unfinished play, not deleted work  
**Lineage:** all of **niodv4** led into Niodoo bridge / team_build → this stack

---

## The thing we never played out

**Mixed multi-codec bake-off** — secret sauce + TEDE + RAVE (+ CodebookVQ), **try them all at once**, not one winner by vibes.

Deep dive: **2026-05-03_179** “The Latent Transport Layer”  
Roadmap: **P3-B** / `codec_consensus` (reframed from MSE-pick to correspondence)

Same 64D (and 4096D for RAVE) state → three+ transports:

| Codec | Space | Job |
|--------|--------|-----|
| **CodebookVQ** | 1 of 256 centroids | discrete route bucket |
| **Secret sauce V3** | 64 Unicode glyphs (`target_z_unicode_v3`) | near-lossless pack |
| **RAVE** | 64 ↔ 4096 | learned hidden force / specialist path |
| **TEDE** | dipole expert | correction specialist (niodv4 train → Rust port) |

**Intent:** measure whether codecs **agree** on state identity; run **causal AB** (SS only / RAVE only / TEDE only / mix / consensus-as-gate); merge what earns green.

---

## What already exists (pieces ≠ match)

Live tree: `~/projects/niodoo-hidden-state-steering/niodoo/`

| Piece | Path | Done? |
|--------|------|--------|
| Secret sauce v1/v2/v3 | `src/runtime/secret_sauce_codec.rs`, `bridge/secret_sauce.rs` | encode/decode + packet field |
| `target_z_unicode_v3` | correction_packets + mint / CLI `--correction-packet-out-unicode-v3` | opt-in transport |
| CodebookVQ | `bridge/codebook.rs` | in bridge |
| RAVE | `bridge/rave_codec.rs` + `--rave-codec-path` | load + smokes; real-hiddens reconstruct **failed** vs scalar |
| TEDE | `bridge/tede_corrector.rs` | port + load smoke |
| **codec_consensus** | `bridge/codec_consensus.rs` | **math + unit tests only** — *no live routing* |
| Weights | team_build `runtime_assets/rave_codec*.safetensors` (backup copies) | not lost |
| Hydro | Jul-30 pick bridge | **64→2560 RAVE parked** |

Empirical already on record:

- Scalar / unicode route preserve **won** (~top-1 0.983) vs trained RAVE collapse.
- Secret sauce **transport** yes; **causal generation lift** never locked green.
- Joint multi-arm campaign **never run**.

---

## When to reopen (order of gates)

1. **jlens greens** (stance / first-thought instrument) — current priority.  
2. Then **architect Hydro** (Gemma 4 transition) with multi-address memory in mind.  
3. Then **this TODO**: multi-codec bake-off as its own campaign — do not silently drop.

Optional later bridge: J-keys **index** basins; codec stack **packs/pulls** 64D/4096D. Different ridges; can meet.

---

## Concrete bake-off when we pull this sticker

Fixed prompts · honest yellow/green · no product-default claims:

- [ ] SS only  
- [ ] VQ only  
- [ ] RAVE only (if weights + real-dist reconstruct re-check)  
- [ ] TEDE only  
- [ ] Mix / all on  
- [ ] `codec_agreement_64d` (+ `with_rave`) as **gate/telemetry**, not MSE-pick  
- [ ] Write one research_log + CLAIMS row per arm  

**Do not** re-myth the stack as if it already merged.

---

## Cold-start one-liner (paste)

> niodv4 multi-codec unfinished: secret sauce + TEDE + RAVE + VQ were supposed to be **tried together** via correspondence (`codec_consensus`). Pieces exist; joint play never ran. Sticky: `docs/grok_home/TODO_NIODOV4_CODEC_MERGE.md`. Reopen after jlens green / Hydro arch — not forgotten.

---

## Related

- Hydro: `research_logs/2026-07-21_rave-codec-recovery-and-three-tree-merge.md`  
- Hydro: `research_logs/2026-07-30_pick_bridge.md` (RAVE parked)  
- Niodoo: `src/bridge/codec_consensus.rs` module docs  
- Grok session 2026-08-03: unicode_z_v3 = `target_z_unicode_v3`; status yellow on causality  

## Provenance correction (2026-08-03)
**Multi-key clustering** and **niodoo/TCS into this pipeline** = Jason invent. Grok implements/remembers. See `PROVENANCE_TEAM.md`.
