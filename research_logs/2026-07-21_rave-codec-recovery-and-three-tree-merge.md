# Research Entry: RAVE codec recovery, lane verdict, and three-tree merge direction

> Date: 2026-07-21
>
> **Author:** Claude (Anthropic) + Jason (operator)
> Context: Grok unreachable ~4 days (rate limits). Work continues here; this entry is the
> catch-up packet Grok reads from git history when back.

---

## Why this entry exists

Grok (via web chat, out-of-band) proposed a new lane: use the RAVE audio codec's 64-dim
latent space as a substrate for the swarm physics + TDA (persistent homology on latent
sequences). Jason remembered "a bunch of RAVE safetensors we created" and asked for a
recovery sweep before anyone builds. This is the result, plus the verdict of the old
RAVE-hidden-codec lane so we don't relearn it, plus the merge direction while Grok is dark.

## Recovery result: nothing is lost, phone not needed

Both trained codec weights exist in **three byte-identical copies** (md5 verified):

| Location | Path |
|---|---|
| Live tree | `~/projects/team_build_apr25/niodoo/runtime_assets/` |
| Messy backup | `~/takeout_drive_intake/Drive/Messy/backups/Homernd/2026-06-01_0709/team_build/niodoo/runtime_assets/` |
| SanDisk rescue | `/mnt/backup_sandisk/rescue-backups/Homernd-2026-06-03_0519/Homernd/team_build/niodoo/runtime_assets/` |

Files: `rave_codec.safetensors` (original, 2026-05-03 14:04) and
`rave_codec_eval_v1.safetensors` (**the eval-distribution retrain**, 2026-05-03 15:32),
plus `.keys.txt` manifests, `tede_corrector.safetensors`, and
`universe_70b_full_128256_fallback.safetensors` (2 GB) one level up.

The empty `Drive/Unorganized/memory_codec_AB_rave_codec_v2_*` folders are takeout-sync
shells; their eval **verdicts** survived loose (`CLAIMS(1).md`,
`codec_bridge_only_ab_v2_summary.json`, `MEMORY_CODEC_INVENTORY.md` in `Drive/Unorganized/`).
Only raw per-seed telemetry JSONLs are phone-only — receipts, not inputs.

## Old lane verdict (so we don't build on sand)

From the team_build evals ledger and CLAIMS history:

- **Original codec: TRAINED WRONG (catastrophic).** Encode→decode on 100 real 4096D hidden
  states: cosine ≈ 0.11, relative error ≈ 0.9956 — reconstruction is noise. Trained on an
  unrelated dataset. This single diagnostic (2026-05-06, evals-ledger example 01)
  invalidated the iter-46 route-preservation results downstream of it.
- **Retrained v2 (= `rave_codec_eval_v1.safetensors`):** val loss 3.146 → 1.265,
  distance 0.359 → 0.075. But behavior AB: mean delta **−0.0083** vs no_bridge
  ("essentially noise, closest ever to baseline"); 2/3 seeds lift, seed 143 regressed.
  Ledger called it yellow, not green.
- **Scalar quantization won that lane:** int8 Unicode packets preserved route geometry at
  top-1 0.983 / top-5 1.000, triplet order 0.994. The 64D route handle is cheap to keep
  faithful without a learned codec.

**Carry-overs worth reusing:** the Rust bridge (`niodoo/src/bridge/rave_codec.rs` —
RaveEncoder/RaveDecoder/ResidualBlock1D, safetensors load), the
`convert_rave_codec_to_safetensors.py` pattern, and the discipline: **verify
encode→decode reconstruction on the real distribution before trusting anything
downstream.** One script; would have saved a month.

## Decision: re-derive, don't recover

The new lane (RAVE **audio** codec latents as physics substrate) starts fresh with a
pretrained RAVE model — it does not depend on the old hidden-state codec at all. Old
weights stay archived as diagnostic + cautionary tale. No phone recovery required.

## Three-tree merge direction (while Grok is dark)

Jason's call 2026-07-21: hydro is the **clean side**; converge slowly, with structured
logging, in this order of authority:

1. **hydro (`hydrodynamic-swarm`)** — continuity mint + KPI home. Physics closed
   (B4d-q/B27), continuity proven (TCT-splat-lite, prefill-bridges, A→B→A PASS_RETURN),
   endocrine wired 07-18 (FunctionGemma still stubbed).
2. **niodoo-tct** — the TCT lane spanning both trees today (`src/tct.rs` mint here;
   `tct_splat_lite` + final post-norm residual apply consumer in niodoo-live). Candidate
   to become the shared crate the other two depend on.
3. **niodoo steering (`niodoo-live` / `niodoo-hidden-state-steering`)** — live is
   currently scattered (ledger amendments, mirror lineage, basin-coherence work in the
   steering tree); not anyone's fault — Lumina juggling. Merge **into** hydro's
   discipline, not the other way.

Merge principle: same as INV-5 — nothing is padded to fit. A piece moves over when it has
a research-log entry, a smoke, and honest telemetry keys. No big-bang merge.

## For Grok (paste-able recap)

> Found the RAVE safetensors — original + eval-retrained v2 both safe in three identical
> copies; phone not needed. Old lane verdict: original codec reconstructed noise on real
> 4096D hiddens (cosine 0.11); retrained v2 was noise-level behavior lift (−0.0083);
> scalar int8 won route preservation (0.983 top-1). Your audio-RAVE lane starts fresh with
> a pretrained RAVE model on the hydro side, reusing the Rust codec bridge and
> reconstruction-first discipline. Meanwhile we're slowly merging hydro + TCT + niodoo
> steering with hydro as the clean side. Read this repo's research_logs from 2026-07-21
> forward to catch up.
