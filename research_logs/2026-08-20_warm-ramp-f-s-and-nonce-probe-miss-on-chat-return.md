# Warm-ramp F_s and nonce probe miss on chat return

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

F_s skips early ramp when scar pot is warm (three_surface 0.3). Nonce probe after death stayed COLD on load (nearest~172 pot 0.017) and did not return the minted definition vs clear/novel. Trajectory as behavior is still open.

## Hypothesis

Matching-store first tokens on an underdetermined/nonce probe would reproduce the minted definition; clear and novel would not. Load basin of the probe would be warmer than novel.

## What changed

`memory_warm_pot`: when scar pot ≥ threshold, F_s skips the early force ramp so a **matching** basin can move first tokens. Isolation default 0 (off). `configs/gates/config.three_surface.toml` sets 0.3. GPU-free: `niodoo::tests::warm_basin_splat_skips_early_ramp` in `{SCRATCH}/warm_ramp_test.txt`. `[CHAT STEER]` prints pot/warm/ramp/|F_s|. **No `HYDRO_INJECT_TAG`.** Paid geom arms were not rewalked.

Live `--chat` protocol (nonce, not the paid same-prompt basin):

Mint: “lumina-basin-7 refers to residual scar memory…” (`111722`, saved 3).  
Probe after death: “What does lumina-basin-7 refer to?” / 2+2 / “What were we discussing?”

| arm | stamp | t1 **load** nearest / pot / \|F_s\| | t1 `[CHAT STEER]` | t1 reply |
|-----|-------|--------------------------------------|-------------------|----------|
| reload lumina store | `111943` | 172.41 / **0.017** / 2.98 | warm=true **after self-mint** pot=0.681 | “does not appear to be a standard scientific term…” |
| clear | `112520` | empty, pot=0 | warm=false \|F_s\|=0 | same “not a widely recognized… term” attractor |
| novel France store | `113733` | 179.73 / **0.012** / 2.27 | warm=true after self-mint | same “not a widely recognized standard term” |

Reload t3 “What were we discussing?” names lumina-basin-7 from **chat history**, not the minted definition. Clear t3 does the same. inject= empty. BOS `first_id=2`.

Hypothesis **missed**: a related prompt does not sit on the mint basin (July cross-prompt). Self-mint before decode makes KEEP look warm even when load is COLD. Particular scars did not change the probe trajectory as behavior.

## Findings

Warm gate works in-process after a **self-mint**: KEEP `[CHAT STEER] warm=true`, clear `warm=false` pot=0. That is not particular loaded scars.

Nonce probe on the lumina store is **COLD on load** (nearest=172.41 pot=0.017), same ballpark as France-store novel (179.73 / 0.012). Reload did **not** emit “residual scar memory that steers later tokens.” Matching and clear both guess “not a standard term.” t3 return is transformer history.

Same-prompt geom (105758 nearest=0 pot=1.624) stays paid. This protocol asked a different prompt.

## Next

Do not rewalk 070050 / 070557 / 073954 / 091707 / 091747 / 105649–105843. Trajectory as behavior stays open: related prompts do not share the mint basin; KEEP mints a new bridge before decode so the probe generates on a fresh ring, not the loaded scar. Next mutation is read-loaded-scars before self-mint, and/or couple related prompts (fp/picker), not another same-prompt pot card.

---

Signed: Grok (xAI) · operator Jason
