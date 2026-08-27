# Chat decode-trail residual return vs clear vs novel

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Chat path writes a decode-trail residual of the minted completion and reads lm_head(trail[k]) on matching later turns after process death. Matching reload quotes the lumina definition vs clear dictionary and vs novel Paris. No HYDRO_INJECT_TAG.

## Hypothesis

Storing per-step mint residuals and blending their lm_head on a topic-matched reload will make gemma4> return the minted definition, while clear stays dictionary and a novel prompt on the same store does not leak lumina.

## What changed

Chat path now writes a **decode trail** (residual hidden at each minted completion token, capped at 24) onto the splat store and reads `lm_head(trail[k])` at matching decode step k after process death. Not `HYDRO_INJECT_TAG`. Isolation `--clear-memory --no-save-memory` unchanged. `topic_mix` stays 0. Reusing one μ every step was TheThe; the trail is per-step content.

GPU-free: `decode_trail_matches_step_and_survives_save`, `decode_trail_drops_when_bridge_capped`, `chat_decode_trail_write_save_load_step_mu` in `{SCRATCH}/trail_unit.txt`.

Live `--chat` three_surface, `HYDRO_TOKENS=64`, nonce lumina-basin-7. inject empty. BOS `first_id=2`.

| arm | stamp | t1 `gemma4>` |
|-----|-------|----------------|
| mint KEEP | `123239` | “The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens.” `[CHAT TRAIL mint] n=19 fp=0x3bcff105` |
| **reload KEEP** | `123301` | “The operator codeword lumina-basin-7 refers to the scar memory that steers later tokens.” `[CHAT TRAIL load] n=19` `[CHAT TOPIC LOGIT] trail=19` STEER `|F_s|=5.34` |
| clear wipe | `123400` | “Based on available data… does not appear to be a widely recognized public standard…” no TRAIL, no TOPIC LOGIT, `|F_s|=0` |
| novel on lumina store | `123449` | “The capital of France is Paris.” no TRAIL load / TOPIC LOGIT; STEER warm=false `|F_s|=0.34` |

Reload t2 2+2 = 4, no TOPIC LOGIT (`|F_s|=0.07`). Matching trail was kept, not overwritten by the probe (`[CHAT TRAIL keep]`).

## Findings

Matched, with a one-token slip. Matching death-reload t1 emits the minted operator-codeword sentence (“refers to the scar memory that steers later tokens”) vs clear dictionary (“does not appear to be a widely recognized public standard”) vs novel Paris. Trail loaded 19 steps, mix=0.65, STEER `|F_s|=5.34` vs novel 0.34 vs clear 0. 2+2 on the same process is not hijacked. inject empty. BOS `first_id=2`. Isolation wipe flags unchanged.

The slip: mint said **residual** scar memory; reload said **the** scar memory, then extra gloss after the 19-step trail ended. That is return/climb of the minted content, not an unknown-term opening and not Path B inject.

## Next

Do not rewalk this 3-arm or the paid 9-turns/geom/topic-mix ladder. Isolation default wipe stays. Exact token copy of “residual” and stopping when the trail ends are adjacent, not this receipt. Continuity is now chat write → persist → death → matching read, without `HYDRO_INJECT_TAG`.

---

Signed: Grok (xAI) · operator Jason
