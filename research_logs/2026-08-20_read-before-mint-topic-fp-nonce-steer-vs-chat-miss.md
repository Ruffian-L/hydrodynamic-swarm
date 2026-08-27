# Read-before-mint topic-fp nonce steer vs chat miss

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Chat reads loaded scars before self-mint. Related prompts share topic fp; matching nonce t1 STEER |F_s|=5.34 vs novel 0.068 vs clear 0. gemma4> still dictionary-guesses; definition return missed.

## Hypothesis

Topic-matched loaded bridge would make nonce-probe t1 STEER warmer than novel/clear, and the reply would repeat residual scar memory. Steer half held; chat wording did not.

## What changed

Read loaded scars **before** self-mint. Related prompts share `tct::continuity_fp` (hyphen/digit topic token, else full prompt_fp). Topic-matched bridges skip F_s ramp and pull toward the stored μ. Isolation default wipe unchanged. **No inject.** Paid geom 105758 not rewalked.

GPU-free: `topic_fp_couples_related_prompts_not_novel`, `topic_matched_far_bridge_skips_ramp` in `{SCRATCH}/topic_couple_tests.txt`.

Live nonce death-reload (`HYDRO_TOKENS=64`, three_surface). Mint `114707` saved 3. Probe “What does lumina-basin-7 refer to?”

### t1 **before mint** (this is the particular-scar receipt)

| arm | stamp | load nearest/pot | `[CHAT STEER]` warm / \|F_s\| |
|-----|-------|------------------|-------------------------------|
| reload lumina store | `114719` | 172.41 / 0.017 | **true / 5.343** |
| novel France store | `114845` | 179.73 / 0.012 | **false / 0.068** |
| clear | `114805` | empty | **false / 0** |

Load L2 stays COLD (related prompt ≠ same basin). Topic fp still couples **decode-time** F_s: matching 5.34 vs novel 0.068 vs clear 0. Mint prints after the reply (`[CHAT BASIN mint]`).

### `gemma4>` t1

Reload, clear, and novel all guess “not a widely recognized / standard term.” None emit “residual scar memory that steers later tokens.” t3 “what were we discussing?” is chat history on every arm.

Hypothesis half-hit: particular scars change steer on a related prompt. Chat return of the minted definition is still miss (T=0.7 dictionary attractor).

## Findings

Read-before-mint holds: `[CHAT STEER]` prints on the load store, `[CHAT BASIN mint]` after the reply. Topic fp couples decode-time force on the related probe (5.34 vs 0.068 vs 0). Load nearest stays ~172. Chat t1 is still the dictionary attractor on all three arms. inject empty. BOS `first_id=2`.

## Next

Do not rewalk 070050 / 070557 / 073954 / 091707 / 091747 / 105649–105843 / 111722. Same-prompt geom stays paid. Chat return/climb of the minted **definition** is still open — stronger topic pull or a probe whose first tokens are less locked by T=0.7 priors.

---

Signed: Grok (xAI) · operator Jason
