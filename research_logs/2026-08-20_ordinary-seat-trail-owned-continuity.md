# Ordinary-seat trail-owned continuity vs clear vs novel vs sequence

> Date: 2026-08-20
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Ordinary three_surface --chat persists decode-trail residuals without HYDRO_KEEP_MEMORY=1. Matching reload owns minted token ids and stops at trail end. Sequence after intervening+fail returns lumina; aurora is a second scar; clear/novel do not quote. Isolation wipe unchanged. No inject.

## Hypothesis

Making persist the ordinary three_surface seat and letting matching trails own token ids will quote minted definitions after death, across intervening and failed turns, without KEEP and without soup on matching reads.

## What changed

Ordinary full-stack `--chat` on `configs/gates/config.three_surface.toml` now persists residual trails **without** `HYDRO_KEEP_MEMORY=1`. Isolation wipe flags stay the isolation baseline. Matching decode **owns** the minted token ids and **stops** when the trail ends (closes residual→the slip and extra gloss). `commit_decode_trail` is the shipped keep rule: a later failed write of the same fp does not replace. Distinct topic fps no longer eat each other’s prefill-bridges by L2. No `HYDRO_INJECT_TAG`.

GPU-free (`{SCRATCH}/trail_ordinary_unit.txt`): `decode_trail_commit_keeps_existing_and_two_fp_roundtrip`, `chat_decode_trail_write_save_load_step_mu`, `distinct_fp_bridges_do_not_replace_by_l2`. three_surface flags persist without KEEP; isolation still wipes; `KEEP=0` is the measured clear arm.

Live `--chat`, `keep_memory=` empty, BOS `first_id=2`, inject empty:

| arm | stamp | t1 `gemma4>` |
|-----|-------|----------------|
| mint (no KEEP) | `125640` | lumina + aurora definitions; `[CHAT TRAIL mint] n=19` and `n=15`; saved 34 trail steps |
| **reload** | `130355` | “The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens.” `own=yes` `[CHAT TRAIL own] stop n=19` `[CHAT TRAIL keep]` |
| clear `KEEP=0` | `130445` | dictionary “does not appear to be a widely recognized public standard…”; no TRAIL; `--clear-memory` |
| novel | `130537` | “The capital of France is Paris.” no TRAIL load |
| **sequence** | `130657` | 2+2 → `4`; zed-anchor-9 dictionary fail; lumina exact return; aurora “second residual scar” (not lumina); lumina still exact |
| long 128 | `131008` | two readable residual-stream paragraphs; named collapse classes gone |
| long 256 | `130845` | same opening, then late `esese` lock — not TheThe/theed/Wait/math-thrash |

Reload t2 `2+2 = 4` not hijacked. Matching trail kept. Sequence is intervening + fail + return + second scar + first still holds.

## Findings

Matched on continuity as the ordinary seat. Matching death-reload quotes the minted definition **including residual**, stops at 19 trail tokens (no gloss), keep_memory empty. Clear is dictionary. Novel is Paris. Sequence after 2+2 and a failed zed-anchor-9 probe still returns lumina; aurora is a second particular scar and does not leak lumina; a later lumina probe still returns lumina. inject empty. BOS `first_id=2`. Isolation wipe unchanged.

256-token unmatched generation hits a late `esese` lock. 128-token bound and the 5-turn sequence do not. Named TheThe/theed/Wait/math-thrash stay gone. Ordinary mint stored two trails (34 steps) even when L2 replace left one prefill-bridge; trail-own does not need that bridge.

## Next

Do not rewalk this ordinary-seat 3-arm or sequence as unpaid. Isolation default wipe stays. Unmatched long generation past ~128 tokens can still lock; that is not this continuity receipt.

---

Signed: Grok (xAI) · operator Jason
