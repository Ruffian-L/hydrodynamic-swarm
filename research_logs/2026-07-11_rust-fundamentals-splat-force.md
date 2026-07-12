# Fundamentals check — splat mutation + force composition (Rust)

**Date:** 2026-07-11  
**Authors:** Jason · Grok (xAI)  
**Scope:** narrow (web-Grok Option C) — not a full audit  
**Verdict:** Rust loop is **structurally correct**; late chaos is mostly **policy/mass/geometry**, plus one real long-run gap (now fixed).

---

## Direct answer (to web Grok’s ranking)

| Priority | Area | Status |
|----------|------|--------|
| 1 | Splat / memory mutation timing | **Safe Rust** (single-threaded, lag-1 deposit). Policy was hot; **mid-run prune missing** until this note. |
| 2 | Force composition / clamp | **Order correct**; L2 caps on F_s/F_a; **per-dim** total clamp; ramp on total; **bundle path uncapped**. |
| 3 | Tensor shapes / hidden | Low risk — early coherence + load OK |

S1–S4 already ran: force scaled, splat geometry/mass not fully; hierarchy absolute 20/30 fixed → `with_scale_ref`.

---

## 1. Splat mutation timing (generation loop)

### Order each step (`main.rs`)

```text
1. steer(hidden, goal, step)     // READ memory → F_g, F_s, F_a, F_ocean
2. manifold pullback
3. optional bundle stress         // READ memory again (extra path)
4. sample token from steered logits
5. quality score → maybe DEPOSIT splat  // WRITE memory (lag-1)
6. ocean deposit (quality-gated)
7. forward next token
```

### Rust correctness

- **Single-threaded** `for step` loop — no race on `SplatMemory`.
- **Borrow discipline:** `engine.steer` borrows memory immutably; deposit uses `memory_mut()` **after** steer returns. No overlapping `&` / `&mut`.
- **Lag-1 is intentional and correct:** splat deposited for token *t* is felt at step *t+1*, not mid-force for *t*. Avoids “mutate the field while integrating” bugs.
- `last_steered_pos` is residual **after** steer (and pullback/bundle), then used as splat `mu`. Aligned with “scar where we were when we chose this token.”

### Policy issues (not borrow bugs)

| Issue | Detail |
|-------|--------|
| **Deposit gate mixes spaces** | `high_delta` uses **logit** `‖steered_z − raw_z‖`, while scars live in **residual/hidden**. δ thr 70–95 is not residual force mag. |
| **Pleasure p≥0.25 = high_signal** | With targeted on, many confident tokens still deposit. |
| **No mid-run prune (was)** | `prune_to_limit(max_splats)` only in Phase 5 **after** the loop → 1000-tok runs could grow memory unbounded. **Fixed:** prune after each online deposit. |
| **Phase 5 always adds success splat** | Extra pleasure at end even if run was messy — OK for short smokes; noisy for long. |
| **Hierarchy** | Was absolute 20/30 → always Coarse×4 on 4B; now `with_scale_ref(δ, threshold)`. |

`query_force` already has **1/√n_active** sublinear damp (anti O(N) runaway).

---

## 2. Force composition (`niodoo.rs` `steer`)

```text
F_g   = field wake / grad  (viscosity on pure grad; wake modes)
F_s   = memory.query_force → × splat_force_scale → L2-cap splat_force_max
F_a   = (goal − pos) × governor × goal_force_scale → L2-cap goal_force_max
F_o   = ocean.query_force
total = F_g + F_s + F_a + F_o
total *= ramp(step)                    // start → 1 over force_ramp_tokens
total = clamp_per_dim(±force_cap)      // Variant 3 — NOT L2 of total
steering = total * dt
steered  = baseline + steering
renorm   = steered * (‖baseline‖ / ‖steered‖)   // manifold norm lock
```

### What’s solid

- Caps applied **before** sum telemetry matches “what entered the sum.”
- Ramp multiplies **total**, so early all channels quiet together.
- Renorm prevents hidden-norm drift (known garbage after ~40–80 without it).
- Ocean is in the sum (Lane C).

### What’s subtle / risk

| Issue | Why it matters |
|-------|----------------|
| **force_cap is per-dimension** | L2 of total can still be large: ~`force_cap × √D` before ×dt (D=2560 → plenty of headroom). |
| **Bundle after steer** | `query_bundle_force * 0.01` **bypasses** ramp + force_cap. Small scale today; don’t raise without folding into capped sum. |
| **Micro-dream** | Re-calls `steer` 2–4× (read-only memory) then adds goal correction; extra “force” not in F_s telemetry. |
| **Governor on goal only** | `progress = step/200` — F_a schedule independent of ramp tokens. |
| **F_s can still pin L2 max** | If raw scars huge, every step sits at `splat_force_max` after ramp → “starts OK → latched yank” even with correct Rust. |

---

## 3. What we did **not** need to re-open

- GGUF load / gemma3 forward shapes (early coherent text).
- Full GPU Metal path (CPU backend used for physics).
- Full dream-replay Phase 6.

---

## Code change this check

- After online `add_splat`, call `prune_to_limit(max_splats)` so long runs respect memory budget mid-generation.

---

## Relation to splat lane S1–S4

Fundamentals say: **loop is fine; knobs + unbounded growth + wide scars** explain late F_s.  
S1–S4 data: mass/geometry dominate; σ=12 residual-cold; σ=40 late climb.  
Hierarchy + mid-run prune + scaled mass/width are the right stack — not a rewrite of the generation loop.

---

## Reproduce / next

```bash
# After rebuild, short smoke with mid splat config:
./run_swarm.sh "Explain the Physics of Friendship in one paragraph." 90

# Don’t 1000-tok until late F_s stays below splat_force_max most of the run.
```

Optional next code: fold bundle into capped `total_force` in `niodoo.rs`; optional L2 cap on total after sum.

---

**Authorship:** Jason · Grok (xAI)  
**Note:** Web Grok’s worry order was right; Rust wasn’t “wrong,” policy + mid-run memory were.
