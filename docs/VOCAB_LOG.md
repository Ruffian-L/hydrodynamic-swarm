# Vocab log — hydro × gemma-lab × TermSplat

**Full team map (canonical):** [`docs/VOCAB.md`](VOCAB.md) — preferred terms, force names, picker, files.

Started **2026-07-25** after: universe on disk, endocrine blooms on **native** geometry, TinyEmbed out.

Rule: names should be scannable in a log line. If you can’t spot PASS from the word, rename it.

---

## Core insight (why this log exists)

| Old vibe | Clean read |
|----------|------------|
| “FACT” = fake text stub | FACT **text** can still be stub; FACT **place** is not |
| Retrieval = string match / hash embed | Retrieval = **where the fact sits in native model geometry** |
| Universe = random side file | Universe = frozen **token landscape** (positions ± mass/charge) |
| TinyEmbed | **Dead path.** Do not revive for production endocrine |

**Oh, right:** when a bloom fires, we don’t invent a second geometry. We mean-pool the bloom’s tokens through the **same** `tok_embeddings` the field uses (Gemma: × √d pre-layer). So “facts” are anchors in **native geometry** — better mental model for retrieval than “another embedding API.”

---

## A–Z (living)

### Bloom
Text result from the endocrine enzyme worker. Geometry is **not** computed in the worker. Main applies **native embed** → Monolith.

Log: `[BLOOM native] …`

### ContinuousField / Diderot field
Gaussian density over token (or memory) positions in embedding space. Live path: built from **model `tok_embeddings`**. Offline path: can load universe `positions` (gemma-lab / niodoo).

### Endocrine
Hormone path: pain / high-δ → signal → enzyme text → bloom → **native** Monolith → Eureka window in steer.

Log: `Endocrine: ON (text worker + native model embed — no TinyEmbed)`

### Eureka
Short truth window after a Monolith: cooler viscosity, impulse decay (~0.92×/token), optional **pull toward native bloom vector**.

### FACT
Enzyme text label (still often `[FACT #n]` stub). **Not** the geometry. Geometry = native mean emb of that string. When real FunctionGemma lands, only the **text** quality changes; the embed path stays native.

### FieldFrame
TermSplat contract: one tick of entropy weather (`tick`, `entropy`, `tier`, `splats`, …). Swarm / lens dumps → terminal paint. Not a second brain.

### gemma-lab
`pheonix_squad/gem/gemma-lab` — offline Gemma geometry lab. Universe extract + topological heartbeat. **Study** landscape; live endocrine uses the **running** model (dim may differ from 26B lab universe).

### Monolith
High-mass truth injection. `pos4` = projected telemetry; full-D **native** target drives Eureka pull when present.

Log: `Monolith applied … native=true`

### Pain budget
Soft backstop: `max_pain_splats` + `max_pain_mass`. Prefer **dissipation** (`online_decay_rate`, `pain_decay_factor`↑) over hard cull. Log: `[PAIN BUDGET] dropped …`.

### Pleasure answer
After `pleasure_answer_after` pain deposits in a row → soft **+α near goal**. Log: `[PLEASURE ANSWER]`. Heart of anti-snowball: pleasure answers pain.

### Manifold pullback (“leave the wound”)
`physics.manifold_pullback` — each step blend residual toward pre-steer baseline. Not a second ghost model: **geometry home** so the walk doesn’t drift off-manifold after pain. Raise slightly when grammar frays under pull.

### Missing (control loop — mostly patched 07-25)
1. ~~Pleasure answer~~ → wired  
2. Force vs scar — still when F_s > 0  
3. Leave the wound → `manifold_pullback`  
4. Vanilla A/B → `scripts/ab_vanilla_vs_hydro.sh`

### Native geometry
The live model’s token embedding matrix as the physics body:

1. Raw rows from `token_embeddings` (V × D)
2. **Gemma pre-layer scale:** × √`hidden_dim` (same as forward)
3. Mean-pool tokens of a string → (D,) attractor

Not hash. Not TinyEmbed. Not a second model unless you **are** that model.

### Pre-layer scale
Gemma habit: embeddings enter the stack already scaled by √d. Bloom embed must do the same or it sits on the wrong shell.

### Pre-layer vs post-layer (two roads)

| Road | What it is | Feels like | Your stack today |
|------|------------|------------|------------------|
| **Pre-layer** | Token table → (Gemma: ×√d) → **before** transformer blocks | Dictionary / map of the vocabulary | Field from `tok_embeddings`; bloom mean-pool + √d |
| **Post-layer** | Hidden state **after** blocks (residual / last-token hidden) | Where the thought actually is while generating | Steering residual; scars in residual-ish space; “wake” because residual ≠ emb shell |

They diverge on purpose:

- **Pre-layer** answers: *what does this string mean in the model’s token geometry?* Good for retrieval anchors, universe maps, “where is this FACT.”
- **Post-layer** answers: *where is generation right now?* Good for steering, entropy weather, pain/δ, “is the walk settling.”

Interview one-liner: *“Embedding table is the map; residual stream is the hiker. Same mountain, different coordinates — L2 norms don’t match; that’s why pure ∇ρ on residual underflows and we use field wake.”*

Not mystical: different **layers of representation**, same model. You pick the road by the job (retrieve vs steer), not by belief.

### Retrieval (geometry read)
Finding or re-anchoring meaning = locating / pulling toward a region of **native** space (field, scars, monolith, goal), not only matching surface strings. Universe = map of the token terrain; live residual = where you are walking.

### Rot dial / entropy weather
TermSplat: high H → terminal goes feral; clean → geometry settles. Mirror of swarm micro-dream thresholds (~3 / ~4).

### Splat / learned will
Gaussian memory in field space (+α attract / −α repel). **Public name: learned will** (not scar; not poison).  
Wire/jsonl may still say `scar_*` until schema migration — see `docs/VOCAB.md`.  
Persistence: `splat_memory.safetensors` (+ TCT).

### SplatLens
Eye: `--viz` → `.viz.json` museum. TermSplat `lens` paints the same dump as weather.

### TinyEmbed
**Retired** for endocrine. Was hash → fake 4D. Do not use for “facts.”

### Universe
Safetensors landscape of token geometry, usually:

- `positions` (N × D)
- `mass`, `charge` (optional physics)
- `*_token_map.json`

Examples:

- Lab: `…/gemma-lab/universe_gemma_26b_top60000.safetensors` (2816-D, top 60k)
- Niodoo: `niodoo-live/universe_top60000.safetensors`

**Offline map.** Live run still prefers the model that is speaking unless dims/tokenizer match.

---

## Log line cheat sheet (PASS scan)

| Line | Means |
|------|--------|
| `Endocrine: ON (… native model embed — no TinyEmbed)` | Path is correct |
| `[ENDOCRINE] signal sent at step …` | Hormone fired |
| `[BLOOM native]` | Text arrived; main will / did native-embed |
| `Monolith applied … native=true` | Eureka has real (D,) geometry |
| `termsplat lens — … H∈[…]` | Weather pipe alive |

---

## Live pipe (us, stateless)

| Piece | Path |
|-------|------|
| Hydro weather out | `logs/<session>.termsplat.jsonl` + `logs/latest.termsplat.jsonl` |
| TermSplat paint | `termsplat pipe …/latest.termsplat.jsonl` or `--follow` |
| Skip weather | hydro `--no-termsplat` |
| Enzyme text (optional) | `ENDOCRINE_URL=http://host:port/v1` · `ENDOCRINE_MODEL=…` · else `[FACT #n]` stub |
| Geometry | always native tok_embeddings on main (not the enzyme host) |

“I” in speech = **us**. Scars and logs are the memory; no claim theater required.

## Open (not redefined yet)

- Gemma **4** loader in hydro (still Gemma-3 harness)  
- Whether field itself should store **pre-layer-scaled** positions vs raw matrix (today field loads raw; blooms scale)  
- Enzyme sidecar script (optional; env is enough)

---

## How to add a word

1. One short definition  
2. What it is **not**  
3. One log line or file path if it has one  
4. Date if the meaning flipped  

*Lab is for geometry study. Production geometry is the model that is speaking. Facts live there too.*
