# Model-size scaling doc + Gemma-3-4B prime

**Date:** 2026-07-11  
**Authors:** Jason (Algo_WIPjuly / co-engineer) · Grok (xAI) (swarm mapping + prime)

---

## Question

Does Algo_WIPjuly still apply when we scale down? Do we need it to prime a small model? What should we share with others on size changes?

## Answer

**Yes — still applies.** The √-law and type multipliers are the size/type prior. We mapped them into current swarm knobs and fixed a script bug that was crushing small-model caps by anchoring to 27B instead of the **3B golden mid-zone**.

| Artifact | Path |
|----------|------|
| Share doc (main) | `docs/MODEL_SIZE_PHYSICS_SCALING.md` |
| Documents copy | `Documents/MODEL_SIZE_PHYSICS_SCALING_swarm.md` (+ `.md` alias) |
| Generator | `scripts/scale_physics_for_model.py` |
| Algo source | `Documents/Algo_WIPjuly.md` (+ cross-link footer) |
| 4B default config | `config.toml` (scaled 4B instruct) |
| Launcher | `run_swarm.sh` → `gemma-3-4b-it-Q4_K_M.gguf` |

### 4B instruct suggested knobs (script)

```
scale=1.155  type_mult=0.9  intensity=1.039
Algo: σ=0.156 θ=2.08 β=115.5 rep=2.08
force_cap≈3.1  ramp 15@0.15  logit α=0  targeted_splat_only=true
```

Near golden mid-zone (as expected: 4B instruct ≈ 3B standard mass × 0.9 type).

### What still helps when scaling down

1. **√(params/3B)** — smaller → gentler  
2. **type_mult** — thinking 0.4 / coding 0.27 often beats size  
3. **Hard clamps** — chaos / frozen boundaries  
4. **Do not paste 27B ceilings onto 1B–4B** — 1B gibberish path  
5. **Extra after Algo_WIP:** nearest-emb wake, force ramp, targeted splat, quality ocean, loader arch gate

### What is *not* in the √-law alone

- Residual vs emb **geometry** (F_g dead zones)  
- `splat_sigma` scar width (geometry, not force mass)  
- Prompt/control-surface prefill (Qwen35 lesson)

---

## 4B smoke (prime)

```text
model: data/google/gemma-3-4b-it-Q4_K_M.gguf
arch:  gemma3  hidden=2560  blocks=34  heads=8
field: mean_dist=1.38  sigma_auto=7.59  (emb shell for 4B)
prompt: Physics of Friendship
tokens: 45  clear-memory  ramp 15@0.15  targeted ON
```

**Output (coherent, a bit soft at the end — expected at 45 tok):**

> Friendship, surprisingly involves a complex interplay of physics and social dynamics akin to physical interactions between two individuals— it not just several aspects like gravity-like behavior is more than mutual attraction forces that can be understood as an concept friendship

**Telemetry notes:** Pleasure splats fired (p≈0.7–0.95, δ≈100–120). Ocean deposits ran. No gibberish / no Fason. Field auto-sigma much tighter than 27B residual-era intuition — **geometry retune separate from force scale**.

Log: `logs/2026-07-11_22-58-26_gemma3-27b_v3-forcecap3_T0_8_s40_a1_d30.jsonl`  
(variant tag still said `gemma3-27b` on that run; fixed to `gemma3` in source after.)

---

## Next when iterating small model

1. Optional: 50-tok smoke + force means from live log  
2. Re-run `./learning_lane_ablation.sh` with `MODEL=...4b...`  
3. If F_g weak: wake geometry, not more force  
4. If garble: lower intensity / longer ramp — **not** 27B caps

---

**Authorship:** Jason · Grok (xAI) — failures logged so size-changes don’t re-walk 1B gibberish / 27B over-govern.
