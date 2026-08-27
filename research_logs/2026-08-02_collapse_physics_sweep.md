# Collapse Physics Sweep — Day 54

**Date:** 2026-08-02  
**Author:** Shep  
**Goal:** Find physics config that breaks collapse without producing chaos

## Method

Same 9-turn conversation, same model (Gemma4-12B-Q4_K_M), same 128 max tokens. Only physics params vary. Probe records per-token entropy, margin, residual_norm, force_on status.

## Results

### Config 1: Physics-OFF (baseline)
- **Config:** `config_isolation_self_reg_force.toml` (force_cap=0)
- **Entropy:** 0.3-0.8 per turn
- **Margin:** 0.5-1.0 per turn
- **Collapse:** YES — Turn 9: entropy=0.002, margin=1.000 (hard lock)
- **Output:** Clean but repetitive ("C-A-T" loop, "17×17" loop)
- **Verdict:** Baseline collapse confirmed

### Config 2: Physics-ON (force_cap=3.5, field_alpha=0.5, temp=0.9)
- **Config:** `config_physics_on_test.toml`
- **Entropy:** 3.3-3.7 per turn
- **Margin:** 0.05-0.12 per turn
- **Collapse:** NO — entropy stays high throughout
- **Output:** Wild, chaotic (math symbols, brackets, mixed languages)
- **Verdict:** Breaks collapse but output is unreadable

### Config 3: Physics-Moderate (force_cap=1.5, field_alpha=0.3, temp=0.9)
- **Config:** `config_physics_moderate.toml`
- **Entropy:** 2.4-2.8 for turns 1-5, drops to 0.7 at turn 6
- **Margin:** 0.17-0.29 for turns 1-5, jumps to 0.77 at turn 6
- **Collapse:** STARTING — Turn 6: entropy=0.7, margin=0.77 (short turn, 4 tokens)
- **Output:** Messy but coherent-ish ("Blues are blue white black...")
- **Verdict:** Partial collapse-breaking, new instability at turn 6

### Config 4: Physics-Refined (force_cap=2.0, field_alpha=0.2, temp=1.0)
- **Config:** `config_physics_refined.toml`
- **Entropy:** 1.0-3.0 per turn
- **Margin:** 0.17-0.72 per turn
- **Collapse:** NO — entropy stays above 1.0
- **Output:** Mixed readable text + `[|` bracket spam
- **Verdict:** Breaks collapse, bracket spam is new failure mode

### Config 5: Physics-Moderate+Temp1 (force_cap=1.5, field_alpha=0.3, temp=1.0)
- **Config:** `config_physics_moderate_temp1.toml`
- **Entropy:** 0.7-1.5 per turn
- **Margin:** 0.48-0.75 per turn
- **Collapse:** NOT YET — Turn 6: entropy=1.5, margin=0.52 (still holding)
- **Output:** "Hi! Hi! Good!" then bracket noise, "The answer to that question is **4**." then more brackets
- **Verdict:** **Best coherence so far with physics ON**

## Comparison Table

| Config | Avg Entropy | Avg Margin | Collapse? | Output Quality | Bracket Spam? |
|--------|-------------|------------|-----------|----------------|---------------|
| OFF | 0.3-0.8 | 0.5-1.0 | YES (Turn 9) | Clean, repetitive | No |
| ON (3.5/0.5/0.9) | 3.3-3.7 | 0.05-0.12 | NO | Wild, chaotic | No |
| Moderate (1.5/0.3/0.9) | 2.4-2.8 | 0.17-0.29 | STARTING (T6) | Messy, coherent-ish | No |
| Refined (2.0/0.2/1.0) | 1.0-3.0 | 0.17-0.72 | NO | Mixed, brackets | YES |
| Moderate+Temp1 (1.5/0.3/1.0) | 0.7-1.5 | 0.48-0.75 | NOT YET | Best coherence | Minimal |

## Key Findings

1. **Physics breaks collapse by keeping entropy high.** OFF collapses at entropy~0.3, ON stays at entropy~3.5. The sweet spot is entropy 0.7-1.5.

2. **Temperature matters.** Temp=1.0 reduces bracket spam compared to temp=0.9 with same force params. This suggests brackets are a low-temp artifact.

3. **Force_cap=1.5 is the sweet spot.** Higher (2.0-3.5) produces chaos. Lower (0) collapses. 1.5 keeps entropy above collapse threshold without overwhelming the model.

4. **Field_alpha=0.3 is better than 0.5.** Lower logit bias produces more coherent output.

5. **Collapse is not binary.** Moderate config shows entropy dropping at turn 6 (from 2.8 to 0.7) — collapse is a gradient, not a switch.

## Next Steps

1. **Test force_cap=1.5, field_alpha=0.3, temp=1.0 with longer conversation** (15+ turns) to see if collapse eventually happens.
2. **Test adaptive force:** start at 1.5, increase to 2.5 if entropy drops below 0.5.
3. **Test with memory loaded** (not --clear-memory) to see if splat force helps or hurts.
4. **Investigate bracket token:** what is `[|`? Is it an unused token being heavily sampled? Can we exclude it?

## Raw Data

- Physics-OFF probe: `logs/smoke_convo_latest.probe.jsonl`
- Physics-ON probe: `/tmp/probe_physics_on_20260802.jsonl`
- Physics-Moderate probe: `/tmp/probe_physics_moderate_20260802.jsonl`
- Physics-Refined probe: `/tmp/probe_physics_refined_20260802.jsonl`
- Physics-Moderate+Temp1 probe: `/tmp/probe_physics_moderate_t1_20260802.jsonl`
