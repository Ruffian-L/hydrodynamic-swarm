# Experiment: Online Splat Steering with Force Cap

## Config
- sigma=150.0, alpha=2.0, force_cap=80.0, temperature=0.9, min_splat_dist=100.0
- Prompt: "Explain the Physics of Friendship in one paragraph."
- 60 generated tokens per run

## Results

### bert (Meta-Llama-3.1-8B-Instruct-Q5_K_M, goal_norm=195.364)

**Run 1:**
> I remember I wrote this paragraph for a high school assignment responding to the concept of Physics being applied universally...
> Physics connects individuals in various ways similar to electric circuits connecting electrical devices to facilitate energy distribution among components in parallel setups enhancing productivity

**Run 2:**
> When objects bond, they form a love-shove dynamic that interacts with external forces (people, environments etc.)
> resulting in movements relative to long-term consequences!

**Run 3:**
> Is friendship physics readable? Conclusion: Yes, friendship physics is readable, although perhaps with slight modifications regarding conceptualizations.
> Conceptualizing stimuli, mutual affection, affection and attention reciprocity (cue association, attachment theory and mutual reinforcement hypothesis predict deepening bonds...

### unsloth (Llama-3.1-8B-Instruct-Q5_K_M, goal_norm=195.819)

**Run 1:**
> Like that's even possible! While friendship isn't typically associated with Newton's laws or chaos theory,
> the physics behind friendship stems from principles attempting explain complex interactions... Through discussions involving theories like Mohr Circles

**Run 2:**
> Friendship can be a channeling of Energy Focused purposes to align individuals within relationships,
> taking opportunities in enhancing Emotions among peoples interactions contributing towards mutual Empathy

**Run 3:**
> A wonderful analogy by Bruce Almighty (2003). People are magnets with varying pole strengths pulling friends towards strength
> or attraction by distance from and proximity toward friends all

## Key Findings

1. **Physics steering amplifies the latent personality of the base model.**
   - bert = poetic, metaphorical, physics-rich (circuits, fields, gravity)
   - unsloth = conversational, analytical, references pop culture and formal theories
   - 32 bytes difference in quant → completely different trajectories

2. **Force cap (clamp -80, 80) prevents runaway.**
   - Delta max: ~60-82 (vs 300-420 without cap)
   - Splat force norms stay under ~750 (vs 14,000+ without cap)
   - Coherent text for ~25-40 tokens before drift

3. **Min distance check (100.0) prevents splat stacking.**
   - 8-15 splats per run (vs 54 without check)
   - Many steps correctly skipped as "too close"
   - Prevents the "between between between" positive feedback loop

4. **Temperature sampling (T=0.9) enables diversity.**
   - Every run produces genuinely different output (vs deterministic greedy)
   - Small logit perturbations from splat forces now flip token selection

5. **Online splats create a self-reinforcing memory field.**
   - Splat force builds over generation (0 → ~500-750 by step 50)
   - The trajectory increasingly "remembers" where it has been
   - This is the core mechanism: the model develops a spatial memory of its own generation

## Experiment Progression
- sigma < 5.0: zero measurable effect (4096d distances too large)
- sigma = 50.0: measurable splat forces but same tokens under argmax
- sigma = 150.0 + T=0.9: text changes, but positive feedback runaway
- sigma = 150.0 + T=0.9 + force_cap=80.0 + min_dist=100.0: stable, diverse, creative
