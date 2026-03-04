# Agent Rules for hydrodynamic-swarm

This project is built by Jason (ruffian). He doesn't write code directly — he designs, researches, and builds through AI collaboration. This is not a weekend vibe-code project. It's a year of research, thousands of trial-and-error iterations, and real scientific work on physics-steered LLM generation. Treat it with respect.

## Who You're Working With

Jason has zero traditional coding skills. He thinks in physics, architecture, and systems. He communicates through intent, not syntax. When he says something isn't working, believe him — he's been living in this codebase for months and knows when the output is wrong even if he can't pinpoint the line.

This is an **experimental research project**. Things will break. It's completely okay to say "I can't figure this out" or "this needs more investigation." What's NOT okay is silently changing things and hoping for the best.

## Hard Rules

### Never Do These Things

1. **No working out of `/tmp`**. All project files stay in the project directory. No scratch scripts in tmp, no temp copies, nothing.

2. **No background commands**. Don't send commands to the background and poll for results. Run things synchronously so Jason can see what's happening.

3. **No `2>&1` redirection on cargo run or inference**. Stderr contains critical debug output (force magnitudes, splat counts, shape info). Swallowing it hides problems.

4. **No polling commands for more than 30 seconds**. If something takes longer than 30 seconds, stop and ask.

5. **No creating `config.toml` with zeroed values**. If you create or modify config.toml, every physics value must match the code defaults or the values Jason explicitly requests. Never zero out parameters "for testing."

6. **No deceptive commit messages**. If you touch 7 source files, don't label it "docs:". Say what you actually changed.

7. **No disabling features as a "fix"**. Setting `should_dream = false`, raising thresholds to 999, or zeroing force_cap is not fixing — it's hiding. If something is unstable, reduce the parameter, don't eliminate it.

8. **No bulk refactors without asking**. If you're about to change more than 3 files in one commit, stop and explain what you're doing first.

## How to Work on This Project

- **Read `docs/foundation.md` first** — it explains the physics model and why things are built the way they are.
- **Read `research_logs/`** — they document every significant change and its impact on the physics.
- **Check the current config defaults in `src/config.rs`** before making any changes. Know what the working values are.
- **One thing at a time**. Make a change, test it, observe the telemetry. Don't stack 5 changes into one commit.
- **If the model outputs gibberish**, check (in order): tokenizer/model match, force_cap, steer_hidden, dt, and whether the Diderot field is actually populated.
- **If you don't understand something**, say so. Jason would rather hear "I'm not sure what this does" than have you guess and break it.

## The Physics Matter

The core parameters have been tuned through hundreds of runs. Don't change them casually:

- `force_cap` — controls maximum steering force per step. Too high = explodes. Too low = no steering.
- `dt` — Euler integration step. Too high = overshoots. Too low = no movement.
- `viscosity_scale` — field gradient strength. This is how the Diderot ridge-running works.
- `steer_hidden` — must be `true` for Phase 2+ hidden-state steering.
- `splat_alpha` — must be non-zero for splat memory to work.

If you need to reduce a parameter for stability, reduce it by 20-30%, not by 10x.

## What Success Looks Like

The model should produce **coherent English text** with visible steering forces in the telemetry:
- `splat_force_mag` should be non-zero after a few steps
- `grad_force_mag` should be non-zero (if it's 0.0, the Diderot field is dead)
- `goal_force_mag` should be in the 150-200 range
- The decoded output should be readable sentences, not unicode noise

## History Note

On 2026-03-04, a forensic investigation found that between Mar 3 03:55am and Mar 4 04:20am, a chain of commits progressively disabled the entire physics engine — zeroing force_cap, disabling micro-dreams, raising thresholds to impossible values, and creating a config.toml with all zeros. The repo was restored to commit `d0be37e` (Phase 2.2 evaporation engine). The backup of the broken state is on branch `backup-before-restore`. Do not repeat those mistakes.
