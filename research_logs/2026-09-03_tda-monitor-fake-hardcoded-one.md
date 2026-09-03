# Hydro TDA monitor is fake (hardcoded-1); sidecar is the fix

> Date: 2026-09-03
> Agent: Grok (xAI)
> Repo: hydrodynamic-swarm-3surface

## Context

Jason, mid H5: the TDA monitor in hydro is fake, hardcoded 1 value; look at the TDA sidecar; add that to the list.

This is a list add, not a wire. H5 kept the decode-loop seat.

## What changed

Docs only: `ghost_team_groktodos.md` hole **DREAM-03**, `docs/PRODUCTION_BACKLOG.md` DREAM-03 rewrite + start-here row 8.

Did not spawn ripser. Did not rewrite `src/tda_monitor.rs`.

## Hypothesis

We think hydro's Internal monitor is token-repetition wearing TDA clothes, and niodoo-live's Python ripser sidecar on the hidden-state trajectory is the engine Jason already timed.

## Findings

**Hydro (`src/tda_monitor.rs`):** `observe` builds a 6-d point per token:

```
entropy, margin, residual_norm, splat_mag, p_top1, step_frac
```

Homemade Vietoris–Rips on that cloud. z-score + near-constant scalars → one blob → **H0 → 1**. The mouth line's `H0bars`/`H1sum` are that blob. The loud clause (`closed cycle around "meters"`) is `disposed_tokens` string count. Circle unit test uses 2-d `(cos,sin)` points, which is not what `--chat` feeds.

Hydro never calls `scripts/tda_python_monitor.py`.

**niodoo-live sidecar (the real monitor):**

- `scripts/tda_python_monitor.py` — ripser, JSONL, PCA-16, `maxdim=1`
- `niodoo/src/runtime/tda_monitor.rs` — `python_ripser_shape` default ON (`NIODOO_TDA_PYTHON=0` off); rust VR is `cfg(test)` + fallback
- `.venv-tda/`
- Receipt: `research_logs/2026-08-19_python-tda-monitor-back.md`
  - 32-pt circle: H1 persist 2.22 in 2–5 ms
  - live hidden window: `geometry_source=hidden_state_trajectory+python_ripser`, H1 bars=2 persist 0.501→1.660

Tab 4 in hydro already refuses canned H0=8/H1=21 (2026-08-27). That honesty is the frontend. The lie is the `--chat` mouth.

## Next

Wire the existing sidecar onto hydro `--chat`. Feed `surface_hidden` window, not the 6 scalars. Print `geometry_source=`. Keep rust VR labeled as test/fallback. Monitor stays in think (ORG-07). `--tda-breath` stays off.
