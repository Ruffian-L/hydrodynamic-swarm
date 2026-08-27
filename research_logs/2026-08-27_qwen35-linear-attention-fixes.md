# 2026-08-27 — Qwen3.5 Linear Attention Fixes

We did: Ported Qwen3.5 hybrid architecture into `hydrodynamic-swarm-3surface`. We fixed a prefill crash caused by `forward_linear_attn_static` explicitly bailing on sequences `> 1` token, replacing it with a sequential token loop. We also fixed severe numerical bugs:
1. GQA keys/queries were interleaved improperly using `cat` instead of `repeat_interleave`, scrambling the heads.
2. The keys and queries were L2-normalized but then divided by `sqrt(d)`, pushing the attention magnitude to $1 / (d\sqrt{d})$, which we changed to multiply by `sqrt(d)`.
3. The `softplus` activation used `(x.exp() + 1).log()`, which overflows `f32::MAX` to `inf` when `x > 88.7`, randomly wiping the state. Fixed with a numerically stable masked `minimum(20.0)`.

We think: The combination of the wrong head matching, microscopic QK-norm scale, and softplus overflow caused the `ssm_state` (and output) to degrade over time. Re-scaling it to $O(1)$ variance, fixing the GQA head matching, and masking `softplus` restores the state precision without precision loss in `f32`.

Next: Re-run a measured evaluation on a long context to verify the Qwen3.5 model maintains generation coherence over long horizons.
