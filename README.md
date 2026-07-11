# Hydrodynamic Swarm

**Built with Llama** — runs [Meta Llama 3.1](https://llama.meta.com/llama-downloads) and [Google Gemma 3](https://ai.google.dev/gemma) model weights. See [License](#license) and [`NOTICE`](NOTICE).

**A working Rust harness for on-line, per-token vector-field steering of an 8B language model's residual stream, with a persistent on-disk memory of past hidden states that reloads across runs.**

The system intervenes in the Llama 3.1 forward pass at the pre-`lm_head` hidden state, computes a steering update from three sources — a continuous field over the model's own embedding matrix, a memory of Gaussian "splats" deposited on prior trajectories, and a goal attractor from the prompt — and writes the result back into the residual before sampling. The splat memory is persisted as `safetensors` so the generator carries spatial memory of its own history across sessions. No fine-tuning, no LoRA, no architectural change to the base model.

Every knob is in a TOML config, every step writes per-token JSONL telemetry, and every substantive change has a dated research-log entry pointing at the telemetry file it describes.

---

## What the system actually does

For every generated token, the loop in [`src/main.rs`](src/main.rs) and [`src/niodoo.rs`](src/niodoo.rs) does the following:

1. Run a vendored quantized Llama 3.1 8B forward pass ([`src/llama.rs`](src/llama.rs)) and read the pre-`lm_head` hidden state $h_t \in \mathbb{R}^{4096}$.
2. Compute three vectors in that same hidden space:
   - **Field gradient** $g_t$: gradient of a scalar field built over the model's own token-embedding matrix ($128256 \times 4096$), approximated with a Top-K probe ($K = 2048$ by default) to keep the cost tractable. See [`src/field.rs`](src/field.rs).
   - **Memory force** $s_t$: vector sum of Gaussian-kernel forces from a memory of stored "splats" $(\mu_i, \sigma_i, \alpha_i)$ deposited on previous trajectories. See [`src/memory.rs`](src/memory.rs), [`src/splat.rs`](src/splat.rs).
   - **Goal attractor** $a_t = e_{\text{prompt}} - h_t$, with the prompt's mean embedding as a fixed point.
3. Sum these with a momentum term and small Langevin noise; apply a per-step manifold pullback that subtracts the component of the steering update that has accumulated off the empirical Llama representation manifold. See [`niodoo::steer`](src/niodoo.rs).
4. Project the corrected hidden state $h_t' = h_t + \Delta t \cdot F_t$ through `lm_head`, apply a repetition penalty, and sample.
5. Optionally deposit new splats — a "pleasure" splat for low-surprise tokens, a "pain" splat for high-surprise tokens — with a minimum-distance guard so splats can't stack at one location.

Between sessions, splats are written to `safetensors` and reloaded on the next run, so the generator carries spatial memory of its own past output forward in time. This persistence is the actual experimental novelty — everything else in the loop is a re-implementation of well-known ideas (see *Related work*).

---

## Status

| | |
|---|---|
| **Maturity** | Active research code, v0.2. End-to-end loop is working; API and on-disk format are still moving. |
| **Base model** | Llama 3.1 8B Instruct, GGUF Q5_K_M, vendored via Candle 0.9 quantized loader |
| **Runtime** | Rust 2021 edition, Candle 0.9, `cudarc` 0.19, CUDA 13 toolkit |
| **Tested on** | NVIDIA Blackwell GB10 (`sm_121a`), aarch64 Ubuntu 24.04. Any CUDA GPU with ≥ 8 GB should work; the wgpu/Metal path in `kernels/` is sketched but not wired in. |
| **Codebase** | ~6.9 kLOC Rust across 15 modules; 42 unit tests in tree |
| **License** | MIT-0 (this repo's code). Model weights: Meta Llama / Google Gemma / GGUF quantizer terms — see [`NOTICE`](NOTICE) |

---

## Method

### The steering update

At step $t$ the residual is updated by

$$h_t' = h_t + \Delta t \cdot \mathrm{clip}\big(F_t, \pm c\big), \quad F_t = \lambda_g\, g_t + \lambda_s\, s_t + \lambda_a\, a_t + \mu\, v_{t-1} + \eta_t$$

with viscosity scale $\lambda_g$, a per-token force cap $c$, momentum $\mu$, and isotropic Gaussian noise $\eta_t$. The manifold pullback ([`niodoo::pullback`](src/niodoo.rs)) tracks cumulative off-manifold drift and removes it.

Default knobs (TOML, see *Configuration*): $\Delta t = 0.035$, $\lambda_g = 0.35$, $c = 7.5$, pullback $= 0.15$, sampling temperature $T = 0.9$.

### What we measure per step

Every generation writes per-token JSONL telemetry via [`src/viz.rs`](src/viz.rs):

```
delta_mean, delta_max, delta_min     — magnitude of the steering update
splat_force_norm                     — ||s_t||
goal_attractor_norm                  — ||a_t||
field_gradient_norm                  — ||g_t||
splats_active, splats_deposited_step
```

The two screenshots below are direct captures of these telemetry streams during live generation; they are documentation of behaviour, not claims of correctness.

![Persistent memory loading and influencing generation from the first token](docs/img/persistent_memory_proof.png)
*Persistent splats reloaded from disk are non-trivially present in the very first generation step: the per-token steering delta is nonzero from token 0, rather than building up over the run.*

![Live generation showing splat forces building and steering tokens](docs/img/live_steering_output.png)
*Within a single run with no prior memory, the splat-force norm grows as pleasure splats accumulate. By construction the model's hidden state is being pulled by its own deposits; the figure shows that pull is large enough to be measurable, not that it improves quality.*

![bert produces poetic metaphors, unsloth produces analytical explanations](docs/img/model_ab_comparison.png)
*Two quantizations of the same Llama 3.1 8B weights — identical architecture, ≈ 32 bytes difference in the GGUF — were run under identical physics knobs and same prompt. The generated text diverges into noticeably different stylistic registers. This is consistent with the hypothesis that the steering amplifies the latent representational structure of each base model rather than overwriting it, but it has not been tested rigorously enough to be called more than that.*

### Logged numerical anchors

From the dated research notes, each tied to a JSONL telemetry file:

- A/B sweep, `force_cap = 80.0`, `T = 0.9`, $\sigma = 150$: maximum applied steering delta peaked at **79.67** without exceeding the cap; runs were stable. ([`research_logs/2026-03-01_bert-ab-sweep-and-tokenizer.md`](research_logs/2026-03-01_bert-ab-sweep-and-tokenizer.md))
- Hidden-state steering at the same config, `delta_mean = 19.81`, `delta_max = 37.30`, `goal_attractor_norm = 195.82`. ([`research_logs/2026-03-03_hidden-state-steering.md`](research_logs/2026-03-03_hidden-state-steering.md))
- Reloading persistent splats produces a **nonzero step-0 steering delta** on the next generation (≈ 21.7 in a representative `logs/latest.jsonl` run vs. exactly 0.0 in matched fresh-start runs from the bert sweep). This is the basic check that on-disk memory is actually crossing the session boundary.
- Sigma sweep ([`docs/experiments.md`](docs/experiments.md)):
  - $\sigma < 5$: no measurable effect; in 4096-dim space all distances dominate the kernel.
  - $\sigma = 50$: measurable forces, but argmax sampling still picks the same tokens.
  - $\sigma = 150$, $T = 0.9$: text actually changes; without a cap, positive-feedback runaway.
  - $\sigma = 150$, $T = 0.9$, cap $= 80$, min-splat-distance $= 100$: stable for the logged sweep window.

**Provenance of the evidence.** Across 2026-03-01..04, this repo committed **611 per-step JSONL telemetry files** to the `logs/` directory — 138 unsloth + 8 bert generation traces in the named-config format, plus ~11 in the newer `session_*.jsonl` format, covering six `force_cap` values (0, 3, 5, 8, 35, 80) and a sigma/temperature/min-distance grid. On 2026-03-03 those files were removed from the tracked tree as a repo-hygiene chore (commit [`a8f26d9`](../../commit/a8f26d9) — "chore: remove logs directory from git tracking"). They remain available in git history. To recover the full set:

```bash
git checkout a8f26d9^ -- logs/
```

A reviewer who wants to audit the numbers above should do exactly that.

The figures are engineering plots that drove knob selection: which sigma actually moves tokens, which force cap prevents runaway, what min-splat-distance kills the stacking loop. The full per-step traces are in the JSONL archive recovered by the command above.

---

## Related work and where this sits

This repository borrows ideas from three distinct lines of work and does not invent them. The novelty, if any, is the specific combination plus the persistence mechanism.

- **3D Gaussian Splatting** (Kerbl et al., 2023) — the splat representation $(\mu, \Sigma, \alpha)$ is taken from the rendering literature and re-used as a memory-of-trajectories rather than a representation of geometry.
- **Activation steering / representation engineering** (Turner et al., *Activation Addition*; Zou et al., *Representation Engineering*; Subramani et al.) — adding a vector to the residual stream to bias generation is the same primitive used here; the differences are (a) the steering vector is computed every token from a parametric field rather than precomputed, and (b) the field has a memory.
- **Control vectors and steering for LLMs** (Anthropic's "steering Claude" demonstrations; `llama.cpp`'s control-vector tooling) — same family. This work is a Rust implementation that combines on-line, per-token computation with persistent state.
- **Energy-based / score-matching views of generation** — the language of "field gradient", "manifold pullback", and "Langevin noise" comes from this tradition; the use here is engineering-flavoured, not theoretically derived.

What this repository is **not**: a new architecture, a fine-tune, a RAG system, or a claim about consciousness or emergence. The word "physics" in older notes is shorthand for the update rule above, nothing more.

---

## Repository layout

```
src/
  main.rs       CLI, generation loop, telemetry wiring
  llama.rs      Vendored quantized Llama 3.1 with forward_with_hidden() / project_to_logits()
  gemma.rs      Same shape for Gemma 3; not wired into main today
  field.rs      Continuous field over the token-embedding matrix + Top-K gradient probe
  splat.rs      Splat type: (μ, Σ, α) + safetensors persistence
  memory.rs     Splat store: deposit, decay, consolidation, force query
  niodoo.rs     The steering update: force composition + manifold pullback
  dream.rs      Offline trajectory replay with Langevin noise
  ridge.rs      Vietoris–Rips H1 collapse detector (early-warning signal)
  gpu.rs        Candle/CUDA tensor ops, batched field gradient
  tui.rs        Multi-turn chat front end
  viz.rs        Per-step JSONL telemetry
  config.rs     TOML config loader
  logger.rs     Session logger; writes via O_CREAT|O_EXCL (no symlink follow)
docs/
  foundation.md   The core token loop, as code
  experiments.md  Tuning sweeps and observed behaviour
  roadmap.md      Phase 2/3 plan (aspirational; do not read as status)
research_logs/    One dated note per substantive change, linked to a telemetry file
kernels/          Compute-shader sketches for the wgpu/Metal backend (not yet integrated)
```

A wider map is in [`CODE_MAP.md`](CODE_MAP.md).

---

## Running

```bash
# Default run
cargo run --release --bin hydrodynamic-swarm

# Custom prompt
cargo run --release --bin hydrodynamic-swarm -- --prompt "Describe a wave function"

# Token budget
cargo run --release --bin hydrodynamic-swarm -- --tokens 200

# Fresh start, no persisted splats
cargo run --release --bin hydrodynamic-swarm -- --clear-memory

# Interactive multi-turn chat
cargo run --release --bin hydrodynamic-swarm -- --chat

# Explicit model / tokenizer paths
cargo run --release --bin hydrodynamic-swarm -- \
    --model /path/to/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
    --tokenizer /path/to/tokenizer.json
```

A wrapper for the chat case lives at [`scripts/chat.sh`](scripts/chat.sh).

### Requirements

- Rust 1.75+ (stable).
- An NVIDIA GPU with the CUDA 13 toolkit. Developed on Blackwell GB10 (`sm_121a`).
- Llama 3.1 8B Instruct in GGUF Q5_K_M, plus the matching tokenizer:
  - `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf` (~5.7 GB)
  - `tokenizer.json` from the same upstream
- Place them in `data/` or pass `--model` / `--tokenizer`.

### Configuration

Everything lives in a single optional `config.toml`. Defaults are in [`src/config.rs`](src/config.rs); the file below shows the canonical structure:

```toml
[physics]
dt                = 0.035
viscosity_scale   = 0.35
force_cap         = 7.5
splat_sigma       = 35.0
splat_alpha       = 2.0
manifold_pullback = 0.15
steer_hidden      = true
gradient_topk     = 2048

[generation]
max_tokens         = 500
temperature        = 0.9
rep_penalty        = 1.18
min_success_tokens = 15
pleasure_alpha     = 1.8
pain_alpha         = -0.9

[memory]
max_splats         = 500
consolidation_dist = 80.0
decay_rate         = 0.98

[micro_dream]
entropy_threshold  = 3.0
blend_normal       = 0.10
blend_high_entropy = 0.15
topocot_threshold  = 6.0
```

### Tests

```bash
cargo test           # 42 unit tests across splat/memory/field/config/gpu
cargo clippy
```

Note: on aarch64 hosts without `fullfp16` (e.g. some Cortex-X925 cores), the `gemm-f16` transitive build can fail. This is upstream in `gemm-common` and unrelated to the project; CUDA paths still work because the GPU kernels do their own f16.

---

## Reproducibility

What's checked in to make a run repeatable:

- **The exact base model is identified by SHA in the GGUF header**; both bert and unsloth quantizations were used and their byte difference is what produces the A/B figure above.
- **Per-step JSONL telemetry** for every run, written by `viz::VizCollector`. The files in [`logs/`](logs/) are the actual generation logs that the research notes cite.
- **One research-log entry per substantive change**, dated and linked to the telemetry file it describes ([`research_logs/`](research_logs/)). The template is in [`research_logs/TEMPLATE.md`](research_logs/TEMPLATE.md).
- **A symlink-resistant logger** ([`src/logger.rs`](src/logger.rs)) so re-running into an existing log path never silently follows a redirected file.

A scripted "command → table" eval that turns the JSONL archive into a single reproducible summary is the next piece of reproducibility work.

---

## Known constraints

These are the things a contributor or reviewer should know up front, stated as facts about the implementation rather than as warnings:

- **Stable generation window in the logged sweeps is the 25–40 generated-token range** at the default `σ = 150`, `force_cap = 80`, `T = 0.9` config. Longer runs at this config drift; tighter caps (5–35) extend the window at the cost of weaker steering.
- **Top-K gradient is an approximation.** `K = 2048` against a 128256-row embedding matrix is a deliberate compute/accuracy trade; it is the per-step bottleneck, and tightening it is one of the wins on the table.
- **The splat-memory `safetensors` format is versioned and will change**; cross-version loads are not supported.
- **Two backends are scaffolded, one is live.** `Model::Llama` is wired into `main`; `Model::Gemma` exposes the same trait but is not yet driven by the generation loop.
- **The wgpu/Metal kernels in `kernels/` are sketches**, not yet routed through `PhysicsBackend`. CUDA is the production path.
- **No safety, alignment, or content-policy layer.** This is a research harness over an open-weight model and inherits the base model's failure modes.

---

## License

Three different things — don't mix them up:

| What | License | Where |
|---|---|---|
| **This repo's Rust code** | MIT-0 | [`LICENSE`](LICENSE) |
| **Loader libraries** (Candle, tokenizers) | Apache-2.0 OR MIT | [`NOTICE`](NOTICE) — this is *code*, not model weights |
| **Llama 3.1 weights** | Llama 3.1 Community License (Meta) | **Built with Llama** — not Apache |
| **Gemma 3 weights** | Gemma Terms of Use (Google) | not Apache (Gemma 4 is Apache; we use Gemma 3) |

**Model weights we run:** Meta Llama 3.1 8B Instruct (primary history; GGUF from [bartowski](https://huggingface.co/bartowski) and [Unsloth](https://huggingface.co/unsloth)), Google Gemma 3 27B IT (current default), EmbeddingGemma 300M (Unsloth Hub package of a Google model).

**Required notices** (verbatim, in [`NOTICE`](NOTICE)):
- *Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.*
- *Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms*

*Update 2026-07-11: attributions added. We had omitted these earlier; corrected now.*
