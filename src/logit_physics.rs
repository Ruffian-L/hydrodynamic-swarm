//! Logit-surface physics.
//!
//! The residual engine (`niodoo::NiodooEngine::steer`) moves the hidden state; this
//! module moves the *vocab distribution* after `lm_head`. Both surfaces run every step —
//! logit bias does not replace residual physics, it tips the sampling distribution
//! toward tokens the field/memory geometry favours.
//!
//! Engines are additive `(1, V)` biases summed in a fixed order by [`LogitChain`], applied
//! between `project_to_logits` and the repetition/n-gram guards so those guards keep the
//! last word.
//!
//! Geometry note: residual `‖h‖ ≈ 450` sits far off the embedding shell `‖μ‖ ≈ 1`
//! (see `research_logs/2026-07-11_diderot-field-geometry-divergence.md`), so every
//! residual↔vocab comparison here is **cosine**, never raw L2. That is the one change
//! from the ported reference implementations, which assumed a shared scale.

use candle_core::{Result, Tensor};

use crate::field::ContinuousField;
use crate::memory::{MemoryPickConfig, SplatMemory};
use crate::niodoo::SteerResult;

/// Everything an engine may read at the logit boundary. Borrowed — no clones per step.
///
/// Residual-physics fields remain optional so callers can run a logits-only chain.
/// Engines that need missing geometry abstain, leaving the Governor — which needs only
/// logits — active.
pub struct StepCtx<'a> {
    pub step: usize,
    /// Post-steer, pre-`lm_head` residual, shape `(1, D)`. `None` on the vanilla path.
    pub steered_hidden: Option<&'a Tensor>,
    /// Force magnitudes and unit field direction from the residual steer.
    pub steer: Option<&'a SteerResult>,
    /// Token embedding matrix `(V, D)` — the field's own point cloud.
    pub token_embeddings: &'a Tensor,
    /// Diderot field. `field.positions` *is* the embedding matrix, so field index == token id.
    pub field: Option<&'a ContinuousField>,
    pub memory: Option<&'a SplatMemory>,
    /// Ranked-picker settings shared with residual scar force.
    pub memory_pick: Option<&'a MemoryPickConfig>,
    /// Current prompt fingerprint used by ranked bridge selection.
    pub prompt_fp: u32,
}

/// One physics engine acting on the vocab distribution.
pub trait LogitEngine {
    fn name(&self) -> &'static str;

    /// Additive `(1, V)` bias, or `None` to abstain this step.
    fn bias(&mut self, logits: &Tensor, ctx: &StepCtx) -> Result<Option<Tensor>>;

    /// L∞ magnitude of the last emitted bias, for telemetry. 0 when abstaining.
    fn last_mag(&self) -> f32;

    /// Two engine-specific telemetry scalars, reported even on abstain.
    /// Default is none; [`Governor`] returns `(velocity, viscosity)`.
    fn probe(&self) -> (f32, f32) {
        (0.0, 0.0)
    }

    /// Live-tunable parameters as `(name, value, min, max)`.
    ///
    /// Exposed so the chat REPL can render sliders and adjust them mid-session —
    /// tuning these should never require a rebuild.
    fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        Vec::new()
    }

    /// Set a parameter by name. Returns false if this engine does not own it.
    /// Implementations clamp to the range advertised by [`LogitEngine::params`].
    fn set_param(&mut self, _name: &str, _value: f32) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 0. Backslash penalty — breaks the `\` loop collapse mode
// ═══════════════════════════════════════════════════════════════════════════════

/// Subtract a fixed penalty from the `\` token logit to break the `\ \ \ \` collapse loop.
///
/// The Gemma4 tokenizer maps `\` to ID 236785. When the model enters the `\` loop
/// (observed at step 93 in run 2026-08-02_09-00-23), the Governor's centrifugal brake
/// is insufficient — the `\` token has a logit advantage that the governor can't
/// overcome because the loop is self-reinforcing: each `\` keeps entropy low, which
/// keeps velocity high, which brakes the top-1, but `\` is always the top-1 or
/// close enough that the brake doesn't reach ranks 2–5.
///
/// This is a surgical fix: target the specific token causing the loop, not a global
/// temperature change. The penalty is additive (subtracted from the logit), so a
/// penalty of 2.0 makes `\` 2.0 logit units less likely — roughly equivalent to
/// multiplying its probability by e^(-2.0) ≈ 0.135.
pub struct BackslashPenalty {
    pub penalty: f32,
    /// Token ID for `\` in the Gemma4 tokenizer.
    pub backslash_id: u32,
    last_mag: f32,
}

impl BackslashPenalty {
    pub fn new(penalty: f32, backslash_id: u32) -> Self {
        Self {
            penalty,
            backslash_id,
            last_mag: 0.0,
        }
    }
}

impl LogitEngine for BackslashPenalty {
    fn name(&self) -> &'static str {
        "backslash"
    }

    fn bias(&mut self, logits: &Tensor, _ctx: &StepCtx) -> Result<Option<Tensor>> {
        self.last_mag = 0.0;
        if self.penalty <= 0.0 {
            return Ok(None);
        }
        let vocab = logits.dim(1)?;
        if self.backslash_id as usize >= vocab {
            // Token ID out of vocab range — abstain rather than panic.
            return Ok(None);
        }
        // Emit a bias vector that is zero everywhere except at the backslash token.
        let mut bias = vec![0.0f32; vocab];
        bias[self.backslash_id as usize] = -self.penalty;
        self.last_mag = self.penalty;
        Ok(Some(
            Tensor::from_vec(bias, vocab, logits.device())?
                .to_dtype(logits.dtype())?
                .unsqueeze(0)?,
        ))
    }

    fn last_mag(&self) -> f32 {
        self.last_mag
    }

    fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![("backslash.penalty", self.penalty, 0.0, 10.0)]
    }

    fn set_param(&mut self, name: &str, value: f32) -> bool {
        if name == "backslash.penalty" {
            self.penalty = value.clamp(0.0, 10.0);
            true
        } else {
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Field bias — z += α · normalize(E û_g)
// ═══════════════════════════════════════════════════════════════════════════════

/// Surface field bias: tips vocab toward tokens aligned with the field force direction.
///
/// Ported verbatim from the inline block this module replaced, so that a run with only
/// this engine active is token-identical to the pre-chain build.
pub struct FieldBias {
    pub alpha: f32,
    last_mag: f32,
}

impl FieldBias {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            last_mag: 0.0,
        }
    }
}

impl LogitEngine for FieldBias {
    fn name(&self) -> &'static str {
        "field"
    }

    fn bias(&mut self, _logits: &Tensor, ctx: &StepCtx) -> Result<Option<Tensor>> {
        self.last_mag = 0.0;
        let Some(steer) = ctx.steer else {
            return Ok(None); // no residual physics this path
        };
        if self.alpha <= 0.0 || steer.grad_mag <= 1e-8 {
            return Ok(None);
        }
        let emb = ctx.token_embeddings; // (V, D)
        let v = steer
            .field_dir
            .to_dtype(emb.dtype())?
            .to_device(emb.device())?;
        // scores: (V,) = E @ û_g
        let scores = emb.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
        // Peak-normalize so α is a comparable logit-scale knob across steps
        let peak: f32 = scores.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-8);
        self.last_mag = self.alpha;
        let bias = scores
            .affine((self.alpha / peak) as f64, 0.0)?
            .unsqueeze(0)?; // (1, V)
        Ok(Some(bias))
    }

    fn last_mag(&self) -> f32 {
        self.last_mag
    }

    fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![("field.alpha", self.alpha, 0.0, 1.0)]
    }

    fn set_param(&mut self, name: &str, value: f32) -> bool {
        if name == "field.alpha" {
            self.alpha = value.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Splat bias — per-scar, token-targeted vocab bias
// ═══════════════════════════════════════════════════════════════════════════════

/// Scar-tissue bias in vocab space.
///
/// Port of `YinYangQSMA/src/physics/steering.rs::SteeringEngine::get_logit_bias`, whose own
/// doc comment names the blocker it never solved: it needed a particle→token_id table and
/// never had one. Here the Diderot field is built straight from the model's embedding
/// matrix (`ContinuousField::from_embeddings`), so a field index *is* a token id and
/// `nearest_tokens` supplies the mapping directly.
///
/// Two deliberate departures from the reference:
/// * **cosine, not inverse L2.** Scar centres live in residual space; embedding rows do
///   not. Inverse distance between the two is meaningless here (see module header).
/// * **bounded work.** The reference scanned every particle every step. This ranks scars by
///   local activation `|α|·exp(−d²/σ²)` (O(D) each) and only runs the O(N·D) neighbour
///   search for the strongest `top_m`.
pub struct SplatBias {
    pub scale: f32,
    /// How many scars may contribute in one step.
    pub top_m: usize,
    /// Tokens biased per contributing scar.
    pub top_k: usize,
    last_mag: f32,
}

impl SplatBias {
    pub fn new(scale: f32, top_m: usize, top_k: usize) -> Self {
        Self {
            scale,
            top_m,
            top_k,
            last_mag: 0.0,
        }
    }
}

impl LogitEngine for SplatBias {
    fn name(&self) -> &'static str {
        "splat"
    }

    fn bias(&mut self, logits: &Tensor, ctx: &StepCtx) -> Result<Option<Tensor>> {
        self.last_mag = 0.0;
        if self.scale <= 0.0 || self.top_m == 0 || self.top_k == 0 {
            return Ok(None);
        }
        // Needs the scar store, the field's token map, and the current residual.
        let (Some(memory), Some(field), Some(hidden)) = (ctx.memory, ctx.field, ctx.steered_hidden)
        else {
            return Ok(None);
        };
        let splats = memory.splats_ref();
        if splats.is_empty() {
            return Ok(None);
        }

        let pos = hidden.squeeze(0)?; // (D,)
        let candidate_indices: Vec<usize> =
            if ctx.steer.map(|steer| steer.memory_ranked).unwrap_or(false) {
                let Some(pick) = ctx.memory_pick else {
                    return Ok(None);
                };
                memory.ranked_splat_indices(
                    &pos,
                    pick.k.max(1),
                    ctx.prompt_fp,
                    pick.quality_weight,
                    pick.fp_weight,
                )?
            } else {
                (0..splats.len()).collect()
            };

        // Rank scars by local activation |α|·exp(−d²/σ²) — cheap, O(D) per scar.
        let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(candidate_indices.len());
        for i in candidate_indices {
            let s = &splats[i];
            if s.alpha.abs() < 1e-6 || s.sigma <= 1e-6 {
                continue;
            }
            if s.mu.dims() != pos.dims() {
                continue;
            }
            let d_sq: f32 = (&s.mu - &pos)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            let mut act = s.alpha.abs() * (-d_sq / (s.sigma * s.sigma)).exp();
            // Related-prompt: fp-matched prefill-bridge still biases vocab even when L2 is COLD.
            if ctx.prompt_fp != 0
                && crate::memory::SplatMemory::is_prefill_bridge(s)
                && crate::memory::SplatMemory::bridge_prompt_fp(s) == ctx.prompt_fp
            {
                act = act.max(s.alpha.abs().max(1.0));
            }
            if act.is_finite() && act > 1e-8 {
                ranked.push((i, act));
            }
        }
        if ranked.is_empty() {
            return Ok(None);
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(self.top_m);

        let vocab = logits.dim(1)?;
        let mut acc = vec![0.0f32; vocab];
        let mut touched = false;

        for (idx, act) in ranked {
            let s = &splats[idx];
            // Sign follows scar valence: pleasure attracts, pain repels.
            let sign = if s.alpha >= 0.0 { 1.0 } else { -1.0 };
            let neigh = field.nearest_tokens(&s.mu, self.top_k).map_err(|e| {
                candle_core::Error::Msg(format!("splat vocab-neighbour query failed: {e}"))
            })?;
            for (tok, cos_sim) in neigh {
                let t = tok as usize;
                if t >= vocab || !cos_sim.is_finite() {
                    continue;
                }
                acc[t] += sign * self.scale * act * cos_sim;
                touched = true;
            }
        }
        if !touched {
            return Ok(None);
        }

        self.last_mag = acc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let bias = Tensor::from_vec(acc, vocab, logits.device())?
            .to_dtype(logits.dtype())?
            .unsqueeze(0)?;
        Ok(Some(bias))
    }

    fn last_mag(&self) -> f32 {
        self.last_mag
    }

    fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![
            ("splat.scale", self.scale, 0.0, 1.0),
            ("splat.top_m", self.top_m as f32, 0.0, 16.0),
            ("splat.top_k", self.top_k as f32, 0.0, 128.0),
        ]
    }

    fn set_param(&mut self, name: &str, value: f32) -> bool {
        match name {
            "splat.scale" => {
                self.scale = value.clamp(0.0, 1.0);
                true
            }
            "splat.top_m" => {
                self.top_m = value.clamp(0.0, 16.0) as usize;
                true
            }
            "splat.top_k" => {
                self.top_k = value.clamp(0.0, 128.0) as usize;
                true
            }
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Governor — centrifugal brake, viscosity, minority report
// ═══════════════════════════════════════════════════════════════════════════════

/// Rolling window of normalized entropy — the "sleepwalking" detector.
struct InertiaTracker {
    window: Vec<f32>,
    capacity: usize,
}

impl InertiaTracker {
    fn new(capacity: usize) -> Self {
        Self {
            window: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn update(&mut self, h_norm: f32) {
        if self.capacity == 0 {
            return;
        }
        self.window.push(h_norm);
        if self.window.len() > self.capacity {
            self.window.remove(0);
        }
    }

    /// Sustained low entropy → viscosity. 0 until the window is full.
    fn viscosity(&self, threshold: f32, gain: f32) -> f32 {
        if self.capacity == 0 || self.window.len() < self.capacity {
            return 0.0;
        }
        let mean_h: f32 = self.window.iter().sum::<f32>() / self.window.len() as f32;
        let momentum = 1.0 - mean_h;
        if momentum > threshold {
            (momentum - threshold) * gain
        } else {
            0.0
        }
    }
}

/// Fluid-dynamics brakes on a collapsing distribution.
///
/// Port of `YinYangQSMA/src/main.rs::sample_token` (the same triad that now lives in
/// niodoo-live's `prepare_sampling_logits`). Three stacked effects:
///
/// * **Centrifugal governor** — normalized Shannon entropy gives `velocity = 1 − H_norm`.
///   Past `velocity_threshold` the top-1 logit is braked proportionally: a targeted drag on
///   the attractor token, not a global temperature change.
/// * **Viscosity** — sustained low entropy over a rolling window subtracts from the top 3.
/// * **Minority report** — the same viscosity is *added* to ranks 4 and 5, so braking the
///   leaders promotes specific alternatives instead of flattening the tail.
///
/// This is the surface that applies entropy-dependent logit physics without
/// terminating generation or rewriting the decoded history.
/// Ceiling on how far the governor may move any single logit, in logit units.
///
/// The reference implementation was unbounded: at `brake = 15` a fully collapsed
/// distribution subtracts up to 0.75 from the top-1, and viscosity at `gain = 35`
/// can subtract far more. On a model whose top-2 logit gap is often < 1.0, that
/// does not nudge the distribution — it rewrites it. Everything the governor emits
/// is clamped to ±`max_bias` so it can influence sampling without deciding it.
pub struct Governor {
    pub enabled: bool,
    pub velocity_threshold: f32,
    pub brake: f32,
    pub viscosity_threshold: f32,
    pub viscosity_gain: f32,
    pub max_bias: f32,
    tracker: InertiaTracker,
    last_mag: f32,
    last_velocity: f32,
    last_viscosity: f32,
}

impl Governor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enabled: bool,
        velocity_threshold: f32,
        brake: f32,
        window: usize,
        viscosity_threshold: f32,
        viscosity_gain: f32,
        max_bias: f32,
    ) -> Self {
        Self {
            enabled,
            velocity_threshold,
            brake,
            viscosity_threshold,
            viscosity_gain,
            max_bias,
            tracker: InertiaTracker::new(window),
            last_mag: 0.0,
            last_velocity: 0.0,
            last_viscosity: 0.0,
        }
    }

    /// `velocity = 1 − H/ln|V|` from the last step. Telemetry only.
    pub fn last_velocity(&self) -> f32 {
        self.last_velocity
    }

    /// Viscosity applied on the last step. Telemetry only.
    pub fn last_viscosity(&self) -> f32 {
        self.last_viscosity
    }
}

impl LogitEngine for Governor {
    fn name(&self) -> &'static str {
        "governor"
    }

    fn bias(&mut self, logits: &Tensor, _ctx: &StepCtx) -> Result<Option<Tensor>> {
        self.last_mag = 0.0;
        self.last_viscosity = 0.0;

        let vocab = logits.dim(1)?;
        if !self.enabled || vocab < 6 {
            return Ok(None);
        }

        // Entropy at T=1 on device; only the scalar comes back to the host.
        let probs = candle_nn::ops::softmax(&logits.to_dtype(candle_core::DType::F32)?, 1)?;
        let log_probs = probs.clamp(1e-12f32, 1.0f32)?.log()?;
        let entropy: f32 = -(probs.mul(&log_probs)?.sum_all()?.to_scalar::<f32>()?);
        if !entropy.is_finite() {
            return Ok(None);
        }
        let h_norm = (entropy / (vocab as f32).ln()).clamp(0.0, 1.0);
        let velocity = 1.0 - h_norm;
        self.last_velocity = velocity;

        self.tracker.update(h_norm);
        let viscosity = self
            .tracker
            .viscosity(self.viscosity_threshold, self.viscosity_gain);
        self.last_viscosity = viscosity;

        let braking = velocity > self.velocity_threshold && self.brake > 0.0;
        if !braking && viscosity <= 0.0 {
            return Ok(None);
        }

        // Ranks are needed for the targeted terms — one host copy, same order of cost as
        // the `to_vec1` the sampler already performs.
        let flat: Vec<f32> = logits
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?;
        let mut order: Vec<usize> = (0..flat.len()).collect();
        // Only the top 5 matter; partial-select instead of a full sort.
        order.select_nth_unstable_by(4, |&a, &b| {
            flat[b]
                .partial_cmp(&flat[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top: Vec<usize> = order[..5].to_vec();
        top.sort_by(|&a, &b| {
            flat[b]
                .partial_cmp(&flat[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut acc = vec![0.0f32; vocab];

        // Centrifugal governor: targeted drag on the attractor token.
        if braking {
            acc[top[0]] -= (velocity - self.velocity_threshold) * self.brake;
        }

        // Viscosity on the leaders, minority report on ranks 4 and 5.
        if viscosity > 0.0 {
            for &t in top.iter().take(3) {
                acc[t] -= viscosity;
            }
            acc[top[3]] += viscosity * 0.25;
            acc[top[4]] += viscosity * 0.175;
        }

        // Ceiling: influence sampling, never decide it. See `max_bias` docs.
        if self.max_bias > 0.0 {
            for v in acc.iter_mut() {
                *v = v.clamp(-self.max_bias, self.max_bias);
            }
        }

        self.last_mag = acc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if self.last_mag <= 0.0 {
            return Ok(None);
        }
        let bias = Tensor::from_vec(acc, vocab, logits.device())?
            .to_dtype(logits.dtype())?
            .unsqueeze(0)?;
        Ok(Some(bias))
    }

    fn last_mag(&self) -> f32 {
        self.last_mag
    }

    fn probe(&self) -> (f32, f32) {
        (self.last_velocity, self.last_viscosity)
    }

    fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![
            ("gov.on", if self.enabled { 1.0 } else { 0.0 }, 0.0, 1.0),
            ("gov.velocity", self.velocity_threshold, 0.5, 1.0),
            ("gov.brake", self.brake, 0.0, 15.0),
            ("gov.visc_thresh", self.viscosity_threshold, 0.5, 1.0),
            ("gov.visc_gain", self.viscosity_gain, 0.0, 35.0),
            ("gov.max_bias", self.max_bias, 0.0, 10.0),
        ]
    }

    fn set_param(&mut self, name: &str, value: f32) -> bool {
        match name {
            "gov.on" => {
                self.enabled = value >= 0.5;
                true
            }
            "gov.velocity" => {
                self.velocity_threshold = value.clamp(0.5, 1.0);
                true
            }
            "gov.brake" => {
                self.brake = value.clamp(0.0, 15.0);
                true
            }
            "gov.visc_thresh" => {
                self.viscosity_threshold = value.clamp(0.5, 1.0);
                true
            }
            "gov.visc_gain" => {
                self.viscosity_gain = value.clamp(0.0, 35.0);
                true
            }
            "gov.max_bias" => {
                self.max_bias = value.clamp(0.0, 10.0);
                true
            }
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chain
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-step report, one entry per engine that did not abstain.
#[derive(Debug, Default, Clone)]
pub struct ChainReport {
    pub field_mag: f32,
    pub splat_mag: f32,
    pub governor_mag: f32,
    pub velocity: f32,
    pub viscosity: f32,
    pub engines_fired: usize,
}

/// Ordered stack of logit engines.
pub struct LogitChain {
    engines: Vec<Box<dyn LogitEngine>>,
    last: ChainReport,
}

impl LogitChain {
    pub fn new(engines: Vec<Box<dyn LogitEngine>>) -> Self {
        Self {
            engines,
            last: ChainReport::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    pub fn last_report(&self) -> &ChainReport {
        &self.last
    }

    /// Every live-tunable parameter across the chain, as `(name, value, min, max)`.
    pub fn params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        self.engines.iter().flat_map(|e| e.params()).collect()
    }

    /// Set a parameter by name on whichever engine owns it.
    /// Returns false if no engine claims the name.
    pub fn set_param(&mut self, name: &str, value: f32) -> bool {
        for e in self.engines.iter_mut() {
            if e.set_param(name, value) {
                return true;
            }
        }
        false
    }

    /// Render the current parameters as inline ASCII sliders.
    ///
    /// Deliberately plain stdout rather than a full-screen TUI: the generation loop
    /// streams tokens to the same terminal, and an alt-screen UI would fight with it.
    pub fn render_sliders(&self) -> String {
        let mut out = String::from("  logit physics — /set <param> <value> to adjust live\n");
        for (name, value, min, max) in self.params() {
            let span = (max - min).max(1e-6);
            let frac = ((value - min) / span).clamp(0.0, 1.0);
            let width = 24usize;
            let filled = (frac * width as f32).round() as usize;
            let bar: String = (0..width)
                .map(|i| if i < filled { '#' } else { '.' })
                .collect();
            out.push_str(&format!(
                "    {:<16} [{}] {:>7.3}   ({} … {})\n",
                name, bar, value, min, max
            ));
        }
        out
    }

    /// Apply every engine in order, returning the biased logits.
    ///
    /// Biases are added one at a time in registration order rather than summed first, so a
    /// chain holding a single engine reproduces the previous inline arithmetic exactly.
    pub fn apply(&mut self, logits: &Tensor, ctx: &StepCtx) -> Result<Tensor> {
        let mut out = logits.clone();
        let mut report = ChainReport::default();
        for engine in self.engines.iter_mut() {
            if let Some(bias) = engine.bias(&out, ctx)? {
                out = (&out + &bias)?;
                report.engines_fired += 1;
            }
            let mag = engine.last_mag();
            match engine.name() {
                "field" => report.field_mag = mag,
                "splat" => report.splat_mag = mag,
                "governor" => {
                    report.governor_mag = mag;
                    // Reported even on abstain — velocity is measured every step.
                    let (velocity, viscosity) = engine.probe();
                    report.velocity = velocity;
                    report.viscosity = viscosity;
                }
                _ => {}
            }
        }
        self.last = report;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const DIM: usize = 8;

    /// An engine that always emits a constant bias, for chain arithmetic tests.
    struct ConstBias(f32, f32);
    impl LogitEngine for ConstBias {
        fn name(&self) -> &'static str {
            "const"
        }
        fn bias(&mut self, logits: &Tensor, _ctx: &StepCtx) -> Result<Option<Tensor>> {
            let n = logits.dim(1)?;
            self.1 = self.0.abs();
            Ok(Some(
                Tensor::from_vec(vec![self.0; n], n, logits.device())?.unsqueeze(0)?,
            ))
        }
        fn last_mag(&self) -> f32 {
            self.1
        }
    }

    /// An engine that never contributes.
    struct Abstain;
    impl LogitEngine for Abstain {
        fn name(&self) -> &'static str {
            "abstain"
        }
        fn bias(&mut self, _l: &Tensor, _c: &StepCtx) -> Result<Option<Tensor>> {
            Ok(None)
        }
        fn last_mag(&self) -> f32 {
            0.0
        }
    }

    fn logits_of(dev: &Device, v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    /// Owns the borrowed pieces so tests can build a real `StepCtx` — no `unsafe`,
    /// no dummy trait objects. CPU only, no GGUF needed.
    struct Fixture {
        field: ContinuousField,
        memory: SplatMemory,
        embeddings: Tensor,
        steer: SteerResult,
        hidden: Tensor,
    }

    impl Fixture {
        fn new(dev: &Device) -> Self {
            let field = ContinuousField::load_dummy(DIM, 16, dev).unwrap();
            let memory = SplatMemory::new(dev.clone());
            let embeddings = Tensor::zeros((16, DIM), candle_core::DType::F32, dev).unwrap();
            let hidden = Tensor::zeros((1, DIM), candle_core::DType::F32, dev).unwrap();
            let steer = SteerResult::zeroed(
                hidden.clone(),
                Tensor::zeros(DIM, candle_core::DType::F32, dev).unwrap(),
            );
            Self {
                field,
                memory,
                embeddings,
                steer,
                hidden,
            }
        }

        fn ctx(&self) -> StepCtx<'_> {
            StepCtx {
                step: 0,
                steered_hidden: Some(&self.hidden),
                steer: Some(&self.steer),
                token_embeddings: &self.embeddings,
                field: Some(&self.field),
                memory: Some(&self.memory),
                memory_pick: None,
                prompt_fp: 0,
            }
        }

        /// The vanilla `--chat` shape: no residual physics available.
        fn bare_ctx(&self) -> StepCtx<'_> {
            StepCtx {
                step: 0,
                steered_hidden: None,
                steer: None,
                token_embeddings: &self.embeddings,
                field: None,
                memory: None,
                memory_pick: None,
                prompt_fp: 0,
            }
        }
    }

    #[test]
    fn empty_chain_is_identity() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let logits = logits_of(&dev, &[1.0, 2.0, 3.0]);
        let mut chain = LogitChain::new(vec![]);
        assert!(chain.is_empty());
        let out = chain.apply(&logits, &fx.ctx()).unwrap();
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn abstaining_engine_does_not_change_logits() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let logits = logits_of(&dev, &[0.5, -1.0, 4.0]);
        let mut chain = LogitChain::new(vec![Box::new(Abstain)]);
        let out = chain.apply(&logits, &fx.ctx()).unwrap();
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        assert_eq!(a, vec![0.5, -1.0, 4.0]);
        assert_eq!(chain.last_report().engines_fired, 0);
    }

    #[test]
    fn biases_accumulate_in_order() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let logits = logits_of(&dev, &[0.0, 0.0]);
        let mut chain = LogitChain::new(vec![
            Box::new(ConstBias(1.0, 0.0)),
            Box::new(ConstBias(0.25, 0.0)),
        ]);
        let out = chain.apply(&logits, &fx.ctx()).unwrap();
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        assert_eq!(a, vec![1.25, 1.25]);
        assert_eq!(chain.last_report().engines_fired, 2);
    }

    #[test]
    fn field_bias_abstains_when_field_force_is_dead() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        // Fixture has grad_mag = 0.0 → dead field → abstain regardless of alpha.
        let logits = logits_of(&dev, &[0.0; 16]);
        let mut e = FieldBias::new(0.15);
        assert!(e.bias(&logits, &fx.ctx()).unwrap().is_none());
        assert_eq!(e.last_mag(), 0.0);
    }

    #[test]
    fn splat_bias_abstains_on_empty_memory() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let logits = logits_of(&dev, &[0.0; 16]);
        let mut e = SplatBias::new(1.0, 3, 4);
        assert!(e.bias(&logits, &fx.ctx()).unwrap().is_none());
    }

    #[test]
    fn splat_bias_uses_model_embedding_token_ids_and_stays_finite() {
        let dev = Device::Cpu;
        let embeddings = Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 0.0, // token 0: aligned with scar
                0.0, 1.0, 0.0, 0.0, // token 1
                -1.0, 0.0, 0.0, 0.0, // token 2: opposite
                0.0, -1.0, 0.0, 0.0, // token 3
            ],
            (4, 4),
            &dev,
        )
        .unwrap();
        let field = ContinuousField::from_embeddings(&embeddings, &dev).unwrap();
        let center = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &dev).unwrap();
        let hidden = center.unsqueeze(0).unwrap();
        let mut memory = SplatMemory::new(dev.clone());
        memory.add_splat(crate::splat::Splat::new(center, 1.0, 1.0));
        let steer = SteerResult {
            splat_mag: 1.0,
            ..SteerResult::zeroed(
                hidden.clone(),
                Tensor::zeros(4, candle_core::DType::F32, &dev).unwrap(),
            )
        };
        let ctx = StepCtx {
            step: 0,
            steered_hidden: Some(&hidden),
            steer: Some(&steer),
            token_embeddings: &embeddings,
            field: Some(&field),
            memory: Some(&memory),
            memory_pick: None,
            prompt_fp: 0,
        };
        let logits = logits_of(&dev, &[0.0; 4]);
        let mut e = SplatBias::new(0.2, 1, 1);
        let bias = e
            .bias(&logits, &ctx)
            .unwrap()
            .expect("an active pleasure scar should emit a vocab bias");
        assert_eq!(bias.dims(), &[1, 4]);
        let values: Vec<f32> = bias.squeeze(0).unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values[0] > 0.0, "aligned model token should be attracted");
        assert!(e.last_mag() > 0.0);
    }

    #[test]
    fn inertia_tracker_waits_for_full_window() {
        let mut t = InertiaTracker::new(3);
        t.update(0.0);
        t.update(0.0);
        // Window not yet full — no viscosity even though momentum would be 1.0.
        assert_eq!(t.viscosity(0.5, 10.0), 0.0);
        t.update(0.0);
        // momentum = 1.0, threshold 0.5, gain 10 → 5.0
        assert!((t.viscosity(0.5, 10.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn inertia_tracker_quiet_when_entropy_high() {
        let mut t = InertiaTracker::new(2);
        t.update(1.0);
        t.update(1.0);
        // momentum = 0.0 → below any positive threshold
        assert_eq!(t.viscosity(0.92, 35.0), 0.0);
    }

    #[test]
    fn governor_abstains_on_flat_distribution() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        // Uniform logits → maximal entropy → velocity ≈ 0 → no brake, no viscosity.
        let logits = logits_of(&dev, &[0.0; 64]);
        let mut g = Governor::new(true, 0.95, 15.0, 6, 0.92, 35.0, 0.0);
        assert!(g.bias(&logits, &fx.ctx()).unwrap().is_none());
        assert!(g.last_velocity() < 0.05);
    }

    #[test]
    fn governor_brakes_a_collapsed_distribution() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        // One token dominates → entropy ≈ 0 → velocity ≈ 1 → brake fires on argmax.
        let mut v = vec![0.0f32; 64];
        v[7] = 60.0;
        let logits = logits_of(&dev, &v);
        let mut g = Governor::new(true, 0.95, 15.0, 6, 0.92, 35.0, 0.0);
        let out = g
            .bias(&logits, &fx.ctx())
            .unwrap()
            .expect("governor should fire on a collapsed distribution");
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        assert!(a[7] < 0.0, "argmax should be braked, got {}", a[7]);
        assert!(g.last_velocity() > 0.95);
        // Non-leader tokens are untouched by the brake alone.
        assert_eq!(a[0], 0.0);
    }

    #[test]
    fn governor_respects_the_bias_ceiling() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let mut v = vec![0.0f32; 64];
        v[7] = 60.0;
        let logits = logits_of(&dev, &v);
        // Reference gains, but ceilinged at 0.2 logits.
        let mut g = Governor::new(true, 0.95, 15.0, 6, 0.92, 35.0, 0.2);
        let out = g.bias(&logits, &fx.ctx()).unwrap().expect("should fire");
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        for x in &a {
            assert!(x.abs() <= 0.2 + 1e-6, "bias {} exceeded ceiling", x);
        }
        assert!(a[7] < 0.0);
        assert!(g.last_mag() <= 0.2 + 1e-6);
    }

    #[test]
    fn disabled_governor_abstains() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let mut v = vec![0.0f32; 64];
        v[7] = 60.0;
        let logits = logits_of(&dev, &v);
        let mut g = Governor::new(false, 0.95, 15.0, 6, 0.92, 35.0, 0.0);
        assert!(g.bias(&logits, &fx.ctx()).unwrap().is_none());
    }

    #[test]
    fn governor_still_works_without_residual_physics() {
        // The vanilla --chat path: no engine, no field, no memory.
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let mut v = vec![0.0f32; 64];
        v[7] = 60.0;
        let logits = logits_of(&dev, &v);
        let mut g = Governor::new(true, 0.95, 3.0, 6, 0.92, 6.0, 1.5);
        assert!(g.bias(&logits, &fx.bare_ctx()).unwrap().is_some());
    }

    #[test]
    fn residual_engines_abstain_without_residual_physics() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        let logits = logits_of(&dev, &[0.0; 16]);
        let mut f = FieldBias::new(0.15);
        let mut s = SplatBias::new(0.02, 3, 24);
        assert!(f.bias(&logits, &fx.bare_ctx()).unwrap().is_none());
        assert!(s.bias(&logits, &fx.bare_ctx()).unwrap().is_none());
    }

    #[test]
    fn set_param_routes_to_the_owning_engine_and_clamps() {
        let mut chain = LogitChain::new(vec![
            Box::new(FieldBias::new(0.15)),
            Box::new(Governor::new(true, 0.95, 3.0, 6, 0.92, 6.0, 1.5)),
        ]);
        assert!(chain.set_param("gov.brake", 7.5));
        assert!(chain.set_param("field.alpha", 0.4));
        assert!(!chain.set_param("nope.nope", 1.0));
        // Out-of-range values clamp rather than corrupting the engine.
        assert!(chain.set_param("gov.brake", 999.0));
        let params = chain.params();
        let brake = params.iter().find(|p| p.0 == "gov.brake").unwrap();
        assert_eq!(brake.1, 15.0, "brake should clamp to its advertised max");
        let alpha = params.iter().find(|p| p.0 == "field.alpha").unwrap();
        assert_eq!(alpha.1, 0.4);
    }

    #[test]
    fn sliders_render_every_param() {
        let chain = LogitChain::new(vec![
            Box::new(FieldBias::new(0.15)),
            Box::new(SplatBias::new(0.02, 3, 24)),
            Box::new(Governor::new(true, 0.95, 3.0, 6, 0.92, 6.0, 1.5)),
        ]);
        let s = chain.render_sliders();
        for (name, _, _, _) in chain.params() {
            assert!(s.contains(name), "slider output missing {name}");
        }
    }

    #[test]
    fn governor_minority_report_promotes_ranks_four_and_five() {
        let dev = Device::Cpu;
        let fx = Fixture::new(&dev);
        // Sustained near-zero entropy fills the window, then viscosity engages.
        let mut v = vec![0.0f32; 64];
        v[0] = 60.0;
        v[1] = 50.0;
        v[2] = 40.0;
        v[3] = 30.0;
        v[4] = 20.0;
        let logits = logits_of(&dev, &v);
        let mut g = Governor::new(true, 0.95, 15.0, 3, 0.92, 35.0, 0.0);
        let mut last = None;
        for _ in 0..4 {
            last = g.bias(&logits, &fx.ctx()).unwrap();
        }
        let out = last.expect("viscosity should engage once the window is full");
        let a: Vec<f32> = out.squeeze(0).unwrap().to_vec1().unwrap();
        assert!(g.last_viscosity() > 0.0);
        // Leaders braked, ranks 4 and 5 promoted.
        assert!(a[0] < 0.0 && a[1] < 0.0 && a[2] < 0.0);
        assert!(a[3] > 0.0, "rank 4 should be promoted, got {}", a[3]);
        assert!(a[4] > 0.0, "rank 5 should be promoted, got {}", a[4]);
        assert!(a[3] > a[4], "rank 4 gets more lift than rank 5");
    }
}
