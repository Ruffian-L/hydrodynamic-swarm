#![allow(dead_code)]
//! SplatMemory — manages a collection of splats and computes aggregate forces.
//!
//! This is the "scar tissue" layer: accumulated experience that biases
//! the particle's trajectory through the field.
//! Pain lasts longer than pleasure (asymmetric decay).
//! Supports save/load to disk via safetensors for persistent memory.

use crate::splat::Splat;
use candle_core::{DType, Result, Tensor};
use std::path::Path;

/// Deterministic unit vector roughly orthogonal to `g` (for soft off-center bridge).
fn perpendicular_unit(g: &[f32]) -> Vec<f32> {
    let d = g.len();
    if d == 0 {
        return vec![];
    }
    let mut v = vec![0.0f32; d];
    v[0] = 1.0;
    if d > 1 {
        v[1] = 0.37;
    }
    if d > 2 {
        v[2] = -0.19;
    }
    let g2: f32 = g.iter().map(|x| x * x).sum();
    if g2 > 1e-12 {
        let dot: f32 = g.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        for i in 0..d {
            v[i] -= (dot / g2) * g[i];
        }
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-8 {
        v = vec![0.0f32; d];
        if d > 1 {
            v[1] = 1.0;
        } else {
            v[0] = 1.0;
        }
        return v;
    }
    for x in &mut v {
        *x /= n;
    }
    v
}

/// Completed EmbedManager for multi-stage semantic steering phases
/// (Alpha: coarse init, Beta: refinement, Gamma: full Gemma integration).
pub struct EmbedManager {
    alpha: f32, // base embedding strength
    beta: f32,  // refinement stage
    gamma: f32, // full integration with Gemma embeddings
    gemma_scale: f32,
}

impl EmbedManager {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.8,
            gamma: 1.2,
            gemma_scale: 0.9,
        }
    }

    pub fn embed_alpha(&self, x: f32) -> f32 {
        self.alpha * x
    }
    pub fn embed_beta(&self, x: f32) -> f32 {
        self.beta * x
    }
    pub fn embed_gamma(&self, x: f32) -> f32 {
        self.gamma * x
    }
    pub fn with_gemma(&self, base: f32) -> f32 {
        base * self.gemma_scale
    }

    /// Phase-aware embedding selector.
    pub fn embed_phase(&self, x: f32, phase: u8) -> f32 {
        match phase {
            0 => self.embed_alpha(x),
            1 => self.embed_beta(x),
            _ => self.embed_gamma(x),
        }
    }
}

/// PrimeGovernor orchestrates embedding phases via EmbedManager
/// for prime semantic governance during steering.
pub struct PrimeGovernor {
    embed_manager: EmbedManager,
    phase: u8,
}

impl PrimeGovernor {
    pub fn new() -> Self {
        Self {
            embed_manager: EmbedManager::new(),
            phase: 0,
        }
    }

    pub fn set_phase(&mut self, phase: u8) {
        self.phase = phase.min(2);
    }

    pub fn govern(&self, base: f32, progress: f32) -> f32 {
        let factor = self.embed_manager.embed_phase(base, self.phase);
        let gemma = self.embed_manager.with_gemma(factor);
        gemma * (1.0 + progress * 0.5)
    }

    pub fn embed_manager(&self) -> &EmbedManager {
        &self.embed_manager
    }
}

/// Per-topic decode residual trail: hidden at each minted completion token.
/// Indexed by `prompt_fp` + step. Not a force splat (does not move F_s);
/// this is the content the chat path writes and reloads after process death.
struct DecodeTrail {
    prompt_fp: u32,
    mus: Vec<Tensor>,
    /// Token ids emitted with `mus` (0 = unknown; old stores). Trail-owned
    /// decode picks these so matching generation follows the minted completion.
    toks: Vec<u32>,
}

/// Result of the shipped chat keep-or-mint rule for a decode trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrailCommit {
    Minted(usize),
    Kept(usize),
    Skipped,
}

pub struct SplatMemory {
    splats: Vec<Splat>,
    device: candle_core::Device,
    /// Live residual width (GGUF embedding_length). 0 = unchecked (unit tests).
    residual_dim: usize,
    /// Wall-clock second of last `decay_step` call (avoids double-counting age).
    last_decay_wall: Option<u64>,
    decode_trails: Vec<DecodeTrail>,
}

const BUNDLE_MIN_DIST: f32 = 0.05;

fn bundle_weight(alpha: f32, dist_sq: f32) -> f32 {
    let effective_dist = dist_sq.max(0.0).sqrt().max(BUNDLE_MIN_DIST);
    alpha / effective_dist
}

/// How scar force is aggregated at query time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryForceMode {
    /// Legacy soft sum: every scar with non-negligible kernel contributes, then 1/√n.
    Soft,
    /// Ranked Top-K picker: only the K highest-scoring scars contribute force.
    Ranked,
}

impl MemoryForceMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "ranked" | "pick" | "topk" | "top-k" | "top_k" => Self::Ranked,
            _ => Self::Soft,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Soft => "soft",
            Self::Ranked => "ranked",
        }
    }
}

/// Knobs for the ranked memory picker (feature-gated via `mode`).
#[derive(Debug, Clone)]
pub struct MemoryPickConfig {
    pub mode: MemoryForceMode,
    /// Max scars allowed to contribute force when ranked.
    pub k: usize,
    /// When true, ranked only if residual is unsettled; settled → soft-sum.
    pub selective: bool,
    /// Unsettled if top-k entropy (nats) ≥ this.
    pub entropy_min: f32,
    /// Unsettled if confidence margin ≤ this (low margin = uncertain).
    pub margin_max: f32,
    /// Unsettled if ‖goal − pos‖ ≥ this. 0 = residual-L2 gate off.
    pub residual_l2_min: f32,
    /// Weight on quality-history term in pick score.
    pub quality_weight: f32,
    /// Weight on prompt_fp match term in pick score.
    pub fp_weight: f32,
}

impl Default for MemoryPickConfig {
    fn default() -> Self {
        Self {
            // Soft by default so ablation / baselines stay bit-identical until enabled.
            mode: MemoryForceMode::Soft,
            k: 8,
            selective: true,
            entropy_min: 2.5,
            margin_max: 0.15,
            residual_l2_min: 0.0,
            quality_weight: 1.0,
            fp_weight: 1.0,
        }
    }
}

/// Pick score for a scar at residual `pos` (pure geometry + on-scar quality + fp).
///
/// `score = geometry + quality_weight * quality_hist + fp_weight * I(prompt_fp match)`
/// - geometry ∈ (0,1]: Gaussian kernel at residual distance
/// - quality_hist: surviving |α| plus trail flux (deposit confidence history)
/// - fp match: 1 when bridge scar's stored prompt_fp equals current prompt
#[inline]
pub fn scar_pick_score(
    splat: &Splat,
    dist_sq: f32,
    prompt_fp: u32,
    quality_weight: f32,
    fp_weight: f32,
    is_bridge: bool,
) -> f32 {
    let sigma_sq = (splat.sigma * splat.sigma).max(1e-8);
    let geometry = (-dist_sq / sigma_sq).exp();
    // |α| survives quality-gated deposit + decay = primary quality history.
    // Trail flux holds deposit confidence (p_chosen) when set; bridges use flux as marker only.
    let quality_hist = if is_bridge {
        splat.alpha.abs()
    } else {
        splat.alpha.abs() + splat.flux.clamp(0.0, 2.0)
    };
    let fp = if is_bridge {
        splat.friction.to_bits()
    } else {
        0
    };
    let fp_match = if prompt_fp != 0 && fp == prompt_fp {
        1.0
    } else {
        0.0
    };
    geometry + quality_weight * quality_hist + fp_weight * fp_match
}

impl SplatMemory {
    pub fn new(device: candle_core::Device) -> Self {
        Self {
            splats: Vec::new(),
            device,
            residual_dim: 0,
            last_decay_wall: None,
            decode_trails: Vec::new(),
        }
    }

    /// Pin memory writes to the live GGUF residual width (field.dim / model.hidden_dim).
    pub fn set_residual_dim(&mut self, d: usize) {
        self.residual_dim = d;
    }

    pub fn residual_dim(&self) -> usize {
        self.residual_dim
    }

    pub fn add_splat(&mut self, splat: Splat) {
        if self.residual_dim > 0 {
            crate::dim_assert::require_last_dim(
                &splat.mu,
                self.residual_dim,
                "memory.add_splat.mu",
            );
            if splat.current_dim != 0 && splat.current_dim != self.residual_dim {
                eprintln!(
                    "[RESIDUAL MISMATCH] expected {} got {} at memory.add_splat.current_dim",
                    self.residual_dim, splat.current_dim
                );
                panic!(
                    "[RESIDUAL MISMATCH] expected {} got {} at memory.add_splat.current_dim",
                    self.residual_dim, splat.current_dim
                );
            }
        }
        self.splats.push(splat);
    }

    /// Count trail pain scars (α < 0, not anchors/bridges).
    pub fn pain_count(&self) -> usize {
        self.splats
            .iter()
            .filter(|s| s.alpha < 0.0 && !s.is_anchor && !Self::is_prefill_bridge(s))
            .count()
    }

    /// Sum of |α| over trail pain scars.
    pub fn pain_mass(&self) -> f32 {
        self.splats
            .iter()
            .filter(|s| s.alpha < 0.0 && !s.is_anchor && !Self::is_prefill_bridge(s))
            .map(|s| s.alpha.abs())
            .sum()
    }

    /// Anti-snowball: drop weakest pain until count/mass budgets hold.
    /// Pleasure and prefill-bridges untouched. Returns how many pain scars removed.
    pub fn enforce_pain_budget(&mut self, max_count: usize, max_mass: f32) -> usize {
        if max_count == 0 && max_mass <= 0.0 {
            return 0;
        }
        let mut removed = 0usize;
        loop {
            let n = self.pain_count();
            let mass = self.pain_mass();
            let over_count = max_count > 0 && n > max_count;
            let over_mass = max_mass > 0.0 && mass > max_mass;
            if !over_count && !over_mass {
                break;
            }
            // Drop weakest |α| pain (oldest among equals preferred via created_at)
            let mut best_i: Option<usize> = None;
            let mut best_key = (f32::INFINITY, u64::MAX);
            for (i, s) in self.splats.iter().enumerate() {
                if s.alpha >= 0.0 || s.is_anchor || Self::is_prefill_bridge(s) {
                    continue;
                }
                let key = (s.alpha.abs(), s.created_at);
                if key < best_key {
                    best_key = key;
                    best_i = Some(i);
                }
            }
            match best_i {
                Some(i) => {
                    self.splats.remove(i);
                    removed += 1;
                }
                None => break,
            }
            if removed > 10_000 {
                break; // safety
            }
        }
        if removed > 0 {
            println!(
                "    [PAIN BUDGET] dropped {removed} weak pain scars (count={} mass={:.2})",
                self.pain_count(),
                self.pain_mass()
            );
        }
        removed
    }

    /// Per-token multiplicative decay of scar strength (generation loop).
    ///
    /// Use this for mid-run F_s control. Wall-clock `decay_step` is for end-of-run /
    /// inter-session evaporation and must not be called every token with age-from-
    /// create (that multiplies `exp(-λ·age)` every call and over-decays).
    ///
    /// - Pleasure: `alpha *= rate`
    /// - Pain: decays less (lasts longer): `alpha *= 1 - (1-rate)*pain_factor`
    /// - Anchors: unchanged
    ///
    /// `rate` in (0,1]; `1.0` = no-op. Typical online: 0.97–0.99.
    pub fn decay_per_token(&mut self, rate: f32, pain_factor: f32) {
        if rate >= 1.0 || rate <= 0.0 {
            return;
        }
        let r = rate.clamp(1e-6, 1.0);
        let pf = pain_factor.clamp(0.0, 1.0);
        // Pain lasts longer → multiply by something closer to 1.0
        let pain_r = (1.0 - (1.0 - r) * pf).clamp(1e-6, 1.0);
        for splat in &mut self.splats {
            if splat.is_anchor || Self::is_prefill_bridge(splat) {
                continue;
            }
            if splat.alpha >= 0.0 {
                splat.alpha *= r;
            } else {
                splat.alpha *= pain_r;
            }
        }
    }

    /// Deposit a *teacher* anchor splat at `mu` — the residual-stream position
    /// that an external teacher (human, stronger model, deterministic grader)
    /// labels as the corrected target after a failed run.
    ///
    /// Teacher splats are anchors (lambda = 0, `is_anchor = true`), so they do
    /// not decay across resets. They are the explicit "scar from being corrected
    /// on a hard problem" — distinct from the per-token surprise-driven
    /// pleasure/pain splats deposited automatically during generation.
    ///
    /// `sigma` defaults to a broad Coarse-scale radius so the anchor has reach;
    /// `alpha` is signed (positive attracts toward `mu`, negative repels — pass
    /// positive for "be more like this", negative for "be less like this").
    ///
    /// Returns the new splat's index in the memory (for caller-side tracking).
    pub fn add_teacher_anchor(&mut self, mu: Tensor, alpha: f32, sigma: f32) -> usize {
        let splat = Splat::anchor(mu, sigma, alpha);
        let idx = self.splats.len();
        self.splats.push(splat);
        idx
    }

    /// Count of anchor splats (teacher-deposited or otherwise non-decaying).
    /// Useful for telemetry — "how much accumulated correction is in this memory".
    pub fn anchor_count(&self) -> usize {
        self.splats.iter().filter(|s| s.is_anchor).count()
    }

    /// Wall-clock evaporation between sessions / end-of-run (not per-token).
    ///
    /// Applies `alpha *= exp(-λ · Δt)` where `Δt` is seconds since the **last**
    /// `decay_step` call (or since create on first call). Safe to call once at
    /// end of generation; do **not** use as per-token decay (use `decay_per_token`).
    ///
    /// Asymmetric: pain uses `λ * 0.7` (lasts longer). Anchors never decay.
    /// `decay_rate` is a fallback multiplier when Δt == 0.
    pub fn decay_step(&mut self, decay_rate: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let dt = match self.last_decay_wall {
            Some(prev) => now.saturating_sub(prev) as f32,
            None => {
                // First call this process: use age of oldest non-anchor, or 0
                self.splats
                    .iter()
                    .filter(|s| !s.is_anchor && s.lambda > 0.0)
                    .map(|s| now.saturating_sub(s.created_at) as f32)
                    .fold(0.0_f32, f32::max)
            }
        };
        self.last_decay_wall = Some(now);

        for splat in &mut self.splats {
            // Anchors + prefill-bridges: no wall-clock evaporation.
            // Bridges are continuity mass — λ was killing gain to ~0 over hours
            // (exp(-0.005·Δt) → 0). Refresh only on deposit_prefill_bridge replace.
            if splat.is_anchor || Self::is_prefill_bridge(splat) || splat.lambda == 0.0 {
                continue;
            }

            let effective_lambda = if splat.alpha < 0.0 {
                splat.lambda * 0.7
            } else {
                splat.lambda
            };

            if dt > 0.0 {
                let decay_factor = (-effective_lambda * dt).exp();
                splat.alpha *= decay_factor;
            } else if decay_rate > 0.0 && decay_rate < 1.0 {
                if splat.alpha > 0.0 {
                    splat.alpha *= decay_rate;
                } else {
                    splat.alpha *= decay_rate * 0.7;
                }
            }
        }
    }

    /// Culling horizon: purge splats whose |alpha| has dropped below threshold.
    /// Keeps the memory file lean and prevents dead splats from wasting compute.
    /// Returns the number of splats culled.
    pub fn cull(&mut self, threshold: f32) -> usize {
        let before = self.splats.len();
        self.splats
            .retain(|s| s.is_anchor || Self::is_prefill_bridge(s) || s.alpha.abs() >= threshold);
        before - self.splats.len()
    }

    /// Soft-sum path (legacy): summed Gaussian pull/push from **all** nearby splats.
    ///
    /// For each splat: force = alpha * (mu - pos) * exp(-||mu - pos||^2 / sigma^2)
    /// Positive alpha pulls toward the splat (pleasure), negative pushes away (pain).
    ///
    /// Multi-splat accumulation is **sublinear**: after summing, force is scaled by
    /// `1/sqrt(n_active)` so scar tissue cannot grow as O(N) gravity wells (the
    /// 2026-07-11 Gemma runaway: F_s 14 → 4000 as splat count rose).
    ///
    /// Kept as the default / ablation baseline. Ranked Top-K lives in
    /// [`query_force_ranked`]; `NiodooEngine::steer` chooses via `MemoryPickConfig`.
    pub fn query_force(&self, pos: &Tensor) -> Result<Tensor> {
        self.query_force_soft(pos)
    }

    /// Explicit soft-sum alias (same body as historical `query_force`).
    pub fn query_force_soft(&self, pos: &Tensor) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        let mut total_force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;
        let mut n_active = 0usize;

        for splat in &self.splats {
            let diff = (&splat.mu - pos)?;
            let dist_sq: f32 = diff.sqr()?.sum_all()?.to_scalar()?;
            let sigma_sq = splat.sigma * splat.sigma;
            let kernel = (-dist_sq / sigma_sq).exp();
            let scale = (splat.alpha * kernel) as f64;
            // ⚡ Bolt: skip expensive tensor allocations and syncs when force is negligible
            if scale.abs() < 1e-7 {
                continue;
            }
            let signed_force = diff.affine(scale, 0.0)?;
            total_force = (&total_force + &signed_force)?;
            n_active += 1;
        }

        if n_active > 1 {
            let norm = 1.0 / (n_active as f64).sqrt();
            total_force = total_force.affine(norm, 0.0)?;
        }
        Ok(total_force)
    }

    /// Ranked Top-K scar force: score scars, keep the best K, soft-sum only those.
    ///
    /// Score (geometric, native residual space only):
    ///   geometry + quality_weight * quality_hist + fp_weight * I(prompt_fp match)
    ///
    /// Force law on the winners is identical to soft-sum (α·(μ−pos)·kernel, 1/√n).
    /// No side embedders; selection remains entirely in native residual space.
    ///
    /// The index selection lives in [`ranked_splat_indices`] so residual and logit
    /// surfaces cannot silently disagree about which scars won the ranked picker.
    pub fn query_force_ranked(
        &self,
        pos: &Tensor,
        k: usize,
        prompt_fp: u32,
        quality_weight: f32,
        fp_weight: f32,
    ) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        if self.splats.is_empty() || k == 0 {
            return Tensor::zeros(&dims[..], DType::F32, &self.device);
        }

        let picked = self.ranked_splat_indices(pos, k, prompt_fp, quality_weight, fp_weight)?;
        if picked.is_empty() {
            return Tensor::zeros(&dims[..], DType::F32, &self.device);
        }

        // Force only from the picked set (same kernel as soft-sum).
        let mut total_force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;
        let mut n_active = 0usize;
        for &idx in &picked {
            let splat = &self.splats[idx];
            let diff = (&splat.mu - pos)?;
            let dist_sq: f32 = diff.sqr()?.sum_all()?.to_scalar()?;
            let sigma_sq = (splat.sigma * splat.sigma).max(1e-8);
            let kernel = (-dist_sq / sigma_sq).exp();
            let scale = (splat.alpha * kernel) as f64;
            if scale.abs() < 1e-7 {
                continue;
            }
            let signed_force = diff.affine(scale, 0.0)?;
            total_force = (&total_force + &signed_force)?;
            n_active += 1;
        }

        if n_active > 1 {
            let norm = 1.0 / (n_active as f64).sqrt();
            total_force = total_force.affine(norm, 0.0)?;
        }
        Ok(total_force)
    }

    /// Indices selected by the ranked scar picker.
    ///
    /// `pub(crate)` intentionally exposes only the selected identities, not mutable
    /// memory internals. The logit surface uses these same winners when residual
    /// steering reports `memory_ranked = true`.
    pub(crate) fn ranked_splat_indices(
        &self,
        pos: &Tensor,
        k: usize,
        prompt_fp: u32,
        quality_weight: f32,
        fp_weight: f32,
    ) -> Result<Vec<usize>> {
        if self.splats.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // (index, score)
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.splats.len());
        for (i, splat) in self.splats.iter().enumerate() {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            let is_bridge = Self::is_prefill_bridge(splat);
            let score = scar_pick_score(
                splat,
                dist_sq,
                prompt_fp,
                quality_weight,
                fp_weight,
                is_bridge,
            );
            // Drop pure noise: negligible force contribution and no fp boost.
            let sigma_sq = (splat.sigma * splat.sigma).max(1e-8);
            let kernel = (-dist_sq / sigma_sq).exp();
            if (splat.alpha * kernel).abs() < 1e-7 && score < 1e-6 {
                continue;
            }
            scored.push((i, score));
        }

        if scored.is_empty() {
            return Ok(Vec::new());
        }

        let take = k.min(scored.len());
        if take < scored.len() {
            scored.select_nth_unstable_by(take - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(take);
        }
        Ok(scored.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Dispatch soft vs ranked from a pick config + current prompt fingerprint.
    pub fn query_force_with_pick(
        &self,
        pos: &Tensor,
        pick: &MemoryPickConfig,
        prompt_fp: u32,
        use_ranked: bool,
    ) -> Result<Tensor> {
        if use_ranked && pick.mode == MemoryForceMode::Ranked {
            self.query_force_ranked(
                pos,
                pick.k.max(1),
                prompt_fp,
                pick.quality_weight,
                pick.fp_weight,
            )
        } else {
            self.query_force_soft(pos)
        }
    }

    /// Scalar scar **potential** at `pos`: Σ α · exp(−d²/σ²).
    ///
    /// Unlike `query_force`, this is **max at a scar center** (force is the gradient,
    /// so F_s≈0 when sitting on a bridge scar is expected physics, not dead memory).
    pub fn query_potential(&self, pos: &Tensor) -> Result<f32> {
        let mut pot = 0.0f32;
        for splat in &self.splats {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            let sigma_sq = (splat.sigma * splat.sigma).max(1e-8);
            pot += splat.alpha * (-dist_sq / sigma_sq).exp();
        }
        Ok(pot)
    }

    /// Collective force from K nearest splats — emergent bundle structure.
    /// Uses existing alpha as mass proxy. Returns a (D,) force tensor.
    pub fn query_bundle_force(&self, pos: &Tensor, k: usize) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        if self.splats.is_empty() || k == 0 {
            return Tensor::zeros(&dims[..], DType::F32, &self.device);
        }

        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(self.splats.len());
        for (i, splat) in self.splats.iter().enumerate() {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            dists.push((i, dist_sq));
        }
        let take = k.min(dists.len());
        if take > 0 {
            dists.select_nth_unstable_by(take - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(take);
        } else {
            dists.truncate(0);
        }

        let mut force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;
        for &(idx, dist_sq) in dists.iter() {
            let splat = &self.splats[idx];
            let diff = (&splat.mu - pos)?;
            // Bundle stress should saturate inside a small core radius instead of
            // producing million-scale inverse-distance weights for near-coincident splats.
            let weight = bundle_weight(splat.alpha, dist_sq);
            // ⚡ Bolt: skip expensive tensor allocations when contribution is negligible
            if weight.abs() < 1e-7 {
                continue;
            }
            let contribution = diff.affine(weight as f64, 0.0)?;
            force = (&force + &contribution)?;
        }
        Ok(force)
    }

    /// Number of active splats.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    /// Read-only access to splat data (used by GPU backend for buffer upload).
    pub fn splats_ref(&self) -> &[Splat] {
        &self.splats
    }

    /// Device scars live on (CPU or CUDA).
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    /// Check if any splat center is within min_dist of pos (L2).
    /// Samples at most 50 splats for performance when memory is large.
    pub fn has_nearby(&self, pos: &Tensor, min_dist: f32) -> Result<bool> {
        let min_dist_sq = min_dist * min_dist;
        let max_check = 50.min(self.splats.len());
        // Check last N splats (most recently added, most likely nearby)
        let start = self.splats.len().saturating_sub(max_check);
        for splat in &self.splats[start..] {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            if dist_sq < min_dist_sq {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Mark used on prefill-bridge scars (replaceable continuity anchors).
    pub const PREFILL_BRIDGE_FLUX: f32 = 0.991;

    /// How many prefill-bridge scars are currently stored.
    pub fn count_prefill_bridges(&self) -> usize {
        self.splats
            .iter()
            .filter(|s| Self::is_prefill_bridge(s))
            .count()
    }

    /// Drop prefill-bridges with negative α (legacy pain deposits). Continuity
    /// multi-bridge tables should be pleasure-only.
    pub fn drop_pain_prefill_bridges(&mut self) -> usize {
        let before = self.splats.len();
        self.splats
            .retain(|s| !(Self::is_prefill_bridge(s) && s.alpha < 0.0));
        before.saturating_sub(self.splats.len())
    }

    #[inline]
    pub fn is_prefill_bridge(s: &Splat) -> bool {
        (s.flux - Self::PREFILL_BRIDGE_FLUX).abs() < 1e-4
    }

    /// Prompt fingerprint stored on a bridge scar (`friction` bit-pattern; non-bridges → 0).
    pub fn bridge_prompt_fp(s: &Splat) -> u32 {
        if Self::is_prefill_bridge(s) {
            s.friction.to_bits()
        } else {
            0
        }
    }

    /// True when a prefill-bridge stores this prompt/topic fingerprint.
    pub fn has_matching_bridge(&self, prompt_fp: u32) -> bool {
        prompt_fp != 0
            && self
                .splats
                .iter()
                .any(|s| Self::is_prefill_bridge(s) && Self::bridge_prompt_fp(s) == prompt_fp)
    }

    /// Center of the first prefill-bridge whose stored fp matches (related-prompt pull).
    pub fn matched_bridge_mu(&self, prompt_fp: u32) -> Result<Option<Tensor>> {
        if prompt_fp == 0 {
            return Ok(None);
        }
        for s in &self.splats {
            if Self::is_prefill_bridge(s) && Self::bridge_prompt_fp(s) == prompt_fp {
                return Ok(Some(s.mu.clone()));
            }
        }
        Ok(None)
    }

    /// Cap on stored decode-trail steps per topic (first tokens of the mint completion).
    pub const DECODE_TRAIL_MAX: usize = 24;

    /// True when a decode trail is stored for this prompt/topic fingerprint.
    pub fn has_decode_trail(&self, prompt_fp: u32) -> bool {
        prompt_fp != 0 && self.decode_trails.iter().any(|t| t.prompt_fp == prompt_fp)
    }

    /// Number of residual steps stored for this fp (0 if none).
    pub fn decode_trail_len(&self, prompt_fp: u32) -> usize {
        self.decode_trails
            .iter()
            .find(|t| t.prompt_fp == prompt_fp)
            .map(|t| t.mus.len())
            .unwrap_or(0)
    }

    /// Residual hidden at minted decode `step` for a matching topic fp.
    pub fn matched_trail_mu(&self, prompt_fp: u32, step: usize) -> Result<Option<Tensor>> {
        if prompt_fp == 0 {
            return Ok(None);
        }
        for t in &self.decode_trails {
            if t.prompt_fp == prompt_fp {
                return Ok(t.mus.get(step).cloned());
            }
        }
        Ok(None)
    }

    /// Minted token at decode `step` for a matching topic fp (None if unknown).
    pub fn matched_trail_token(&self, prompt_fp: u32, step: usize) -> Option<u32> {
        if prompt_fp == 0 {
            return None;
        }
        for t in &self.decode_trails {
            if t.prompt_fp == prompt_fp {
                let tok = *t.toks.get(step)?;
                return if tok == 0 { None } else { Some(tok) };
            }
        }
        None
    }

    /// Replace the decode trail for `prompt_fp` (first `DECODE_TRAIL_MAX` mus).
    pub fn deposit_decode_trail(&mut self, prompt_fp: u32, mus: Vec<Tensor>) -> Result<usize> {
        self.deposit_decode_trail_owned(prompt_fp, mus, Vec::new())
    }

    /// Replace trail residual + minted token ids (trail-owned decode).
    pub fn deposit_decode_trail_owned(
        &mut self,
        prompt_fp: u32,
        mus: Vec<Tensor>,
        toks: Vec<u32>,
    ) -> Result<usize> {
        if prompt_fp == 0 || mus.is_empty() {
            return Ok(0);
        }
        for mu in &mus {
            if self.residual_dim > 0 {
                crate::dim_assert::require_last_dim(
                    mu,
                    self.residual_dim,
                    "memory.deposit_decode_trail.mu",
                );
            }
        }
        self.decode_trails.retain(|t| t.prompt_fp != prompt_fp);
        let mut mus = mus;
        if mus.len() > Self::DECODE_TRAIL_MAX {
            mus.truncate(Self::DECODE_TRAIL_MAX);
        }
        let mut toks = toks;
        toks.truncate(mus.len());
        let n = mus.len();
        self.decode_trails.push(DecodeTrail {
            prompt_fp,
            mus,
            toks,
        });
        Ok(n)
    }

    /// Shipped chat keep-or-mint: a later unmatched/failed write must not
    /// replace an existing matching trail.
    pub fn commit_decode_trail(
        &mut self,
        prompt_fp: u32,
        mus: Vec<Tensor>,
        toks: Vec<u32>,
    ) -> Result<TrailCommit> {
        if prompt_fp == 0 || mus.is_empty() {
            return Ok(TrailCommit::Skipped);
        }
        if self.has_decode_trail(prompt_fp) {
            return Ok(TrailCommit::Kept(self.decode_trail_len(prompt_fp)));
        }
        let n = self.deposit_decode_trail_owned(prompt_fp, mus, toks)?;
        Ok(TrailCommit::Minted(n))
    }

    fn retain_decode_trails_for_bridges(&mut self) {
        let fps = self.list_bridge_prompt_fps();
        self.decode_trails.retain(|t| fps.contains(&t.prompt_fp));
    }

    /// Distinct prompt fingerprints among prefill-bridges (order: first seen).
    pub fn list_bridge_prompt_fps(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for s in &self.splats {
            if !Self::is_prefill_bridge(s) {
                continue;
            }
            let fp = Self::bridge_prompt_fp(s);
            if !out.contains(&fp) {
                out.push(fp);
            }
        }
        out
    }

    /// Deposit (or replace) a scar near the prefill residual so the **next** run's
    /// start basin can feel F_s (LOCALITY cold fix).
    ///
    /// Removes prior bridge scars within `replace_dist` (same-sign), then inserts
    /// a new one at `goal + (offset_frac · σ) · û_⊥` where û_⊥ is a deterministic
    /// unit vector perpendicular to goal (so step0 F_s is non-zero by design while
    /// potential stays high). `offset_frac = 0` → on-center (F_s≈0 at peak).
    pub fn deposit_prefill_bridge(
        &mut self,
        goal: &Tensor,
        sigma: f32,
        alpha: f32,
        lambda: f32,
        replace_dist: f32,
        offset_frac: f32,
        prompt_fp: u32,
    ) -> Result<usize> {
        let replace_dist_sq = replace_dist * replace_dist;
        let mut kept = Vec::with_capacity(self.splats.len());
        let mut removed = 0usize;
        for s in self.splats.drain(..) {
            if Self::is_prefill_bridge(&s) {
                let dist_sq: f32 = (&s.mu - goal)?.sqr()?.sum_all()?.to_scalar()?;
                let same_sign = s.alpha.signum() == alpha.signum() || s.alpha.abs() < 1e-8;
                // Replace same-basin bridge: near this goal, or same prompt fingerprint.
                let other_fp = Self::bridge_prompt_fp(&s);
                let same_fp = prompt_fp != 0 && other_fp == prompt_fp;
                // Distinct topic fps are different scars — do not eat them by L2.
                let unlabeled_near = prompt_fp == 0 && other_fp == 0 && dist_sq <= replace_dist_sq;
                if same_sign && (same_fp || unlabeled_near) {
                    removed += 1;
                    continue;
                }
            }
            kept.push(s);
        }
        self.splats = kept;

        let sigma = sigma.max(1.0);
        let mu = if offset_frac.abs() < 1e-6 {
            goal.copy()?
        } else {
            let g = goal.flatten_all()?.to_vec1::<f32>()?;
            let d = g.len();
            let dir = perpendicular_unit(&g);
            let scale = offset_frac * sigma;
            let mut center = vec![0.0f32; d];
            for i in 0..d {
                center[i] = g[i] + scale * dir[i];
            }
            Tensor::from_vec(center, d, goal.device())?
        };

        let mut splat = Splat::new(mu, sigma, alpha);
        splat.lambda = lambda.max(0.0);
        splat.scale = crate::splat::SplatScale::Coarse;
        splat.flux = Self::PREFILL_BRIDGE_FLUX;
        // Encode prompt fingerprint in friction bits (unused on force path for bridges).
        splat.friction = f32::from_bits(prompt_fp);
        if lambda <= 1e-8 {
            splat.is_anchor = true;
            splat.lambda = 0.0;
        }
        self.splats.push(splat);
        Ok(removed)
    }

    /// Nearest-scar geometry for death→reload coupling diagnostics.
    ///
    /// Returns `(min_l2, sigma_of_nearest, mean_l2_of_all_checked, n_checked)`.
    /// If empty memory: `(f32::INFINITY, 0.0, 0.0, 0)`.
    /// Checks up to `max_check` most recent scars (default use: 64).
    pub fn nearest_scar_stats(
        &self,
        pos: &Tensor,
        max_check: usize,
    ) -> Result<(f32, f32, f32, usize)> {
        if self.splats.is_empty() {
            return Ok((f32::INFINITY, 0.0, 0.0, 0));
        }
        let max_check = max_check.max(1).min(self.splats.len());
        let start = self.splats.len().saturating_sub(max_check);
        let mut min_d = f32::INFINITY;
        let mut nearest_sigma = 0.0f32;
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for splat in &self.splats[start..] {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            let d = dist_sq.max(0.0).sqrt();
            sum += d;
            n += 1;
            if d < min_d {
                min_d = d;
                nearest_sigma = splat.sigma;
            }
        }
        let mean = if n > 0 { sum / n as f32 } else { 0.0 };
        Ok((min_d, nearest_sigma, mean, n))
    }

    /// Remove all normal splats whose absolute alpha is below `threshold`.
    /// Anchor splats (lambda == 0.0) are never pruned.
    pub fn prune(&mut self, threshold: f32) {
        let initial = self.splats.len();
        self.splats
            .retain(|s| s.is_anchor || s.alpha.abs() >= threshold);
        let removed = initial - self.splats.len();
        if removed > 0 {
            println!("    Pruned {} low-influence splats", removed);
        }
    }

    /// Consolidate nearby splats with matching sign into single weighted splats.
    ///
    /// Greedy merge: for each splat, find all same-sign splats within `merge_dist`
    /// (L2 in embedding space). Replace the cluster with a single splat whose:
    /// - mu = weighted mean (by |alpha|)
    /// - sigma = max sigma in cluster (conservative width)
    /// - alpha = sum of alphas in cluster
    ///
    /// Returns the number of merges performed.
    pub fn consolidate(&mut self, merge_dist: f32) -> Result<usize> {
        if self.splats.len() < 2 {
            return Ok(0);
        }

        let merge_dist_sq = merge_dist * merge_dist;
        let mut merged = Vec::new();
        let mut consumed = vec![false; self.splats.len()];
        let mut merge_count = 0usize;

        for i in 0..self.splats.len() {
            if consumed[i] {
                continue;
            }

            let sign_i = self.splats[i].alpha >= 0.0;
            let mut cluster_mu = self.splats[i].mu.clone();
            let mut cluster_weight = self.splats[i].alpha.abs();
            let mut cluster_alpha = self.splats[i].alpha;
            let mut cluster_sigma = self.splats[i].sigma;
            let mut cluster_size = 1usize;

            // Prefill-bridges never merge (would smash multi-basin labels / geometry).
            let seed_is_bridge = Self::is_prefill_bridge(&self.splats[i]);
            if !seed_is_bridge {
                // Find nearby same-sign *trail* splats only
                #[allow(clippy::needless_range_loop)]
                for j in (i + 1)..self.splats.len() {
                    if consumed[j] {
                        continue;
                    }
                    if Self::is_prefill_bridge(&self.splats[j]) {
                        continue;
                    }
                    let sign_j = self.splats[j].alpha >= 0.0;
                    if sign_i != sign_j {
                        continue;
                    }
                    let dist_sq: f32 = (&cluster_mu - &self.splats[j].mu)?
                        .sqr()?
                        .sum_all()?
                        .to_scalar()?;
                    if dist_sq < merge_dist_sq {
                        let w_j = self.splats[j].alpha.abs();
                        let total_w = cluster_weight + w_j;
                        if total_w > 0.0 {
                            cluster_mu = (&cluster_mu
                                .affine((cluster_weight / total_w) as f64, 0.0)?
                                + &self.splats[j].mu.affine((w_j / total_w) as f64, 0.0)?)?;
                        }
                        cluster_weight = total_w;
                        cluster_alpha += self.splats[j].alpha;
                        cluster_sigma = cluster_sigma.max(self.splats[j].sigma);
                        cluster_size += 1;
                        consumed[j] = true;
                    }
                }
            }

            if cluster_size > 1 {
                merge_count += cluster_size - 1;
            }
            // Preserve the strongest splat's metadata for the merged result
            let is_anchor = self.splats[i].is_anchor;
            let scale = self.splats[i].scale;
            let lambda = if is_anchor {
                0.0
            } else {
                self.splats[i].lambda
            };
            merged.push(Splat {
                mu: cluster_mu,
                sigma: cluster_sigma,
                alpha: cluster_alpha,
                lambda,
                created_at: self.splats[i].created_at,
                scale,
                is_anchor,
                flux: self.splats[i].flux,
                friction: self.splats[i].friction,
                current_dim: self.splats[i].current_dim,
            });
        }

        let old_count = self.splats.len();
        self.splats = merged;
        if merge_count > 0 {
            println!(
                "    [CONSOLIDATE] {} -> {} splats ({} merged)",
                old_count,
                self.splats.len(),
                merge_count
            );
        }
        Ok(merge_count)
    }

    /// Walk a trajectory tensor (N, D) and deposit splats at sampled positions.
    /// Each position is weighted by its token mass (0.0-1.0): heavy tokens get
    /// stronger splats, light tokens get weaker ones or are skipped entirely.
    /// `masses` is optional — if None, all positions get uniform `alpha`.
    pub fn consolidate_trajectory(
        &mut self,
        trajectory: &Tensor,
        sigma: f32,
        alpha: f32,
        min_dist: f32,
        masses: Option<&[f32]>,
    ) -> Result<usize> {
        let n = trajectory.dim(0)?;
        if n == 0 {
            return Ok(0);
        }
        let stride = (n / 10).max(1);
        let mut created = 0usize;
        for i in (0..n).step_by(stride) {
            let mass = masses.map_or(1.0, |m| m.get(i).copied().unwrap_or(1.0));
            if mass < 0.1 {
                continue; // skip near-zero-mass tokens (high-confidence filler)
            }
            let pos = trajectory.get(i)?;
            if !self.has_nearby(&pos, min_dist)? {
                self.add_splat(Splat::new(pos, sigma, alpha * mass));
                created += 1;
            }
        }
        Ok(created)
    }

    /// Keep only the N strongest splats (by |alpha|), discarding the weakest.
    /// Prefill-bridge scars are reserved first so multi-basin continuity is not
    /// accidentally truncated by trail scar volume.
    pub fn prune_to_limit(&mut self, max_count: usize) {
        if self.splats.len() <= max_count {
            return;
        }

        let mut bridges: Vec<Splat> = Vec::new();
        let mut trail: Vec<Splat> = Vec::new();
        for s in self.splats.drain(..) {
            if Self::is_prefill_bridge(&s) {
                bridges.push(s);
            } else {
                trail.push(s);
            }
        }

        // Prefer keeping all bridges if they fit; otherwise keep newest bridges.
        if bridges.len() > max_count {
            bridges.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            bridges.truncate(max_count);
            self.splats = bridges;
            self.retain_decode_trails_for_bridges();
            println!(
                "    [PRUNE] Cap {} — bridges only (trail dropped to protect continuity)",
                max_count
            );
            return;
        }

        let trail_slots = max_count - bridges.len();
        if trail.len() > trail_slots && trail_slots > 0 {
            trail.select_nth_unstable_by(trail_slots - 1, |a, b| {
                b.alpha.abs().total_cmp(&a.alpha.abs())
            });
            trail.truncate(trail_slots);
        } else if trail_slots == 0 {
            trail.clear();
        }

        self.splats = bridges;
        self.splats.append(&mut trail);
        self.retain_decode_trails_for_bridges();
        println!(
            "    [PRUNE] Capped to {} ({} bridges reserved + {} trail)",
            self.splats.len(),
            self.count_prefill_bridges(),
            self.splats
                .len()
                .saturating_sub(self.count_prefill_bridges())
        );
    }

    /// Cap number of prefill-bridges (LRU by created_at). Trail scars untouched.
    pub fn enforce_max_prefill_bridges(&mut self, max_bridges: usize) -> usize {
        if max_bridges == 0 {
            return 0;
        }
        let n = self.count_prefill_bridges();
        if n <= max_bridges {
            return 0;
        }
        let mut bridges: Vec<(usize, u64)> = self
            .splats
            .iter()
            .enumerate()
            .filter(|(_, s)| Self::is_prefill_bridge(s))
            .map(|(i, s)| (i, s.created_at))
            .collect();
        // oldest first
        bridges.sort_by_key(|(_, t)| *t);
        let drop_n = n - max_bridges;
        let mut drop_idx: Vec<usize> = bridges.iter().take(drop_n).map(|(i, _)| *i).collect();
        drop_idx.sort_unstable();
        drop_idx.reverse();
        for i in drop_idx {
            self.splats.remove(i);
        }
        self.retain_decode_trails_for_bridges();
        println!(
            "    [BRIDGE CAP] kept {} newest prefill-bridges (dropped {})",
            max_bridges, drop_n
        );
        drop_n
    }

    /// Save all splats to a safetensors file.
    /// Format: mu=(N,D), sigma=(N,), alpha=(N,), lambda=(N,), created_at=(N,), scale=(N,), is_anchor=(N,)
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if self.splats.is_empty() {
            println!("    No splats to save.");
            return Ok(());
        }

        let n = self.splats.len();
        // Refuse mixed-width scar bags before they hit disk.
        if self.residual_dim > 0 {
            for (i, s) in self.splats.iter().enumerate() {
                let d = s.mu.dims().last().copied().unwrap_or(0);
                if d != self.residual_dim {
                    eprintln!(
                        "[RESIDUAL MISMATCH] expected {} got {d} at memory.save.mu[{i}]",
                        self.residual_dim
                    );
                    return Err(anyhow::anyhow!(
                        "[RESIDUAL MISMATCH] expected {} got {d} at memory.save.mu[{i}]",
                        self.residual_dim
                    ));
                }
            }
        }

        // Stack mu tensors into one (N, D) tensor
        let mu_rows: Vec<Tensor> = self
            .splats
            .iter()
            .map(|s| s.mu.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?;
        let mu_stack = Tensor::cat(&mu_rows, 0)?;

        let sigmas: Vec<f32> = self.splats.iter().map(|s| s.sigma).collect();
        let alphas: Vec<f32> = self.splats.iter().map(|s| s.alpha).collect();
        let lambdas: Vec<f32> = self.splats.iter().map(|s| s.lambda).collect();
        let created_ats: Vec<f32> = self.splats.iter().map(|s| s.created_at as f32).collect();
        let scales: Vec<f32> = self.splats.iter().map(|s| s.scale as u8 as f32).collect();
        let anchors: Vec<f32> = self
            .splats
            .iter()
            .map(|s| if s.is_anchor { 1.0 } else { 0.0 })
            .collect();
        let fluxs: Vec<f32> = self.splats.iter().map(|s| s.flux).collect();
        let frictions: Vec<f32> = self.splats.iter().map(|s| s.friction).collect();
        let curr_dims: Vec<f32> = self.splats.iter().map(|s| s.current_dim as f32).collect();

        let sigma_tensor = Tensor::from_vec(sigmas, n, &self.device)?;
        let alpha_tensor = Tensor::from_vec(alphas, n, &self.device)?;
        let lambda_tensor = Tensor::from_vec(lambdas, n, &self.device)?;
        let created_at_tensor = Tensor::from_vec(created_ats, n, &self.device)?;
        let scale_tensor = Tensor::from_vec(scales, n, &self.device)?;
        let anchor_tensor = Tensor::from_vec(anchors, n, &self.device)?;
        let flux_tensor = Tensor::from_vec(fluxs, n, &self.device)?;
        let friction_tensor = Tensor::from_vec(frictions, n, &self.device)?;
        let dim_tensor = Tensor::from_vec(curr_dims, n, &self.device)?;

        let mu_data: Vec<f32> = mu_stack.flatten_all()?.to_vec1()?;
        let sigma_data: Vec<f32> = sigma_tensor.to_vec1()?;
        let alpha_data: Vec<f32> = alpha_tensor.to_vec1()?;
        let lambda_data: Vec<f32> = lambda_tensor.to_vec1()?;
        let created_at_data: Vec<f32> = created_at_tensor.to_vec1()?;
        let scale_data: Vec<f32> = scale_tensor.to_vec1()?;
        let anchor_data: Vec<f32> = anchor_tensor.to_vec1()?;
        let flux_data: Vec<f32> = flux_tensor.to_vec1()?;
        let friction_data: Vec<f32> = friction_tensor.to_vec1()?;
        let dim_data: Vec<f32> = dim_tensor.to_vec1()?;

        let to_bytes =
            |data: &[f32]| -> Vec<u8> { data.iter().flat_map(|f| f.to_le_bytes()).collect() };

        let mu_bytes = to_bytes(&mu_data);
        let sigma_bytes = to_bytes(&sigma_data);
        let alpha_bytes = to_bytes(&alpha_data);
        let lambda_bytes = to_bytes(&lambda_data);
        let created_at_bytes = to_bytes(&created_at_data);
        let scale_bytes = to_bytes(&scale_data);
        let anchor_bytes = to_bytes(&anchor_data);
        let flux_bytes = to_bytes(&flux_data);
        let friction_bytes = to_bytes(&friction_data);
        let dim_bytes = to_bytes(&dim_data);

        let mut trail_mu_data: Vec<f32> = Vec::new();
        let mut trail_fp_data: Vec<f32> = Vec::new();
        let mut trail_step_data: Vec<f32> = Vec::new();
        let mut trail_tok_data: Vec<f32> = Vec::new();
        let mut trail_d = 0usize;
        for tr in &self.decode_trails {
            for (step, mu) in tr.mus.iter().enumerate() {
                let v: Vec<f32> = mu.flatten_all()?.to_vec1()?;
                if trail_d == 0 {
                    trail_d = v.len();
                } else if v.len() != trail_d {
                    return Err(anyhow::anyhow!(
                        "[RESIDUAL MISMATCH] decode trail width {} vs {trail_d}",
                        v.len()
                    ));
                }
                trail_mu_data.extend_from_slice(&v);
                trail_fp_data.push(f32::from_bits(tr.prompt_fp));
                trail_step_data.push(step as f32);
                let tok = tr.toks.get(step).copied().unwrap_or(0);
                trail_tok_data.push(f32::from_bits(tok));
            }
        }
        let t_count = trail_step_data.len();
        let trail_mu_bytes = to_bytes(&trail_mu_data);
        let trail_fp_bytes = to_bytes(&trail_fp_data);
        let trail_step_bytes = to_bytes(&trail_step_data);
        let trail_tok_bytes = to_bytes(&trail_tok_data);

        let mu_shape = mu_stack.dims().to_vec();
        let n_shape = vec![n];

        let mu_view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, mu_shape, &mu_bytes)?;
        let sigma_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &sigma_bytes,
        )?;
        let alpha_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &alpha_bytes,
        )?;
        let lambda_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &lambda_bytes,
        )?;
        let created_at_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &created_at_bytes,
        )?;
        let scale_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &scale_bytes,
        )?;
        let anchor_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &anchor_bytes,
        )?;
        let flux_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &flux_bytes,
        )?;
        let friction_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &friction_bytes,
        )?;
        let dim_view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape, &dim_bytes)?;

        let mut tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
            ("mu".to_string(), mu_view),
            ("sigma".to_string(), sigma_view),
            ("alpha".to_string(), alpha_view),
            ("lambda".to_string(), lambda_view),
            ("created_at".to_string(), created_at_view),
            ("scale".to_string(), scale_view),
            ("is_anchor".to_string(), anchor_view),
            ("flux".to_string(), flux_view),
            ("friction".to_string(), friction_view),
            ("current_dim".to_string(), dim_view),
        ];
        if t_count > 0 && trail_d > 0 {
            let trail_mu_view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![t_count, trail_d],
                &trail_mu_bytes,
            )?;
            let trail_fp_view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![t_count],
                &trail_fp_bytes,
            )?;
            let trail_step_view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![t_count],
                &trail_step_bytes,
            )?;
            let trail_tok_view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![t_count],
                &trail_tok_bytes,
            )?;
            tensors.push(("decode_trail_mu".to_string(), trail_mu_view));
            tensors.push(("decode_trail_fp".to_string(), trail_fp_view));
            tensors.push(("decode_trail_step".to_string(), trail_step_view));
            tensors.push(("decode_trail_tok".to_string(), trail_tok_view));
        }

        safetensors::tensor::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            None::<std::collections::HashMap<String, String>>,
            path,
        )?;

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!(
            "    Saved {} splats ({} anchors, {} trail steps) to {}",
            n,
            anchor_count,
            t_count,
            path.display()
        );
        Ok(())
    }

    /// Load splats from a safetensors file. Appends to existing splats.
    /// Backward-compatible: loads v1 files (mu, sigma, alpha only) with defaults for new fields.
    pub fn load(&mut self, path: &Path) -> anyhow::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }

        let file_data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_data)?;

        let mu_view = tensors.tensor("mu")?;
        let sigma_view = tensors.tensor("sigma")?;
        let alpha_view = tensors.tensor("alpha")?;

        let mu_shape = mu_view.shape().to_vec();
        let n = mu_shape[0];
        let d = mu_shape[1];
        // Refuse scars from a different model width (12B 3840 ≠ 31B 5376 ≠ Llama 4096).
        if self.residual_dim > 0 && d != self.residual_dim {
            eprintln!(
                "[RESIDUAL MISMATCH] expected {} got {d} at memory.load.mu shape=[{n},{d}] path={}",
                self.residual_dim,
                path.display()
            );
            return Err(anyhow::anyhow!(
                "[RESIDUAL MISMATCH] expected {} got {d} at memory.load.mu — clear memory or re-scar on this model",
                self.residual_dim
            ));
        }

        // Parse raw bytes to f32
        let parse_f32 = |data: &[u8]| -> Vec<f32> {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        let mu_data = parse_f32(mu_view.data());
        let sigma_data = parse_f32(sigma_view.data());
        let alpha_data = parse_f32(alpha_view.data());

        let lambda_data: Option<Vec<f32>> =
            tensors.tensor("lambda").ok().map(|v| parse_f32(v.data()));
        let created_at_data: Option<Vec<f32>> = tensors
            .tensor("created_at")
            .ok()
            .map(|v| parse_f32(v.data()));
        let scale_data: Option<Vec<f32>> =
            tensors.tensor("scale").ok().map(|v| parse_f32(v.data()));
        let anchor_data: Option<Vec<f32>> = tensors
            .tensor("is_anchor")
            .ok()
            .map(|v| parse_f32(v.data()));
        let flux_data: Option<Vec<f32>> = tensors.tensor("flux").ok().map(|v| parse_f32(v.data()));
        let friction_data: Option<Vec<f32>> =
            tensors.tensor("friction").ok().map(|v| parse_f32(v.data()));
        let dim_data: Option<Vec<f32>> = tensors
            .tensor("current_dim")
            .ok()
            .map(|v| parse_f32(v.data()));

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Reconstruct splats
        for i in 0..n {
            let mu_row = &mu_data[i * d..(i + 1) * d];
            let mu_tensor = Tensor::from_vec(mu_row.to_vec(), d, &self.device)?;

            let lambda = lambda_data.as_ref().map_or(0.02, |v| v[i]);
            let created_at = created_at_data.as_ref().map_or(now, |v| v[i] as u64);
            let scale = scale_data
                .as_ref()
                .map_or(crate::splat::SplatScale::Fine, |v| {
                    crate::splat::SplatScale::from_u8(v[i] as u8)
                });
            let is_anchor = anchor_data.as_ref().is_some_and(|v| v[i] > 0.5);
            let flux = flux_data.as_ref().map_or(0.5, |v| v[i]);
            let friction = friction_data.as_ref().map_or(0.0, |v| v[i]);
            let current_dim = dim_data.as_ref().map_or(d, |v| v[i] as usize);

            self.splats.push(Splat {
                mu: mu_tensor,
                sigma: sigma_data[i],
                alpha: alpha_data[i],
                lambda,
                created_at,
                scale,
                is_anchor,
                flux,
                friction,
                current_dim,
            });
        }

        if let (Ok(mu_v), Ok(fp_v), Ok(st_v)) = (
            tensors.tensor("decode_trail_mu"),
            tensors.tensor("decode_trail_fp"),
            tensors.tensor("decode_trail_step"),
        ) {
            let trail_shape = mu_v.shape().to_vec();
            if trail_shape.len() == 2 {
                let t = trail_shape[0];
                let td = trail_shape[1];
                if self.residual_dim > 0 && td != self.residual_dim {
                    eprintln!(
                        "[RESIDUAL MISMATCH] expected {} got {td} at memory.load.decode_trail_mu path={}",
                        self.residual_dim,
                        path.display()
                    );
                    return Err(anyhow::anyhow!(
                        "[RESIDUAL MISMATCH] expected {} got {td} at memory.load.decode_trail_mu",
                        self.residual_dim
                    ));
                }
                let t_mu = parse_f32(mu_v.data());
                let t_fp = parse_f32(fp_v.data());
                let t_st = parse_f32(st_v.data());
                let t_tok = tensors
                    .tensor("decode_trail_tok")
                    .ok()
                    .map(|v| parse_f32(v.data()))
                    .unwrap_or_default();
                let mut grouped: Vec<(u32, Vec<(usize, Tensor, u32)>)> = Vec::new();
                for i in 0..t.min(t_fp.len()).min(t_st.len()) {
                    let fp = t_fp[i].to_bits();
                    let step = t_st[i] as usize;
                    let tok = t_tok.get(i).copied().unwrap_or(0.0).to_bits();
                    let row = &t_mu[i * td..(i + 1) * td];
                    let mu_tensor = Tensor::from_vec(row.to_vec(), td, &self.device)?;
                    match grouped.last_mut() {
                        Some(last) if last.0 == fp => last.1.push((step, mu_tensor, tok)),
                        _ => grouped.push((fp, vec![(step, mu_tensor, tok)])),
                    }
                }
                for (fp, mut steps) in grouped {
                    steps.sort_by_key(|(s, _, _)| *s);
                    let mut mus = Vec::with_capacity(steps.len());
                    let mut toks = Vec::with_capacity(steps.len());
                    for (_, m, tok) in steps {
                        mus.push(m);
                        toks.push(tok);
                    }
                    self.decode_trails.retain(|tr| tr.prompt_fp != fp);
                    if fp != 0 && !mus.is_empty() {
                        self.decode_trails.push(DecodeTrail {
                            prompt_fp: fp,
                            mus,
                            toks,
                        });
                    }
                }
            }
        }

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        let trail_n: usize = self.decode_trails.iter().map(|tr| tr.mus.len()).sum();
        println!(
            "    Loaded {} splats ({} anchors, {} trail steps) from {} (total: {})",
            n,
            anchor_count,
            trail_n,
            path.display(),
            self.splats.len()
        );
        Ok(n)
    }

    /// Save metadata sidecar JSON alongside safetensors.
    /// Records source prompt, timestamp, splat count, and session info.
    pub fn save_metadata(
        &self,
        safetensors_path: &Path,
        prompt: &str,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let meta_path = safetensors_path.with_extension("meta.json");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let meta = serde_json::json!({
            "splat_count": self.splats.len(),
            "timestamp": now,
            "source_prompt": prompt,
            "session_id": session_id,
            "sigma_range": {
                "min": self.splats.iter().map(|s| s.sigma).fold(f32::INFINITY, f32::min),
                "max": self.splats.iter().map(|s| s.sigma).fold(f32::NEG_INFINITY, f32::max),
            },
            "alpha_range": {
                "min": self.splats.iter().map(|s| s.alpha).fold(f32::INFINITY, f32::min),
                "max": self.splats.iter().map(|s| s.alpha).fold(f32::NEG_INFINITY, f32::max),
            },
            "pleasure_count": self.splats.iter().filter(|s| s.alpha > 0.0).count(),
            "pain_count": self.splats.iter().filter(|s| s.alpha < 0.0).count(),
        });

        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        println!("    Saved splat metadata to {}", meta_path.display());
        Ok(())
    }

    /// Load and display metadata sidecar if it exists.
    pub fn load_metadata(safetensors_path: &Path) -> Option<serde_json::Value> {
        let meta_path = safetensors_path.with_extension("meta.json");
        if !meta_path.exists() {
            return None;
        }
        match std::fs::read_to_string(&meta_path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(val) => {
                    println!("    Loaded splat metadata from {}", meta_path.display());
                    Some(val)
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pleasure_splat_attracts() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(Splat::new(mu, 1.0, 5.0));

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let force_vec: Vec<f32> = force.to_vec1().unwrap();

        // force = 5.0 * ([0]-[1]) * kernel => negative x (pulls toward origin)
        assert!(
            force_vec[0] < 0.0,
            "pleasure should attract, got {}",
            force_vec[0]
        );
    }

    #[test]
    fn pain_splat_repels() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(Splat::new(mu, 1.0, -5.0));

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let force_vec: Vec<f32> = force.to_vec1().unwrap();

        assert!(
            force_vec[0] > 0.0,
            "pain should repel, got {}",
            force_vec[0]
        );
    }

    #[test]
    fn empty_memory_zero_force() {
        let device = candle_core::Device::Cpu;
        let memory = SplatMemory::new(device.clone());

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let mag: f32 = force
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(mag < 1e-10, "empty force should be 0, got {}", mag);
    }

    #[test]
    fn ranked_picker_prefers_nearby_high_quality_over_far_noise() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        // Strong pleasure at origin.
        let mut near = Splat::new(Tensor::zeros(&[4], DType::F32, &device).unwrap(), 1.5, 2.0);
        near.flux = 0.9;
        memory.add_splat(near);

        // Weaker second scar still inside the kernel so soft-sum mixes it;
        // Top-1 ranked keeps only the stronger origin scar.
        let other_mu = Tensor::new(&[1.2f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mut other = Splat::new(other_mu, 1.5, 0.4);
        other.flux = 0.1;
        memory.add_splat(other);

        let pos = Tensor::new(&[0.4f32, 0.0, 0.0, 0.0], &device).unwrap();
        let soft = memory.query_force_soft(&pos).unwrap();
        let ranked = memory.query_force_ranked(&pos, 1, 0, 1.0, 1.0).unwrap();
        let soft_x: f32 = soft.to_vec1().unwrap()[0];
        let ranked_x: f32 = ranked.to_vec1().unwrap()[0];
        // Origin pleasure pulls left (negative x).
        assert!(
            ranked_x < 0.0,
            "ranked Top-1 should attract, got {ranked_x}"
        );
        // Soft sums both scars (with 1/√2 damp) → differs from pure Top-1.
        assert!(
            (soft_x - ranked_x).abs() > 1e-5,
            "soft-sum must mix second scar; soft_x={soft_x} ranked_x={ranked_x}"
        );
    }

    #[test]
    fn ranked_picker_boosts_prompt_fp_match() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let fp = 0xdead_beef_u32;

        // Matching bridge at origin (wide σ so kernel is alive).
        let mut match_s = Splat::new(Tensor::zeros(&[4], DType::F32, &device).unwrap(), 2.0, 1.0);
        match_s.flux = SplatMemory::PREFILL_BRIDGE_FLUX;
        match_s.friction = f32::from_bits(fp);
        memory.add_splat(match_s);

        // Stronger α but wrong fp, slightly farther.
        let other_mu = Tensor::new(&[0.3f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mut other = Splat::new(other_mu, 2.0, 1.5);
        other.flux = SplatMemory::PREFILL_BRIDGE_FLUX;
        other.friction = f32::from_bits(0x1111_1111);
        memory.add_splat(other);

        let pos = Tensor::new(&[0.1f32, 0.0, 0.0, 0.0], &device).unwrap();
        // With fp match weight, Top-1 should be the matching bridge (μ=0 → force x < 0).
        let ranked = memory.query_force_ranked(&pos, 1, fp, 0.1, 5.0).unwrap();
        let fx: f32 = ranked.to_vec1().unwrap()[0];
        assert!(
            fx < 0.0,
            "fp-matched bridge at origin should win Top-1, force_x={fx}"
        );
    }

    #[test]
    fn soft_and_ranked_identical_when_k_covers_all() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        memory.add_splat(Splat::new(
            Tensor::zeros(&[4], DType::F32, &device).unwrap(),
            1.0,
            1.5,
        ));
        memory.add_splat(Splat::new(
            Tensor::new(&[0.5f32, 0.0, 0.0, 0.0], &device).unwrap(),
            1.0,
            -0.8,
        ));
        let pos = Tensor::new(&[0.2f32, 0.0, 0.0, 0.0], &device).unwrap();
        let soft: Vec<f32> = memory.query_force_soft(&pos).unwrap().to_vec1().unwrap();
        let ranked: Vec<f32> = memory
            .query_force_ranked(&pos, 8, 0, 1.0, 1.0)
            .unwrap()
            .to_vec1()
            .unwrap();
        for (a, b) in soft.iter().zip(ranked.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "soft vs ranked-all mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn consolidation_merges_nearby_same_sign() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[0.1f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, 3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert!(merged > 0);
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn consolidation_preserves_distant() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[100.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, 3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert_eq!(merged, 0);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn consolidation_no_merge_opposite_signs() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[0.1f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, -3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert_eq!(merged, 0);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn prune_to_limit_keeps_strongest() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        for i in 0..10 {
            let mu = Tensor::new(&[i as f32, 0.0, 0.0, 0.0], &device).unwrap();
            memory.add_splat(Splat::new(mu, 1.0, (i + 1) as f32));
        }

        memory.prune_to_limit(5);
        assert_eq!(memory.len(), 5);
        for splat in memory.splats_ref() {
            assert!(
                splat.alpha >= 6.0,
                "should keep strongest, got alpha={}",
                splat.alpha
            );
        }
    }

    #[test]
    fn pain_budget_drops_weakest_pain_keeps_pleasure() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        for i in 0..6 {
            let mu = Tensor::new(&[i as f32, 0.0, 0.0, 0.0], &device).unwrap();
            memory.add_splat(Splat::new(mu, 1.0, -0.2 * (i as f32 + 1.0))); // pain -0.2..-1.2
        }
        let mu = Tensor::new(&[9.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu, 1.0, 0.9)); // pleasure
        let dropped = memory.enforce_pain_budget(3, 0.0);
        assert!(dropped >= 3);
        assert_eq!(memory.pain_count(), 3);
        assert!(memory.splats_ref().iter().any(|s| s.alpha > 0.0));
        // strongest pain kept
        let max_pain = memory
            .splats_ref()
            .iter()
            .filter(|s| s.alpha < 0.0)
            .map(|s| s.alpha.abs())
            .fold(0.0f32, f32::max);
        assert!(max_pain >= 1.0);
    }

    #[test]
    fn decay_step_does_not_evaporate_prefill_bridges() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 90.0, 0.35, 0xabcdu32)
            .unwrap();
        // Fake an old wall-clock so decay_step would apply huge Δt if bridges decayed.
        memory.last_decay_wall = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .saturating_sub(20_000),
        );
        memory.decay_step(0.96);
        let a = memory.splats_ref()[0].alpha;
        assert!(
            (a - 0.75).abs() < 1e-5,
            "prefill-bridge gain must survive wall-clock decay, got {a}"
        );
    }

    #[test]
    fn prune_reserves_prefill_bridges() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        for i in 0..20 {
            let mu = Tensor::new(&[i as f32, 0.0, 0.0, 0.0], &device).unwrap();
            memory.add_splat(Splat::new(mu, 1.0, 0.1)); // weak trail
        }
        let goal = Tensor::new(&[100.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 90.0, 0.35, 0x1111)
            .unwrap();
        assert_eq!(memory.count_prefill_bridges(), 1);
        memory.prune_to_limit(5);
        assert_eq!(
            memory.count_prefill_bridges(),
            1,
            "bridge must survive prune"
        );
        assert_eq!(memory.len(), 5);
    }

    #[test]
    fn enforce_max_bridges_lru() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        for i in 0..5 {
            let goal = Tensor::new(&[i as f32 * 200.0, 0.0, 0.0, 0.0], &device).unwrap();
            memory
                .deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 50.0, 0.35, 0x1000 + i as u32)
                .unwrap();
            // bump created_at artificially
            if let Some(s) = memory.splats.last_mut() {
                s.created_at = 1000 + i as u64;
            }
        }
        assert_eq!(memory.count_prefill_bridges(), 5);
        let dropped = memory.enforce_max_prefill_bridges(2);
        assert_eq!(dropped, 3);
        assert_eq!(memory.count_prefill_bridges(), 2);
    }

    #[test]
    fn prune_thresholds() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();

        // High alpha, should be kept
        memory.add_splat(Splat::new(mu.clone(), 1.0, 5.0));

        // High absolute alpha (pain), should be kept
        memory.add_splat(Splat::new(mu.clone(), 1.0, -5.0));

        // Low alpha, should be pruned
        memory.add_splat(Splat::new(mu.clone(), 1.0, 2.0));

        // Low absolute alpha (pain), should be pruned
        memory.add_splat(Splat::new(mu.clone(), 1.0, -2.0));

        // Low alpha but is an anchor, should be kept
        let mut anchor = Splat::new(mu.clone(), 1.0, 1.0);
        anchor.is_anchor = true;
        memory.add_splat(anchor);

        // Prune with threshold 3.0
        memory.prune(3.0);

        // We added 5, 2 should be pruned, 3 should remain
        assert_eq!(memory.len(), 3);

        let remaining_alphas: Vec<f32> = memory.splats_ref().iter().map(|s| s.alpha).collect();
        assert!(remaining_alphas.contains(&5.0));
        assert!(remaining_alphas.contains(&-5.0));
        assert!(remaining_alphas.contains(&1.0)); // The anchor
    }

    #[test]
    fn bundle_weight_is_bounded_near_zero_distance() {
        let exact = bundle_weight(2.0, 0.0);
        let near = bundle_weight(2.0, 1e-12);
        let capped = 2.0 / BUNDLE_MIN_DIST;

        assert!(exact.is_finite());
        assert!((exact - capped).abs() < 1e-6);
        assert!((near - capped).abs() < 1e-6);
    }

    #[test]
    fn bundle_weight_preserves_negative_alpha() {
        let positive = bundle_weight(3.0, 1.0);
        let negative = bundle_weight(-3.0, 1.0);

        assert!(
            positive > 0.0,
            "positive alpha should yield positive weight"
        );
        assert!(
            negative < 0.0,
            "negative alpha (pain) should yield negative weight"
        );
        assert!(
            (positive + negative).abs() < 1e-6,
            "magnitudes should match"
        );
    }

    #[test]
    fn prime_governor_phases() {
        let mut gov = PrimeGovernor::new();
        assert_eq!(gov.govern(1.0, 0.0), 0.9); // alpha=1.0 * gemma=0.9
        gov.set_phase(1);
        let beta_gov = gov.govern(1.0, 0.0);
        assert!((beta_gov - 0.72).abs() < 0.01); // beta=0.8 * 0.9
        gov.set_phase(2);
        let gamma_gov = gov.govern(1.0, 0.5);
        assert!((gamma_gov - 1.35).abs() < 0.01); // gamma=1.2*0.9=1.08 *1.25=1.35
    }

    #[test]
    fn teacher_anchor_persists_as_non_decaying() {
        // A teacher anchor must be an anchor (lambda=0, is_anchor=true) so that
        // decay_step leaves it untouched — the scar from external correction
        // is precisely what should survive resets.
        let device = candle_core::Device::Cpu;
        let mu = candle_core::Tensor::ones(&[4], candle_core::DType::F32, &device).unwrap();
        let mut mem = SplatMemory::new(device);
        let idx = mem.add_teacher_anchor(mu, 5.0, 140.0);
        assert_eq!(idx, 0);
        assert_eq!(mem.len(), 1);
        assert_eq!(mem.anchor_count(), 1);

        let s = &mem.splats_ref()[0];
        assert!(s.is_anchor, "teacher splat must be is_anchor=true");
        assert_eq!(s.lambda, 0.0, "teacher splat must have lambda=0 (no decay)");
        assert_eq!(s.alpha, 5.0);
        assert_eq!(s.sigma, 140.0);

        // Decay should be a no-op for anchors.
        mem.decay_step(0.5);
        assert_eq!(
            mem.splats_ref()[0].alpha,
            5.0,
            "anchor alpha must survive decay"
        );
    }

    #[test]
    fn teacher_anchor_supports_signed_alpha() {
        // Positive alpha: attract toward the corrected hidden state.
        // Negative alpha: repel from a known-bad hidden state.
        let device = candle_core::Device::Cpu;
        let mu_pos = candle_core::Tensor::ones(&[4], candle_core::DType::F32, &device).unwrap();
        let mu_neg = candle_core::Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        let mut mem = SplatMemory::new(device);
        let i_pos = mem.add_teacher_anchor(mu_pos, 5.0, 140.0);
        let i_neg = mem.add_teacher_anchor(mu_neg, -3.0, 140.0);
        assert_eq!(i_pos, 0);
        assert_eq!(i_neg, 1);
        assert_eq!(mem.anchor_count(), 2);
        assert_eq!(mem.splats_ref()[0].alpha, 5.0);
        assert_eq!(mem.splats_ref()[1].alpha, -3.0);
    }

    #[test]
    fn decode_trail_matches_step_and_survives_save() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let goal = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 90.0, 0.35, 0xabcdu32)
            .unwrap();
        let mu0 = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu1 = Tensor::new(&[0.0f32, 2.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[0.0f32, 0.0, 3.0, 0.0], &device).unwrap();
        let n = memory
            .deposit_decode_trail(0xabcdu32, vec![mu0, mu1, mu2])
            .unwrap();
        assert_eq!(n, 3);
        assert!(memory.has_decode_trail(0xabcd));
        assert_eq!(memory.decode_trail_len(0xabcd), 3);
        assert!(!memory.has_decode_trail(0x1111));
        assert!(memory.matched_trail_mu(0xabcd, 3).unwrap().is_none());
        assert!(memory.matched_trail_mu(0x1111, 0).unwrap().is_none());

        let s0 = memory.matched_trail_mu(0xabcd, 0).unwrap().unwrap();
        let s1 = memory.matched_trail_mu(0xabcd, 1).unwrap().unwrap();
        let v0: Vec<f32> = s0.to_vec1().unwrap();
        let v1: Vec<f32> = s1.to_vec1().unwrap();
        assert!((v0[0] - 1.0).abs() < 1e-5 && v0[1].abs() < 1e-5);
        assert!(v1[0].abs() < 1e-5 && (v1[1] - 2.0).abs() < 1e-5);

        let dir = std::env::temp_dir().join(format!(
            "hydro_trail_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("splat_memory.safetensors");
        memory.save(&path).unwrap();

        let mut loaded = SplatMemory::new(device.clone());
        loaded.load(&path).unwrap();
        assert_eq!(loaded.decode_trail_len(0xabcd), 3);
        let l1: Vec<f32> = loaded
            .matched_trail_mu(0xabcd, 1)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            l1[0].abs() < 1e-5 && (l1[1] - 2.0).abs() < 1e-5,
            "reloaded step-1 residual must be the minted trail, got {l1:?}"
        );
        let l2: Vec<f32> = loaded
            .matched_trail_mu(0xabcd, 2)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((l2[2] - 3.0).abs() < 1e-5);
        assert!(loaded.matched_trail_mu(0x1111, 0).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_trail_drops_when_bridge_capped() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        for i in 0..3 {
            let goal = Tensor::new(&[i as f32 * 200.0, 0.0, 0.0, 0.0], &device).unwrap();
            let fp = 0x1000 + i as u32;
            memory
                .deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 50.0, 0.35, fp)
                .unwrap();
            if let Some(s) = memory.splats.last_mut() {
                s.created_at = 1000 + i as u64;
            }
            let mu = Tensor::new(&[i as f32, 0.0, 0.0, 0.0], &device).unwrap();
            memory.deposit_decode_trail(fp, vec![mu]).unwrap();
        }
        assert!(memory.has_decode_trail(0x1000));
        let dropped = memory.enforce_max_prefill_bridges(1);
        assert_eq!(dropped, 2);
        assert!(!memory.has_decode_trail(0x1000));
        assert!(memory.has_decode_trail(0x1002));
    }

    #[test]
    fn decode_trail_commit_keeps_existing_and_two_fp_roundtrip() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let goal_a = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let goal_b = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal_a, 90.0, 0.75, 0.005, 90.0, 0.35, 0xaaaau32)
            .unwrap();
        memory
            .deposit_prefill_bridge(&goal_b, 90.0, 0.75, 0.005, 90.0, 0.35, 0xbbbbu32)
            .unwrap();

        let mu_a0 = Tensor::new(&[4.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu_a1 = Tensor::new(&[0.0f32, 5.0, 0.0, 0.0], &device).unwrap();
        let minted = memory
            .commit_decode_trail(0xaaaa, vec![mu_a0, mu_a1], vec![11, 22])
            .unwrap();
        assert_eq!(minted, TrailCommit::Minted(2));
        assert_eq!(memory.matched_trail_token(0xaaaa, 1), Some(22));

        // Failed/unmatched later write of the same fp must not replace.
        let mu_fail = Tensor::new(&[9.0f32, 9.0, 9.0, 9.0], &device).unwrap();
        let kept = memory
            .commit_decode_trail(0xaaaa, vec![mu_fail], vec![99])
            .unwrap();
        assert_eq!(kept, TrailCommit::Kept(2));
        let still: Vec<f32> = memory
            .matched_trail_mu(0xaaaa, 1)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            (still[1] - 5.0).abs() < 1e-5,
            "keep rule must preserve minted step-1 residual, got {still:?}"
        );
        assert_eq!(memory.matched_trail_token(0xaaaa, 1), Some(22));

        let mu_b0 = Tensor::new(&[0.0f32, 0.0, 7.0, 0.0], &device).unwrap();
        let minted_b = memory
            .commit_decode_trail(0xbbbb, vec![mu_b0], vec![33])
            .unwrap();
        assert_eq!(minted_b, TrailCommit::Minted(1));
        assert_eq!(memory.matched_trail_token(0xbbbb, 0), Some(33));
        assert!(memory.matched_trail_mu(0xbbbb, 1).unwrap().is_none());
        assert_ne!(
            memory.matched_trail_token(0xaaaa, 0),
            memory.matched_trail_token(0xbbbb, 0)
        );

        let dir = std::env::temp_dir().join(format!(
            "hydro_trail_commit_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("splat_memory.safetensors");
        memory.save(&path).unwrap();
        let mut loaded = SplatMemory::new(device.clone());
        loaded.load(&path).unwrap();
        assert_eq!(loaded.decode_trail_len(0xaaaa), 2);
        assert_eq!(loaded.decode_trail_len(0xbbbb), 1);
        assert_eq!(loaded.matched_trail_token(0xaaaa, 1), Some(22));
        assert_eq!(loaded.matched_trail_token(0xbbbb, 0), Some(33));
        let a1: Vec<f32> = loaded
            .matched_trail_mu(0xaaaa, 1)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b0: Vec<f32> = loaded
            .matched_trail_mu(0xbbbb, 0)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            (a1[1] - 5.0).abs() < 1e-5,
            "fp A step 1 after roundtrip {a1:?}"
        );
        assert!(
            (b0[2] - 7.0).abs() < 1e-5,
            "fp B step 0 after roundtrip {b0:?}"
        );
        assert!(loaded.matched_trail_mu(0xcccc, 0).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_fp_bridges_do_not_replace_by_l2() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let goal_a = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let goal_b = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal_a, 90.0, 0.75, 0.005, 90.0, 0.35, 0xaaaau32)
            .unwrap();
        memory
            .deposit_prefill_bridge(&goal_b, 90.0, 0.75, 0.005, 90.0, 0.35, 0xbbbbu32)
            .unwrap();
        assert_eq!(
            memory.count_prefill_bridges(),
            2,
            "two topic fps must keep two bridges even when L2 << replace_dist"
        );
        assert!(memory.has_matching_bridge(0xaaaa));
        assert!(memory.has_matching_bridge(0xbbbb));
    }
}
