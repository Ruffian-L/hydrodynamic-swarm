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

pub struct SplatMemory {
    splats: Vec<Splat>,
    device: candle_core::Device,
    /// Wall-clock second of last `decay_step` call (avoids double-counting age).
    last_decay_wall: Option<u64>,
}

const BUNDLE_MIN_DIST: f32 = 0.05;

fn bundle_weight(alpha: f32, dist_sq: f32) -> f32 {
    let effective_dist = dist_sq.max(0.0).sqrt().max(BUNDLE_MIN_DIST);
    alpha / effective_dist
}

impl SplatMemory {
    pub fn new(device: candle_core::Device) -> Self {
        Self {
            splats: Vec::new(),
            device,
            last_decay_wall: None,
        }
    }

    pub fn add_splat(&mut self, splat: Splat) {
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

    /// Core function: summed Gaussian pull/push from all nearby splats.
    ///
    /// For each splat: force = alpha * (mu - pos) * exp(-||mu - pos||^2 / sigma^2)
    /// Positive alpha pulls toward the splat (pleasure), negative pushes away (pain).
    ///
    /// Multi-splat accumulation is **sublinear**: after summing, force is scaled by
    /// `1/sqrt(n_active)` so scar tissue cannot grow as O(N) gravity wells (the
    /// 2026-07-11 Gemma runaway: F_s 14 → 4000 as splat count rose).
    pub fn query_force(&self, pos: &Tensor) -> Result<Tensor> {
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
                let same_fp = prompt_fp != 0 && Self::bridge_prompt_fp(&s) == prompt_fp;
                if same_sign && (dist_sq <= replace_dist_sq || same_fp) {
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

        let tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
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

        safetensors::tensor::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            None::<std::collections::HashMap<String, String>>,
            path,
        )?;

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!(
            "    Saved {} splats ({} anchors) to {}",
            n,
            anchor_count,
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

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!(
            "    Loaded {} splats ({} anchors) from {} (total: {})",
            n,
            anchor_count,
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
}
