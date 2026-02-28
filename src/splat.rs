//! Gaussian Splat — individual memory unit with asymmetric decay.
//!
//! Each splat has:
//! - μ (position in embedding space)
//! - Σ (covariance — semantic spread)
//! - α (opacity/viscosity — modulated by reward/pain)
//! - valence (+1 attract, -1 repel)

#![allow(dead_code)]

use candle_core::{Result, Tensor};

/// A single Gaussian splat representing one memory.
pub struct Splat {
    /// Position in embedding space (D,)
    pub position: Tensor,
    /// Scalar spread (simplified from full covariance for v1)
    pub sigma: f32,
    /// Opacity / viscosity — how strongly this splat influences the field
    pub alpha: f32,
    /// +1.0 = attractive (reward), -1.0 = repulsive (pain)
    pub valence: f32,
    /// Timestamp of creation (for decay)
    pub created_at: u64,
}

impl Splat {
    pub fn new(position: Tensor, sigma: f32, alpha: f32, valence: f32, created_at: u64) -> Self {
        Self {
            position,
            sigma,
            alpha,
            valence,
            created_at,
        }
    }

    /// Compute the force this splat exerts on a query position.
    /// Positive valence = pull toward, negative = push away.
    pub fn force_on(&self, query_pos: &Tensor) -> Result<Tensor> {
        // direction: μ - pos (points toward splat if valence > 0)
        let direction = (&self.position - query_pos)?;
        let dist_sq: f32 = direction.sqr()?.sum_all()?.to_scalar()?;
        let sigma_sq = self.sigma * self.sigma;
        let weight = self.alpha * self.valence * (-dist_sq / sigma_sq).exp();
        direction.affine(weight as f64, 0.0)
    }

    /// Apply asymmetric decay: pain decays slower than reward.
    pub fn decay(&mut self, current_time: u64, reward_halflife: f32, pain_halflife: f32) {
        let age = (current_time - self.created_at) as f32;
        let halflife = if self.valence > 0.0 {
            reward_halflife
        } else {
            pain_halflife
        };
        let decay_factor = (-(age / halflife).ln() * 2.0_f32.ln()).exp();
        self.alpha *= decay_factor.clamp(0.0, 1.0);
    }
}
