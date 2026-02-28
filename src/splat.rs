//! Gaussian Splat — individual memory unit.
//!
//! Each splat has:
//! - μ (position in embedding space)
//! - σ (isotropic covariance — scalar for v1)
//! - α (signed opacity: positive = pleasure, negative = pain/trauma)

use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct Splat {
    /// Position in embedding space (D,)
    pub mu: Tensor,
    /// Isotropic covariance (scalar for v1)
    pub sigma: f32,
    /// Signed opacity: + = pleasure, - = pain/trauma
    pub alpha: f32,
}

impl Splat {
    pub fn new(mu: Tensor, sigma: f32, alpha: f32) -> Self {
        Self { mu, sigma, alpha }
    }
}
