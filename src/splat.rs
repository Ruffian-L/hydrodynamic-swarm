#![allow(dead_code)]
//! Gaussian Splat -- individual memory unit.
//!
//! Each splat has:
//! - mu (position in embedding space)
//! - sigma (isotropic covariance -- scalar for v1)
//! - alpha (signed opacity: positive = pleasure, negative = pain/trauma)
//! - lambda (decay rate: 0 = anchor, higher = faster evaporation)
//! - created_at (epoch secs for time-based decay)
//! - scale (hierarchical: 0=fine, 1=medium, 2=coarse)
//! - is_anchor (true = core fact, never decays)

use candle_core::Tensor;

/// Hierarchical splat scale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SplatScale {
    /// Fine-grain: precise scars from small steering deltas
    Fine = 0,
    /// Medium: moderate steering events
    Medium = 1,
    /// Coarse: broad memories from large trajectory warps
    Coarse = 2,
}

impl SplatScale {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Fine,
            1 => Self::Medium,
            _ => Self::Coarse,
        }
    }

    /// Suggested sigma multiplier for this scale.
    pub fn sigma_multiplier(self) -> f32 {
        match self {
            Self::Fine => 1.0,
            Self::Medium => 2.0,
            Self::Coarse => 4.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Splat {
    /// Position in embedding space (D,)
    pub mu: Tensor,
    /// Isotropic covariance (scalar for v1)
    pub sigma: f32,
    /// Signed opacity: + = pleasure, - = pain/trauma
    pub alpha: f32,
    /// Decay rate: controls evaporation speed.
    /// 0.0 = anchor (never decays), default ~0.02 for normal splats.
    /// Pain splats use 70% of this rate (asymmetric decay).
    pub lambda: f32,
    /// Creation timestamp (seconds since Unix epoch).
    /// Used for time-based decay: V(t) = V0 * exp(-lambda * dt).
    pub created_at: u64,
    /// Hierarchical scale (fine/medium/coarse).
    pub scale: SplatScale,
    /// Anchor splat: core fact, lambda forced to 0.
    pub is_anchor: bool,
}

impl Splat {
    /// Create a standard (non-anchor) splat with default lambda.
    pub fn new(mu: Tensor, sigma: f32, alpha: f32) -> Self {
        Self {
            mu,
            sigma,
            alpha,
            lambda: 0.02,
            created_at: now_secs(),
            scale: SplatScale::Fine,
            is_anchor: false,
        }
    }

    /// Create a splat with explicit scale based on steering delta magnitude.
    pub fn with_scale(mu: Tensor, sigma: f32, alpha: f32, delta_norm: f32) -> Self {
        let (scale, sigma_mult) = if delta_norm > 30.0 {
            (SplatScale::Coarse, SplatScale::Coarse.sigma_multiplier())
        } else if delta_norm > 20.0 {
            (SplatScale::Medium, SplatScale::Medium.sigma_multiplier())
        } else {
            (SplatScale::Fine, SplatScale::Fine.sigma_multiplier())
        };
        Self {
            mu,
            sigma: sigma * sigma_mult,
            alpha,
            lambda: 0.02,
            created_at: now_secs(),
            scale,
            is_anchor: false,
        }
    }

    /// Create an anchor splat (lambda=0, never decays).
    pub fn anchor(mu: Tensor, sigma: f32, alpha: f32) -> Self {
        Self {
            mu,
            sigma,
            alpha,
            lambda: 0.0,
            created_at: now_secs(),
            scale: SplatScale::Coarse,
            is_anchor: true,
        }
    }
}

/// Current time in seconds since Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_with_scale() {
        let device = Device::Cpu;
        let mu = Tensor::zeros((1,), candle_core::DType::F32, &device).unwrap();
        let base_sigma = 10.0;
        let alpha = 1.0;
        let epsilon = 1e-6;

        // Test Fine Scale (<= 20.0)
        let splat_fine_10 = Splat::with_scale(mu.clone(), base_sigma, alpha, 10.0);
        assert_eq!(splat_fine_10.scale, SplatScale::Fine);
        assert!((splat_fine_10.sigma - base_sigma * SplatScale::Fine.sigma_multiplier()).abs() < epsilon);

        let splat_fine_20 = Splat::with_scale(mu.clone(), base_sigma, alpha, 20.0);
        assert_eq!(splat_fine_20.scale, SplatScale::Fine);
        assert!((splat_fine_20.sigma - base_sigma * SplatScale::Fine.sigma_multiplier()).abs() < epsilon);

        // Test Medium Scale (> 20.0 and <= 30.0)
        let splat_medium_20_1 = Splat::with_scale(mu.clone(), base_sigma, alpha, 20.1);
        assert_eq!(splat_medium_20_1.scale, SplatScale::Medium);
        assert!((splat_medium_20_1.sigma - base_sigma * SplatScale::Medium.sigma_multiplier()).abs() < epsilon);

        let splat_medium_30 = Splat::with_scale(mu.clone(), base_sigma, alpha, 30.0);
        assert_eq!(splat_medium_30.scale, SplatScale::Medium);
        assert!((splat_medium_30.sigma - base_sigma * SplatScale::Medium.sigma_multiplier()).abs() < epsilon);

        // Test Coarse Scale (> 30.0)
        let splat_coarse_30_1 = Splat::with_scale(mu.clone(), base_sigma, alpha, 30.1);
        assert_eq!(splat_coarse_30_1.scale, SplatScale::Coarse);
        assert!((splat_coarse_30_1.sigma - base_sigma * SplatScale::Coarse.sigma_multiplier()).abs() < epsilon);

        let splat_coarse_100 = Splat::with_scale(mu.clone(), base_sigma, alpha, 100.0);
        assert_eq!(splat_coarse_100.scale, SplatScale::Coarse);
        assert!((splat_coarse_100.sigma - base_sigma * SplatScale::Coarse.sigma_multiplier()).abs() < epsilon);
    }
}
