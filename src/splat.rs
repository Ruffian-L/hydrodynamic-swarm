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
    /// Flux: positive energy from resonance (LivingCell port)
    pub flux: f32,
    /// Friction: dimensional erosion factor from MRL
    pub friction: f32,
    /// Current active dimension after MRL truncation
    pub current_dim: usize,
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
            flux: 0.5,
            friction: 0.0,
            current_dim: 4096,
        }
    }

    /// Create a splat with hierarchical width from steering delta.
    ///
    /// `ref_delta` should be `config.physics.splat_delta_threshold` (or similar
    /// typical high-δ scale for this model). Absolute 20/30 thresholds from the
    /// 27B-era residual walk always classified 4B tokens (δ~80–120) as Coarse×4,
    /// so scars were 4× too wide after a size change that only scaled force caps.
    pub fn with_scale(mu: Tensor, sigma: f32, alpha: f32, delta_norm: f32) -> Self {
        // Legacy absolute bands — only valid when δ lives near 10–40.
        Self::with_scale_ref(mu, sigma, alpha, delta_norm, 30.0)
    }

    /// Hierarchical splat using bands relative to `ref_delta` (deposit threshold).
    ///
    /// - Fine:   δ < 0.85 × ref  → σ × 1
    /// - Medium: δ < 1.25 × ref  → σ × 2
    /// - Coarse: else            → σ × 4
    pub fn with_scale_ref(
        mu: Tensor,
        sigma: f32,
        alpha: f32,
        delta_norm: f32,
        ref_delta: f32,
    ) -> Self {
        Self::with_scale_ref_lambda(mu, sigma, alpha, delta_norm, ref_delta, 0.02)
    }

    /// Like `with_scale_ref` but sets evaporation `lambda` from config.
    pub fn with_scale_ref_lambda(
        mu: Tensor,
        sigma: f32,
        alpha: f32,
        delta_norm: f32,
        ref_delta: f32,
        lambda: f32,
    ) -> Self {
        let ref_d = ref_delta.max(1e-3);
        let fine_max = ref_d * 0.85;
        let med_max = ref_d * 1.25;
        let (scale, sigma_mult) = if delta_norm > med_max {
            (SplatScale::Coarse, SplatScale::Coarse.sigma_multiplier())
        } else if delta_norm > fine_max {
            (SplatScale::Medium, SplatScale::Medium.sigma_multiplier())
        } else {
            (SplatScale::Fine, SplatScale::Fine.sigma_multiplier())
        };
        Self {
            mu,
            sigma: sigma * sigma_mult,
            alpha,
            lambda: lambda.max(0.0),
            created_at: now_secs(),
            scale,
            is_anchor: false,
            flux: 0.5,
            friction: 0.0,
            current_dim: 4096,
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
            flux: 0.5,
            friction: 0.0,
            current_dim: 4096,
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
    use candle_core::{Device, Tensor};

    #[test]
    fn test_splat_scale_from_u8() {
        assert_eq!(SplatScale::from_u8(0), SplatScale::Fine);
        assert_eq!(SplatScale::from_u8(1), SplatScale::Medium);
        assert_eq!(SplatScale::from_u8(2), SplatScale::Coarse);
        assert_eq!(SplatScale::from_u8(100), SplatScale::Coarse);
    }

    #[test]
    fn test_splat_scale_sigma_multiplier() {
        assert_eq!(SplatScale::Fine.sigma_multiplier(), 1.0);
        assert_eq!(SplatScale::Medium.sigma_multiplier(), 2.0);
        assert_eq!(SplatScale::Coarse.sigma_multiplier(), 4.0);
    }

    #[test]
    fn test_splat_new() {
        let device = Device::Cpu;
        let mu = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        let splat = Splat::new(mu, 1.0, 0.5);

        assert_eq!(splat.sigma, 1.0);
        assert_eq!(splat.alpha, 0.5);
        assert_eq!(splat.lambda, 0.02);
        assert_eq!(splat.scale, SplatScale::Fine);
        assert!(!splat.is_anchor);
    }

    #[test]
    fn test_splat_anchor() {
        let device = Device::Cpu;
        let mu = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        let splat = Splat::anchor(mu, 1.0, 0.5);

        assert_eq!(splat.sigma, 1.0);
        assert_eq!(splat.alpha, 0.5);
        assert_eq!(splat.lambda, 0.0);
        assert_eq!(splat.scale, SplatScale::Coarse);
        assert!(splat.is_anchor);
    }

    #[test]
    fn test_splat_with_scale() {
        let device = Device::Cpu;
        let mu = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();

        // Legacy absolute path (ref=30): fine / medium / coarse
        let splat_fine = Splat::with_scale(mu.clone(), 1.0, 0.5, 10.0);
        assert_eq!(splat_fine.scale, SplatScale::Fine);
        assert_eq!(splat_fine.sigma, 1.0);

        let splat_medium = Splat::with_scale(mu.clone(), 1.0, 0.5, 28.0);
        assert_eq!(splat_medium.scale, SplatScale::Medium);
        assert_eq!(splat_medium.sigma, 2.0);

        let splat_coarse = Splat::with_scale(mu.clone(), 1.0, 0.5, 40.0);
        assert_eq!(splat_coarse.scale, SplatScale::Coarse);
        assert_eq!(splat_coarse.sigma, 4.0);

        // Relative to 4B-ish deposit threshold (ref=70): δ=80 is medium not coarse×4
        let rel_med = Splat::with_scale_ref(mu.clone(), 1.0, 0.5, 80.0, 70.0);
        assert_eq!(rel_med.scale, SplatScale::Medium);
        assert_eq!(rel_med.sigma, 2.0);

        let rel_fine = Splat::with_scale_ref(mu.clone(), 1.0, 0.5, 50.0, 70.0);
        assert_eq!(rel_fine.scale, SplatScale::Fine);

        let rel_coarse = Splat::with_scale_ref(mu, 1.0, 0.5, 100.0, 70.0);
        assert_eq!(rel_coarse.scale, SplatScale::Coarse);
    }
}
