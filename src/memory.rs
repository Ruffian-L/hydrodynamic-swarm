//! SplatMemory — manages a collection of splats and computes aggregate forces.
//!
//! This is the "scar tissue" layer: accumulated experience that biases
//! the particle's trajectory through the field.
//! Pain lasts longer than pleasure (asymmetric decay).

use crate::splat::Splat;
use candle_core::{DType, Result, Tensor};

pub struct SplatMemory {
    splats: Vec<Splat>,
    device: candle_core::Device,
}

impl SplatMemory {
    pub fn new(device: candle_core::Device) -> Self {
        Self {
            splats: Vec::new(),
            device,
        }
    }

    pub fn add_splat(&mut self, splat: Splat) {
        self.splats.push(splat);
    }

    /// Asymmetric decay: pain lasts longer than pleasure.
    /// Pain decays at 70% of the pleasure rate.
    pub fn decay_step(&mut self, decay_rate: f32) {
        for splat in &mut self.splats {
            if splat.alpha > 0.0 {
                // pleasure decays faster
                splat.alpha *= decay_rate;
            } else {
                // pain decays slower (70% of decay rate)
                splat.alpha *= decay_rate * 0.7;
            }
            // prevent complete disappearance — scars persist
            if splat.alpha.abs() < 0.01 {
                splat.alpha *= 0.95;
            }
        }
    }

    /// Core function: summed Gaussian pull/push from all nearby splats.
    ///
    /// For each splat: force = α · (μ - pos) · exp(-||μ - pos||² / σ²)
    /// Positive α pulls toward the splat (pleasure), negative pushes away (pain).
    pub fn query_force(&self, pos: &Tensor) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        let mut total_force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;

        for splat in &self.splats {
            // diff = μ - pos (direction toward splat)
            let diff = (&splat.mu - pos)?;
            // dist² = ||μ - pos||²
            let dist_sq: f32 = diff.sqr()?.sum_all()?.to_scalar()?;
            // Gaussian kernel: exp(-dist²/σ²)
            let sigma_sq = splat.sigma * splat.sigma;
            let kernel = (-dist_sq / sigma_sq).exp();
            // force = α · kernel · diff
            let scale = (splat.alpha * kernel) as f64;
            let signed_force = diff.affine(scale, 0.0)?;

            total_force = (&total_force + &signed_force)?;
        }
        Ok(total_force)
    }

    /// Number of active splats.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    /// Prune dead splats below threshold.
    #[allow(dead_code)]
    pub fn prune(&mut self, threshold: f32) {
        self.splats.retain(|s| s.alpha.abs() >= threshold);
    }
}
