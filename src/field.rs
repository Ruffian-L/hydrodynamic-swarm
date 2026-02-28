#![allow(dead_code)]
//! Continuous Diderot Field
//!
//! The field is a sum of Gaussian kernels over all stored memory positions.
//! `probe(pos)` returns a scalar density, `probe_gradient(pos)` returns the
//! gradient vector — the ridge-running force that drives particle steering.

use candle_core::{Device, Result, Tensor};
use std::path::Path;

pub struct ContinuousField {
    /// Memory positions from embeddings, shape (N, D)
    positions: Tensor,
    device: Device,
    /// Controls the width of each Gaussian kernel
    kernel_sigma: f32,
}

impl ContinuousField {
    /// Load embeddings from a safetensors file and build the continuous field.
    ///
    /// For now uses random data — swap in real safetensors load for your
    /// universe_domain embeddings.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let _path = path.as_ref(); // will use this once we load real data

        // TODO: replace with real safetensors load of universe_domain
        // For now we create random embedding space
        let n_points = 20_000;
        let dim = 512;
        let positions = Tensor::randn(0.0f32, 1.0, (n_points, dim), device)?;

        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: 0.08,
        })
    }

    /// Create a small demo field (1k points, 64D) for fast CPU iteration.
    /// Sigma is tuned so kernels are non-trivial at this scale.
    pub fn load_demo(device: &Device) -> Result<Self> {
        let n_points = 1_000;
        let dim = 64;
        // Cluster some points around the origin so the field has structure
        let positions = Tensor::randn(0.0f32, 1.0, (n_points, dim), device)?;

        Ok(Self {
            positions,
            device: device.clone(),
            // In 64D, typical dist between random unit-var points ≈ sqrt(2*64) ≈ 11.3
            // σ=5.0 means kernels are exp(-11.3²/25) ≈ exp(-5.1) ≈ 0.006 — small but non-zero
            kernel_sigma: 5.0,
        })
    }

    /// Load from an existing tensor of positions (N, D).
    pub fn from_positions(positions: Tensor, device: &Device) -> Self {
        Self {
            positions,
            device: device.clone(),
            kernel_sigma: 0.08,
        }
    }

    /// Set kernel width (controls how broad each Gaussian bump is).
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.kernel_sigma = sigma;
        self
    }

    /// Probe the scalar density at a position.
    ///
    /// density(pos) = Σ_i exp(-||pos - μ_i||² / σ²)
    pub fn probe(&self, pos: &Tensor) -> Result<Tensor> {
        // diff: (N, D) = positions - pos broadcast
        let pos_expanded = pos.unsqueeze(0)?; // (1, D)
        let diff = self.positions.broadcast_sub(&pos_expanded)?; // (N, D)
                                                                 // dist_sq: (N,)
        let dist_sq = diff.sqr()?.sum(1)?;
        // kernel: (N,) — Gaussian weight per point
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;
        // scalar density = sum of all kernels
        kernel.sum_all()
    }

    /// Compute the gradient of the density field at a position.
    ///
    /// ∇density(pos) = Σ_i  2(μ_i - pos)/σ²  · exp(-||pos - μ_i||² / σ²)
    ///
    /// This is the ridge-running force — the particle rides this gradient
    /// toward high-density regions of the memory field.
    pub fn probe_gradient(&self, pos: &Tensor) -> Result<Tensor> {
        // diff: (N, D) = positions - pos
        let pos_expanded = pos.unsqueeze(0)?; // (1, D)
        let diff = self.positions.broadcast_sub(&pos_expanded)?; // (N, D)
                                                                 // dist_sq: (N,)
        let dist_sq = diff.sqr()?.sum(1)?;
        // kernel weights: (N,)
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels are zero (underflow), return zero gradient
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, &self.device);
        }

        // weighted diff: (N, D) * (N, 1)
        let kernel_expanded = kernel.unsqueeze(1)?; // (N, 1)
        let weighted = diff.broadcast_mul(&kernel_expanded)?; // (N, D)
                                                              // gradient: (D,) = sum over N, scaled by 2/σ²
        let scale = 2.0 / sigma_sq as f64;
        let grad = weighted.sum(0)?.squeeze(0)?.affine(scale, 0.0)?;
        Ok(grad)
    }

    /// Number of memory points in the field.
    pub fn n_points(&self) -> usize {
        self.positions.dim(0).unwrap_or(0)
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.positions.dim(1).unwrap_or(0)
    }

    /// Reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_probe_gradient_shape() -> Result<()> {
        let device = Device::Cpu;
        let field = ContinuousField::load("dummy", &device)?;

        // Create a query point
        let query = Tensor::randn(0.0f32, 1.0, (512,), &device)?;

        // Gradient should be same shape as query (D,)
        let grad = field.probe_gradient(&query)?;
        assert_eq!(grad.dims(), &[512]);

        Ok(())
    }

    #[test]
    fn test_field_probe_density_scalar() -> Result<()> {
        let device = Device::Cpu;
        let dim = 8; // small dim so kernels are non-trivial

        // Place known points near the query
        let query = Tensor::zeros((dim,), candle_core::DType::F32, &device)?;
        let positions = Tensor::randn(0.0f32, 0.01, (100, dim), &device)?; // tight cluster at origin
        let field = ContinuousField::from_positions(positions, &device).with_sigma(1.0);

        // Density should be a scalar
        let density = field.probe(&query)?;
        assert_eq!(density.dims(), &[] as &[usize]);

        // Density should be positive (sum of exponentials near origin)
        let val: f32 = density.to_scalar()?;
        assert!(val > 0.0, "density should be positive, got {}", val);

        Ok(())
    }

    #[test]
    fn test_field_metadata() -> Result<()> {
        let device = Device::Cpu;
        let field = ContinuousField::load("dummy", &device)?;

        assert_eq!(field.n_points(), 20_000);
        assert_eq!(field.dim(), 512);

        Ok(())
    }
}
