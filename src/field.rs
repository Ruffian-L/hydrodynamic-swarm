#![allow(dead_code)]
//! Continuous Diderot Field
//!
//! The field is a sum of Gaussian kernels over all stored memory positions.
//! Dimension-agnostic: auto-detects D from the safetensors file.
//! `probe_gradient(pos)` returns the gradient vector — the ridge-running force.

use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

pub struct ContinuousField {
    /// Memory positions from embeddings, shape (N, D)
    pub positions: Tensor,
    pub device: Device,
    /// Controls the width of each Gaussian kernel (auto-tuned)
    pub kernel_sigma: f32,
    /// Embedding dimension (auto-detected from data)
    pub dim: usize,
}

impl ContinuousField {
    /// Load real embeddings from a safetensors file.
    /// Dimension-agnostic: auto-detects D and tunes sigma.
    pub fn load_real(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        println!("    Loading: {}", path.display());

        let tensors = candle_core::safetensors::load(path, device)?;

        // Print available keys
        let keys: Vec<_> = tensors.keys().collect();
        println!("    Keys found: {:?}", keys);

        // Try common key names, or take the largest tensor
        let positions = if let Some(t) = tensors.get("embeddings") {
            t.clone()
        } else if let Some(t) = tensors.get("tensor") {
            t.clone()
        } else if let Some(t) = tensors.get("weight") {
            t.clone()
        } else {
            tensors
                .values()
                .max_by_key(|t| t.elem_count())
                .expect("safetensors file is empty")
                .clone()
        };

        let positions = positions.to_dtype(DType::F32)?;
        let dim = positions.dim(positions.dims().len() - 1)?;
        let n = positions.dim(0)?;

        // Auto-tune sigma from actual mean pairwise distance.
        // Sample up to 200 random pairs and compute mean L2 distance,
        // then set sigma = mean_dist * 0.5 so Gaussian kernels overlap.
        let sigma = if n >= 2 {
            let n_pairs = 200usize.min(n * (n - 1) / 2);
            let mut total_dist = 0.0f64;
            let mut rng = 0u64; // simple LCG for deterministic sampling
            for _ in 0..n_pairs {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let i = (rng >> 33) as usize % n;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut j = (rng >> 33) as usize % (n - 1);
                if j >= i {
                    j += 1;
                }
                let pi = positions.get(i)?;
                let pj = positions.get(j)?;
                let diff = (&pi - &pj)?;
                let dist: f32 = diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                total_dist += dist as f64;
            }
            let mean_dist = (total_dist / n_pairs as f64) as f32;
            let s = if mean_dist > 1.0 {
                mean_dist * 0.5
            } else {
                // Fallback for degenerate data
                (dim as f32).sqrt() * 0.035
            };
            println!(
                "    Sigma auto-tuned: mean_dist={:.2}, sigma={:.4}",
                mean_dist, s
            );
            s
        } else {
            (dim as f32).sqrt() * 0.035
        };

        println!(
            "    Field loaded: {} points x {} dims | sigma = {:.4}",
            n, dim, sigma
        );

        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: sigma,
            dim,
        })
    }

    /// Load dummy random embeddings (for testing).
    #[allow(dead_code)]
    pub fn load_dummy(dim: usize, n_points: usize, device: &Device) -> Result<Self> {
        let positions = Tensor::randn(0.0f32, 1.0, (n_points, dim), device)?;
        let sigma = (dim as f32).sqrt() * 0.035;
        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: sigma,
            dim,
        })
    }

    /// Probe the scalar density at a position.
    pub fn probe(&self, pos: &Tensor) -> Result<Tensor> {
        let pos_expanded = pos.unsqueeze(0)?;
        let diff = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;
        kernel.sum_all()
    }

    /// Compute the gradient of the density field at a position.
    /// NaN-safe: returns zero gradient when all kernels underflow (fast path).
    pub fn probe_gradient(&self, pos: &Tensor) -> Result<Tensor> {
        let pos_expanded = pos.unsqueeze(0)?;
        let diff = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient (fast path)
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), DType::F32, &self.device);
        }

        let kernel_expanded = kernel.unsqueeze(1)?;
        let weighted = diff.broadcast_mul(&kernel_expanded)?;
        let scale = 2.0 / sigma_sq as f64;
        let grad = weighted.sum(0)?.squeeze(0)?.affine(scale, 0.0)?;
        Ok(grad)
    }

    pub fn n_points(&self) -> usize {
        self.positions.dim(0).unwrap_or(0)
    }
}
