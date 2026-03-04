//! Continuous Diderot Field
//!
//! The field is a sum of Gaussian kernels over all stored memory positions.
//! Dimension-agnostic: auto-detects D from the safetensors file.
//! `probe_gradient(pos)` returns the gradient vector — the ridge-running force.

use candle_core::{DType, Device, Result, Tensor};

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
    /// Build the field directly from a model's token embedding matrix.
    /// This is the preferred path: no external files, guaranteed alignment
    /// with the actual model, and no risk of all-zero placeholder data.
    ///
    /// `embeddings` should be shape (vocab_size, hidden_dim) -- the raw
    /// `tok_embeddings` tensor from the loaded ModelWeights.
    pub fn from_embeddings(embeddings: &Tensor, device: &Device) -> Result<Self> {
        let positions = embeddings.to_dtype(DType::F32)?.to_device(device)?;
        // L2-normalize each embedding to unit norm so field lives on the
        // unit hypersphere, matching the unit-normalized query pos in steer().
        let norms = positions.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?.unsqueeze(1)?;
        let positions = positions.broadcast_div(&norms)?;
        let dim = positions.dim(positions.dims().len() - 1)?;
        let n = positions.dim(0)?;

        println!("    Building Diderot field from model tok_embeddings...");
        println!("    Shape: {} tokens x {} dims", n, dim);

        // Auto-tune sigma from sampled pairwise distances
        let sigma = if n >= 2 {
            let n_pairs = 200usize.min(n * (n - 1) / 2);
            let mut total_dist = 0.0f64;
            let mut rng = 42u64; // deterministic LCG
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
            let s = if mean_dist > 0.1 {
                mean_dist * 0.5
            } else {
                // Fallback for degenerate data on unit sphere
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
            "    Field LIVE: {} points x {} dims | sigma = {:.4}",
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

    /// Compute the gradient using only the K nearest field points (approximate).
    ///
    /// Instead of evaluating all N field points (O(N*D)), only the K nearest
    /// by L2 distance are used. For N=128K and K=2048, this is ~60x faster.
    /// Uses partial sort (select_nth_unstable) to find K nearest in O(N) time.
    ///
    /// Falls back to exact gradient if K >= N.
    pub fn probe_gradient_topk(&self, pos: &Tensor, k: usize) -> Result<Tensor> {
        let n = self.n_points();
        if k >= n || n == 0 {
            return self.probe_gradient(pos);
        }

        // Compute squared distances to all field points
        let pos_expanded = pos.unsqueeze(0)?;
        let diff_all = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq_all: Vec<f32> = diff_all.sqr()?.sum(1)?.to_vec1()?;

        // Partial sort to find K nearest indices
        let mut indexed: Vec<(usize, f32)> = dist_sq_all
            .iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Gather only the K nearest positions
        let topk_indices: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
        let topk_rows: Vec<Tensor> = topk_indices
            .iter()
            .map(|&i| self.positions.get(i).and_then(|r| r.unsqueeze(0)))
            .collect::<Result<Vec<_>>>()?;
        let topk_positions = Tensor::cat(&topk_rows, 0)?;

        // Standard gradient computation on just the K nearest
        let diff = topk_positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, &self.device);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field(positions: Tensor, sigma: f32, dim: usize) -> ContinuousField {
        ContinuousField {
            device: positions.device().clone(),
            positions,
            kernel_sigma: sigma,
            dim,
        }
    }

    #[test]
    fn gradient_pulls_toward_field_point() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let grad = field.probe_gradient(&pos).unwrap();
        let grad_vec: Vec<f32> = grad.to_vec1().unwrap();

        // diff = positions - query = [0,0,0,0] - [1,0,0,0] = [-1,0,0,0]
        // gradient should be negative x (toward origin from x=1)
        assert!(
            grad_vec[0] < 0.0,
            "expected negative x gradient, got {}",
            grad_vec[0]
        );
    }

    #[test]
    fn density_positive_at_field_point() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        let density: f32 = field.probe(&pos).unwrap().to_scalar().unwrap();
        assert!(
            density > 0.0,
            "density at field point should be > 0, got {}",
            density
        );
    }

    #[test]
    fn gradient_zero_far_away() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 0.1, 4);

        let pos = Tensor::new(&[1000.0f32, 1000.0, 1000.0, 1000.0], &device).unwrap();
        let grad = field.probe_gradient(&pos).unwrap();
        let mag: f32 = grad
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            mag < 1e-10,
            "gradient far from field should be ~0, got {}",
            mag
        );
    }

    #[test]
    fn topk_gradient_matches_exact_for_small_n() {
        // With N=5 points and K=5, Top-K should produce the exact same result
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[5, 4], &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::new(&[0.5f32, -0.3, 0.7, 0.1], &device).unwrap();
        let exact = field.probe_gradient(&pos).unwrap();
        let topk = field.probe_gradient_topk(&pos, 5).unwrap();

        let diff: f32 = (&exact - &topk)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();

        assert!(
            diff < 1e-5,
            "Top-K (K=N) should match exact gradient, diff={}",
            diff
        );
    }
}
