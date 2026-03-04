#![allow(dead_code)]
//! Continuous Diderot Field
//!
//! The field is a sum of Gaussian kernels over all stored memory positions.
//! Dimension-agnostic: auto-detects D from the safetensors file.
//! `probe_gradient(pos)` returns the gradient vector — the ridge-running force.

use candle_core::{DType, Device, Result, Tensor};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

#[derive(Copy, Clone, PartialEq)]
struct HeapEntry {
    val: f32,
    idx: usize,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

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
        // L2-normalize each embedding to unit norm so field lives on the
        // unit hypersphere, matching the unit-normalized query pos in steer().
        let norms = positions.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?.unsqueeze(1)?;
        let positions = positions.broadcast_div(&norms)?;
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
            let s = if mean_dist > 0.1 {
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
        // L2-normalize to unit norm (matches real loader)
        let norms = positions.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?.unsqueeze(1)?;
        let positions = positions.broadcast_div(&norms)?;
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
        // Efficient dist_sq using dot products: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        // positions are unit-normalized, so ||positions_i||^2 = 1.0
        let pos_norm_sq: f32 = pos.sqr()?.sum_all()?.to_scalar()?;
        let dot = self.positions.matmul(&pos.unsqueeze(1)?)?.squeeze(1)?;
        let dist_sq = dot.affine(-2.0, (pos_norm_sq + 1.0) as f64)?;

        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;
        kernel.sum_all()
    }

    /// Compute the gradient of the density field at a position.
    /// NaN-safe: returns zero gradient when all kernels underflow (fast path).
    pub fn probe_gradient(&self, pos: &Tensor) -> Result<Tensor> {
        // Efficient dist_sq using dot products: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        let pos_norm_sq: f32 = pos.sqr()?.sum_all()?.to_scalar()?;
        let dot = self.positions.matmul(&pos.unsqueeze(1)?)?.squeeze(1)?;
        let dist_sq = dot.affine(-2.0, (pos_norm_sq + 1.0) as f64)?;

        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient (fast path)
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), DType::F32, &self.device);
        }

        // Optimized gradient identity: sum(k_i * (x_i - p)) = sum(k_i * x_i) - p * sum(k_i)
        // sum_k_x = kernel (1, N) matmul positions (N, D) -> (1, D)
        let sum_k_x = kernel.unsqueeze(0)?.matmul(&self.positions)?.squeeze(0)?;
        let p_sum_k = pos.affine(kernel_sum as f64, 0.0)?;
        let scale = 2.0 / sigma_sq as f64;
        let grad = (sum_k_x - p_sum_k)?.affine(scale, 0.0)?;
        Ok(grad)
    }

    /// Find the K nearest field point indices (= token IDs) to a position.
    /// Returns Vec of (index, cosine_similarity) sorted by similarity descending.
    pub fn nearest_tokens(&self, pos: &Tensor, k: usize) -> anyhow::Result<Vec<(u32, f32)>> {
        let n = self.n_points();
        let k = k.min(n);
        if k == 0 || n == 0 {
            return Ok(Vec::new());
        }

        // Rank by dot product (higher dot product = smaller L2 distance on unit sphere)
        let dots_tensor = self.positions.matmul(&pos.unsqueeze(1)?)?.squeeze(1)?;
        let dots: Vec<f32> = dots_tensor.to_vec1()?;

        // Use a min-heap to keep track of the Top-K largest dots
        let mut heap = BinaryHeap::with_capacity(k);
        for (i, &val) in dots.iter().enumerate() {
            if heap.len() < k {
                heap.push(Reverse(HeapEntry { val, idx: i }));
            } else if val > heap.peek().unwrap().0.val {
                heap.pop();
                heap.push(Reverse(HeapEntry { val, idx: i }));
            }
        }
        let indices: Vec<usize> = heap.into_iter().map(|Reverse(e)| e.idx).collect();

        // Compute cosine similarities for the Top-K using batch operations
        let topk_indices_tensor = Tensor::new(indices.iter().map(|&i| i as u32).collect::<Vec<_>>().as_slice(), &self.device)?;
        let topk_positions = self.positions.index_select(&topk_indices_tensor, 0)?;

        // cos_sim(a, b) = dot(a,b) / (|a| * |b|)
        // positions are unit-normalized (|a|=1.0)
        let pos_norm: f32 = pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let topk_dots = topk_positions.matmul(&pos.unsqueeze(1)?)?.squeeze(1)?;
        let cos_sims: Vec<f32> = topk_dots.affine(1.0 / (pos_norm.max(1e-12) as f64), 0.0)?.to_vec1()?;

        let mut results: Vec<(u32, f32)> = indices
            .into_iter()
            .zip(cos_sims.into_iter())
            .map(|(idx, sim)| (idx as u32, sim))
            .collect();

        // Sort by cosine similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
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

        // Efficient dist_sq using dot products
        let pos_norm_sq: f32 = pos.sqr()?.sum_all()?.to_scalar()?;
        let dots_tensor = self.positions.matmul(&pos.unsqueeze(1)?)?.squeeze(1)?;
        let dots: Vec<f32> = dots_tensor.to_vec1()?;

        // Use a min-heap to keep track of the Top-K largest dots (nearest neighbors)
        let mut heap = BinaryHeap::with_capacity(k);
        for (i, &val) in dots.iter().enumerate() {
            if heap.len() < k {
                heap.push(Reverse(HeapEntry { val, idx: i }));
            } else if val > heap.peek().unwrap().0.val {
                heap.pop();
                heap.push(Reverse(HeapEntry { val, idx: i }));
            }
        }
        let indices: Vec<usize> = heap.into_iter().map(|Reverse(e)| e.idx).collect();

        // Gather only the K nearest positions
        let topk_indices_tensor = Tensor::new(indices.iter().map(|&i| i as u32).collect::<Vec<_>>().as_slice(), &self.device)?;
        let topk_positions = self.positions.index_select(&topk_indices_tensor, 0)?;

        // Standard gradient computation on just the K nearest
        // Recalculate dist_sq for the topk subset (dots were already computed, just gather)
        let topk_dots = dots_tensor.index_select(&topk_indices_tensor, 0)?;
        let dist_sq = topk_dots.affine(-2.0, (pos_norm_sq + 1.0) as f64)?;

        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, &self.device);
        }

        // Optimized gradient identity: sum(k_i * (x_i - p)) = sum(k_i * x_i) - p * sum(k_i)
        let sum_k_x = kernel.unsqueeze(0)?.matmul(&topk_positions)?.squeeze(0)?;
        let p_sum_k = pos.affine(kernel_sum as f64, 0.0)?;
        let scale = 2.0 / sigma_sq as f64;
        let grad = (sum_k_x - p_sum_k)?.affine(scale, 0.0)?;
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
