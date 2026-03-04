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

    /// Time-based exponential decay: V(t) = V0 * exp(-lambda * delta_t).
    /// Asymmetric: pain decays at 70% of the pleasure rate.
    /// Anchors (lambda=0 or is_anchor=true) never decay.
    /// `decay_rate` is the legacy per-step fallback for splats without lambda.
    pub fn decay_step(&mut self, decay_rate: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for splat in &mut self.splats {
            // Anchors never decay
            if splat.is_anchor || splat.lambda == 0.0 {
                continue;
            }

            let dt = (now.saturating_sub(splat.created_at)) as f32;
            let effective_lambda = if splat.alpha < 0.0 {
                // Pain lasts longer: 70% decay rate
                splat.lambda * 0.7
            } else {
                splat.lambda
            };

            if dt > 0.0 {
                // Exponential decay: alpha *= exp(-lambda * dt)
                let decay_factor = (-effective_lambda * dt).exp();
                splat.alpha *= decay_factor;
            } else {
                // Fallback: per-step decay for freshly created splats
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
        self.splats.retain(|s| s.is_anchor || s.alpha.abs() >= threshold);
        before - self.splats.len()
    }

    /// Core function: summed Gaussian pull/push from all nearby splats.
    ///
    /// For each splat: force = alpha * (mu - pos) * exp(-||mu - pos||^2 / sigma^2)
    /// Positive alpha pulls toward the splat (pleasure), negative pushes away (pain).
    pub fn query_force(&self, pos: &Tensor) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        let mut total_force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;

        for splat in &self.splats {
            let diff = (&splat.mu - pos)?;
            let dist_sq: f32 = diff.sqr()?.sum_all()?.to_scalar()?;
            let sigma_sq = splat.sigma * splat.sigma;
            let kernel = (-dist_sq / sigma_sq).exp();
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

    /// Read-only access to splat data (used by GPU backend for buffer upload).
    pub fn splats_ref(&self) -> &[Splat] {
        &self.splats
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

    /// Remove all normal splats whose absolute alpha is below `threshold`.
    /// Anchor splats (lambda == 0.0) are never pruned.
    pub fn prune(&mut self, threshold: f32) {
        let initial = self.splats.len();
        self.splats.retain(|s| s.is_anchor || s.alpha.abs() >= threshold);
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

            // Find nearby same-sign splats
            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..self.splats.len() {
                if consumed[j] {
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
                    // Weighted mean of mu (cluster_mu is current centroid)
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

            if cluster_size > 1 {
                merge_count += cluster_size - 1;
            }
            // Preserve the strongest splat's metadata for the merged result
            let is_anchor = self.splats[i].is_anchor;
            let scale = self.splats[i].scale;
            let lambda = if is_anchor { 0.0 } else { self.splats[i].lambda };
            merged.push(Splat {
                mu: cluster_mu,
                sigma: cluster_sigma,
                alpha: cluster_alpha,
                lambda,
                created_at: self.splats[i].created_at,
                scale,
                is_anchor,
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

    /// Keep only the N strongest splats (by |alpha|), discarding the weakest.
    pub fn prune_to_limit(&mut self, max_count: usize) {
        if self.splats.len() <= max_count {
            return;
        }
        self.splats
            .sort_by(|a, b| b.alpha.abs().total_cmp(&a.alpha.abs()));
        self.splats.truncate(max_count);
        println!("    [PRUNE] Capped to {} strongest splats", max_count);
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

        // Scalar fields
        let sigmas: Vec<f32> = self.splats.iter().map(|s| s.sigma).collect();
        let alphas: Vec<f32> = self.splats.iter().map(|s| s.alpha).collect();
        let lambdas: Vec<f32> = self.splats.iter().map(|s| s.lambda).collect();
        let created_ats: Vec<f32> = self.splats.iter().map(|s| s.created_at as f32).collect();
        let scales: Vec<f32> = self.splats.iter().map(|s| s.scale as u8 as f32).collect();
        let anchors: Vec<f32> = self.splats.iter().map(|s| if s.is_anchor { 1.0 } else { 0.0 }).collect();

        let sigma_tensor = Tensor::from_vec(sigmas, n, &self.device)?;
        let alpha_tensor = Tensor::from_vec(alphas, n, &self.device)?;
        let lambda_tensor = Tensor::from_vec(lambdas, n, &self.device)?;
        let created_at_tensor = Tensor::from_vec(created_ats, n, &self.device)?;
        let scale_tensor = Tensor::from_vec(scales, n, &self.device)?;
        let anchor_tensor = Tensor::from_vec(anchors, n, &self.device)?;

        // Convert to raw bytes
        let mu_data: Vec<f32> = mu_stack.flatten_all()?.to_vec1()?;
        let sigma_data: Vec<f32> = sigma_tensor.to_vec1()?;
        let alpha_data: Vec<f32> = alpha_tensor.to_vec1()?;
        let lambda_data: Vec<f32> = lambda_tensor.to_vec1()?;
        let created_at_data: Vec<f32> = created_at_tensor.to_vec1()?;
        let scale_data: Vec<f32> = scale_tensor.to_vec1()?;
        let anchor_data: Vec<f32> = anchor_tensor.to_vec1()?;

        let to_bytes = |data: &[f32]| -> Vec<u8> {
            data.iter().flat_map(|f| f.to_le_bytes()).collect()
        };

        let mu_bytes = to_bytes(&mu_data);
        let sigma_bytes = to_bytes(&sigma_data);
        let alpha_bytes = to_bytes(&alpha_data);
        let lambda_bytes = to_bytes(&lambda_data);
        let created_at_bytes = to_bytes(&created_at_data);
        let scale_bytes = to_bytes(&scale_data);
        let anchor_bytes = to_bytes(&anchor_data);

        let mu_shape = mu_stack.dims().to_vec();
        let n_shape = vec![n];

        let mu_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, mu_shape, &mu_bytes)?;
        let sigma_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape.clone(), &sigma_bytes)?;
        let alpha_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape.clone(), &alpha_bytes)?;
        let lambda_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape.clone(), &lambda_bytes)?;
        let created_at_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape.clone(), &created_at_bytes)?;
        let scale_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape.clone(), &scale_bytes)?;
        let anchor_view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape, &anchor_bytes)?;

        let tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
            ("mu".to_string(), mu_view),
            ("sigma".to_string(), sigma_view),
            ("alpha".to_string(), alpha_view),
            ("lambda".to_string(), lambda_view),
            ("created_at".to_string(), created_at_view),
            ("scale".to_string(), scale_view),
            ("is_anchor".to_string(), anchor_view),
        ];

        safetensors::tensor::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            &None::<std::collections::HashMap<String, String>>,
            path,
        )?;

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!("    Saved {} splats ({} anchors) to {}", n, anchor_count, path.display());
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

        // Optional v2 fields (backward-compatible)
        let lambda_data: Option<Vec<f32>> = tensors.tensor("lambda").ok().map(|v| parse_f32(v.data()));
        let created_at_data: Option<Vec<f32>> = tensors.tensor("created_at").ok().map(|v| parse_f32(v.data()));
        let scale_data: Option<Vec<f32>> = tensors.tensor("scale").ok().map(|v| parse_f32(v.data()));
        let anchor_data: Option<Vec<f32>> = tensors.tensor("is_anchor").ok().map(|v| parse_f32(v.data()));

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
            let scale = scale_data.as_ref().map_or(crate::splat::SplatScale::Fine, |v| crate::splat::SplatScale::from_u8(v[i] as u8));
            let is_anchor = anchor_data.as_ref().is_some_and(|v| v[i] > 0.5);

            self.splats.push(Splat {
                mu: mu_tensor,
                sigma: sigma_data[i],
                alpha: alpha_data[i],
                lambda,
                created_at,
                scale,
                is_anchor,
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
}
