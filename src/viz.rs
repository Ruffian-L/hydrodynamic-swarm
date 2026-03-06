//! SplatLens -- Visualization Data Collector
//!
//! Captures per-step snapshots from the generation pipeline,
//! projects 4096D embedding vectors to 3D via random projection,
//! and exports session data as JSON for the Metal renderer or replay.
//!
//! This module is purely READ-ONLY: it never mutates the engine,
//! field, memory, or any generation state.

use candle_core::Tensor;
use serde::Serialize;
use std::path::Path;

// ---------------------------------------------------------------
// Data types
// ---------------------------------------------------------------

/// A single token neighbor with its model probability.
#[derive(Serialize, Clone)]
pub struct TokenNeighbor {
    pub token_id: u32,
    pub token_text: String,
    /// Softmax probability from the model (used for sizing/labeling, not projection).
    pub probability: f32,
    #[serde(skip_serializing_if = "is_zero_position")]
    pub position_3d: [f32; 3],
}

/// Returns true if a 3D position is all-zero (degenerate projection).
fn is_zero_position(pos: &[f32; 3]) -> bool {
    pos[0] == 0.0 && pos[1] == 0.0 && pos[2] == 0.0
}

/// Per-step visualization snapshot.
#[derive(Serialize, Clone)]
pub struct VizSnapshot {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub position_3d: [f32; 3],
    pub steering_delta: f32,
    pub neighbors: Vec<TokenNeighbor>,
}

/// Splat scar projected to 3D.
#[derive(Serialize, Clone)]
pub struct VizSplat {
    pub position_3d: [f32; 3],
    pub alpha: f32,
    pub sigma: f32,
}

/// Full visualization session -- JSON-serializable.
#[derive(Serialize)]
pub struct VizSession {
    pub prompt: String,
    pub embedding_dim: usize,
    pub snapshots: Vec<VizSnapshot>,
    pub field_points_3d: Vec<[f32; 3]>,
    pub splat_scars: Vec<VizSplat>,
    pub goal_position_3d: [f32; 3],
}

/// Lightweight render data passed to the Metal window.
pub struct VizRenderData {
    pub field_points_3d: Vec<[f32; 3]>,
    pub trajectory_3d: Vec<[f32; 3]>,
    pub trajectory_deltas: Vec<f32>,
    pub trajectory_tokens: Vec<String>,
    pub splat_positions_3d: Vec<[f32; 3]>,
    pub splat_alphas: Vec<f32>,
    pub goal_position_3d: [f32; 3],
    pub prompt: String,
    /// Per-step neighbor data: Vec of (step_index, neighbors)
    pub step_neighbors: Vec<Vec<StepNeighbor>>,
    /// Ridge ghost trail (predicted path from ridge runner)
    pub ridge_ghost: Vec<[f32; 3]>,
}

/// Neighbor data for a single step, ready for rendering.
pub struct StepNeighbor {
    pub token_text: String,
    pub probability: f32,
    pub position_3d: [f32; 3],
}

// ---------------------------------------------------------------
// Collector
// ---------------------------------------------------------------

/// Collects visualization data during generation.
/// Created once after the field and goal are available.
pub struct VizCollector {
    /// Random projection matrix, flat layout (D * 3)
    projection: Vec<f32>,
    dim: usize,
    /// Flat copy of field positions for neighbor projection
    field_positions_flat: Vec<f32>,
    snapshots: Vec<VizSnapshot>,
    field_points_3d: Vec<[f32; 3]>,
    goal_3d: [f32; 3],
    prompt: String,
    /// Ridge ghost trail points (projected to 3D)
    ridge_ghost: Vec<[f32; 3]>,
}

impl VizCollector {
    /// Create a new collector.
    ///
    /// Builds a deterministic random projection matrix (seed=42) and
    /// projects all field embedding positions to 3D. Subsamples if
    /// the field has more than 5000 points to keep rendering fast.
    pub fn new(
        field_positions: &Tensor, // (N, D)
        goal_pos: &Tensor,        // (D,)
        prompt: &str,
        dim: usize,
    ) -> anyhow::Result<Self> {
        // Deterministic random projection matrix (D x 3)
        // Simple LCG for reproducibility, no external RNG crate needed.
        let mut rng_state: u64 = 42;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut projection = vec![0.0f32; dim * 3];
        for val in projection.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            *val = (u + v - 1.0) * scale;
        }

        // Keep flat field positions for neighbor projection
        let field_positions_flat: Vec<f32> = field_positions.flatten_all()?.to_vec1()?;

        // Validate that field_positions_flat has the expected layout
        let n = field_positions.dim(0)?;
        let expected_len = n * dim;
        if field_positions_flat.len() != expected_len {
            return Err(anyhow::anyhow!(
                "[VIZ] field_positions_flat length mismatch: got {} expected {} (n={}, dim={}). \
                 Embeddings may not match the full vocabulary layout.",
                field_positions_flat.len(),
                expected_len,
                n,
                dim
            ));
        }

        // Project field positions (subsample large fields)
        let stride = if n > 5000 { n / 5000 } else { 1 };
        let mut field_points_3d = Vec::with_capacity(n / stride);
        for i in (0..n).step_by(stride) {
            let row = &field_positions_flat[i * dim..(i + 1) * dim];
            field_points_3d.push(project_vec(row, &projection, dim));
        }

        // Project goal
        let goal_flat: Vec<f32> = goal_pos.to_vec1()?;
        let goal_3d = project_vec(&goal_flat, &projection, dim);

        println!(
            "    [VIZ] Collector ready: {} field points projected to 3D (vocab={})",
            field_points_3d.len(),
            n
        );

        Ok(Self {
            projection,
            dim,
            field_positions_flat,
            snapshots: Vec::new(),
            field_points_3d,
            goal_3d,
            prompt: prompt.to_string(),
            ridge_ghost: Vec::new(),
        })
    }

    /// Capture a snapshot at the current generation step.
    /// `neighbors` is a list of (token_id, token_text, probability)
    /// where probability is the softmax model probability.
    pub fn snapshot(
        &mut self,
        step: usize,
        token_id: u32,
        token_text: &str,
        steered_pos: &Tensor, // (1, D) or (D,)
        steering_delta: f32,
        neighbors: Vec<(u32, String, f32)>,
    ) -> anyhow::Result<()> {
        let pos_flat: Vec<f32> = steered_pos.flatten_all()?.to_vec1()?;
        let pos_3d = project_vec(&pos_flat, &self.projection, self.dim);

        // Project neighbor positions to 3D using field embedding positions
        let neighbor_data: Vec<TokenNeighbor> = neighbors
            .into_iter()
            .filter_map(|(tid, text, prob)| {
                let idx = tid as usize;
                if idx * self.dim + self.dim <= self.field_positions_flat.len() {
                    let row = &self.field_positions_flat[idx * self.dim..(idx + 1) * self.dim];
                    let pos = project_vec(row, &self.projection, self.dim);
                    Some(TokenNeighbor {
                        token_id: tid,
                        token_text: text,
                        probability: prob,
                        position_3d: pos,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.snapshots.push(VizSnapshot {
            step,
            token_id,
            token_text: token_text.to_string(),
            position_3d: pos_3d,
            steering_delta,
            neighbors: neighbor_data,
        });

        Ok(())
    }

    /// Set the ridge ghost trail (projected from 4096D to 3D).
    #[allow(dead_code)]
    pub fn set_ridge_ghost(&mut self, positions: &[Vec<f32>]) {
        self.ridge_ghost = positions
            .iter()
            .map(|p| project_vec(p, &self.projection, self.dim))
            .collect();
    }

    /// Export all collected data to a JSON file.
    /// Detects degenerate all-zero field_points_3d and omits them with a warning.
    pub fn export_json(&self, path: &Path) -> anyhow::Result<()> {
        // Detect degenerate (all-zero) field_points_3d
        let field_points_degenerate = !self.field_points_3d.is_empty()
            && self
                .field_points_3d
                .iter()
                .all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0);

        if field_points_degenerate {
            eprintln!(
                "    [VIZ] WARNING: field_points_3d is all-zero (degenerate projection). \
                 Omitting from export. Check embedding data and projection matrix."
            );
        }

        let session = VizSession {
            prompt: self.prompt.clone(),
            embedding_dim: self.dim,
            snapshots: self.snapshots.clone(),
            field_points_3d: if field_points_degenerate {
                Vec::new()
            } else {
                self.field_points_3d.clone()
            },
            splat_scars: Vec::new(),
            goal_position_3d: self.goal_3d,
        };

        let json = serde_json::to_string_pretty(&session)?;
        std::fs::write(path, json)?;
        println!(
            "    [VIZ] Exported {} snapshots to {}",
            self.snapshots.len(),
            path.display()
        );
        Ok(())
    }

    /// Convert into render data for the Metal window.
    pub fn into_render_data(self) -> VizRenderData {
        let trajectory_3d: Vec<[f32; 3]> = self.snapshots.iter().map(|s| s.position_3d).collect();
        let trajectory_deltas: Vec<f32> = self.snapshots.iter().map(|s| s.steering_delta).collect();
        let trajectory_tokens: Vec<String> = self
            .snapshots
            .iter()
            .map(|s| s.token_text.clone())
            .collect();

        // Collect per-step neighbor data for rendering
        let step_neighbors: Vec<Vec<StepNeighbor>> = self
            .snapshots
            .iter()
            .map(|s| {
                s.neighbors
                    .iter()
                    .map(|n| StepNeighbor {
                        token_text: n.token_text.clone(),
                        probability: n.probability,
                        position_3d: n.position_3d,
                    })
                    .collect()
            })
            .collect();

        VizRenderData {
            field_points_3d: self.field_points_3d,
            trajectory_3d,
            trajectory_deltas,
            trajectory_tokens,
            splat_positions_3d: Vec::new(),
            splat_alphas: Vec::new(),
            goal_position_3d: self.goal_3d,
            prompt: self.prompt,
            step_neighbors,
            ridge_ghost: self.ridge_ghost,
        }
    }

    /// Number of snapshots collected so far.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    pub fn add_step(&mut self, text: String, current_pos: &Tensor, memory: &crate::memory::SplatMemory, dt: f32) -> anyhow::Result<()> {
        let mut _neighbors: Vec<()> = Vec::new();
        let current_flat = current_pos.flatten_all()?;
        let current_vec: Vec<f32> = current_flat.to_vec1()?;

        for splat in memory.splats_ref() {
            let splat_flat = splat.mu.flatten_all()?;
            let splat_vec: Vec<f32> = splat_flat.to_vec1()?;
            let _dist = self.euclidean_distance(&current_vec, &splat_vec);
        }

        self.snapshots.push(VizSnapshot {
            step: self.snapshots.len(),
            token_id: 0,
            token_text: text,
            position_3d: [0.0, 0.0, 0.0],
            steering_delta: dt,
            neighbors: Vec::new(),
        });

        Ok(())
    }
}

// ---------------------------------------------------------------
// Projection utilities
// ---------------------------------------------------------------

/// Project a D-dimensional vector to 3D via the random projection matrix.
fn project_vec(vec: &[f32], projection: &[f32], dim: usize) -> [f32; 3] {
    let mut result = [0.0f32; 3];
    let len = dim.min(vec.len());
    for j in 0..3 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += vec[i] * projection[i * 3 + j];
        }
        result[j] = sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use crate::memory::SplatMemory;

    #[test]
    fn test_add_step_boundary_cases() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let dim = 4;

        let field_positions = Tensor::zeros((10, dim), candle_core::DType::F32, &device)?;
        let goal_pos = Tensor::zeros((dim,), candle_core::DType::F32, &device)?;

        let mut collector = VizCollector::new(&field_positions, &goal_pos, "test prompt", dim)?;
        assert_eq!(collector.len(), 0);

        let current_pos = Tensor::zeros((dim,), candle_core::DType::F32, &device)?;
        let mut memory = SplatMemory::new(device.clone());

        // Test with empty memory
        collector.add_step("token1".to_string(), &current_pos, &memory, 0.1)?;
        assert_eq!(collector.len(), 1);

        // Add a splat to memory and test again
        let splat_pos = Tensor::ones((dim,), candle_core::DType::F32, &device)?;
        memory.add_splat(crate::splat::Splat::new(splat_pos.clone(), 1.0, 1.0));

        collector.add_step("token2".to_string(), &current_pos, &memory, 0.2)?;
        assert_eq!(collector.len(), 2);

        Ok(())
    }
}
