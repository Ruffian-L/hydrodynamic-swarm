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

/// A single token neighbor with its cosine similarity score.
#[derive(Serialize, Clone)]
pub struct TokenNeighbor {
    pub token_id: u32,
    pub token_text: String,
    pub similarity: f32,
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
    pub splat_positions_3d: Vec<[f32; 3]>,
    pub splat_alphas: Vec<f32>,
    pub goal_position_3d: [f32; 3],
    pub prompt: String,
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
    snapshots: Vec<VizSnapshot>,
    field_points_3d: Vec<[f32; 3]>,
    goal_3d: [f32; 3],
    prompt: String,
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

        // Project field positions (subsample large fields)
        let n = field_positions.dim(0)?;
        let stride = if n > 5000 { n / 5000 } else { 1 };
        let positions_flat: Vec<f32> = field_positions.flatten_all()?.to_vec1()?;
        let mut field_points_3d = Vec::with_capacity(n / stride);
        for i in (0..n).step_by(stride) {
            let row = &positions_flat[i * dim..(i + 1) * dim];
            field_points_3d.push(project_vec(row, &projection, dim));
        }

        // Project goal
        let goal_flat: Vec<f32> = goal_pos.to_vec1()?;
        let goal_3d = project_vec(&goal_flat, &projection, dim);

        println!(
            "    [VIZ] Collector ready: {} field points projected to 3D",
            field_points_3d.len()
        );

        Ok(Self {
            projection,
            dim,
            snapshots: Vec::new(),
            field_points_3d,
            goal_3d,
            prompt: prompt.to_string(),
        })
    }

    /// Capture a snapshot at the current generation step.
    /// This is designed to be cheap -- just a tensor copy + dot products.
    pub fn snapshot(
        &mut self,
        step: usize,
        token_id: u32,
        token_text: &str,
        steered_pos: &Tensor, // (1, D) or (D,)
        steering_delta: f32,
    ) -> anyhow::Result<()> {
        let pos_flat: Vec<f32> = steered_pos.flatten_all()?.to_vec1()?;
        let pos_3d = project_vec(&pos_flat, &self.projection, self.dim);

        self.snapshots.push(VizSnapshot {
            step,
            token_id,
            token_text: token_text.to_string(),
            position_3d: pos_3d,
            steering_delta,
            neighbors: Vec::new(),
        });

        Ok(())
    }

    /// Export all collected data to a JSON file.
    pub fn export_json(&self, path: &Path) -> anyhow::Result<()> {
        let session = VizSession {
            prompt: self.prompt.clone(),
            embedding_dim: self.dim,
            snapshots: self.snapshots.clone(),
            field_points_3d: self.field_points_3d.clone(),
            splat_scars: Vec::new(),
            goal_position_3d: self.goal_3d,
        };

        let json = serde_json::to_string(&session)?;
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
        let trajectory_3d: Vec<[f32; 3]> =
            self.snapshots.iter().map(|s| s.position_3d).collect();
        let trajectory_deltas: Vec<f32> =
            self.snapshots.iter().map(|s| s.steering_delta).collect();

        VizRenderData {
            field_points_3d: self.field_points_3d,
            trajectory_3d,
            trajectory_deltas,
            splat_positions_3d: Vec::new(),
            splat_alphas: Vec::new(),
            goal_position_3d: self.goal_3d,
            prompt: self.prompt,
        }
    }

    /// Number of snapshots collected so far.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.snapshots.len()
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
