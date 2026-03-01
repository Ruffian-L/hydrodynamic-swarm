//! Niodoo Physics Steering Engine
//!
//! The core steering function: apply physics forces to the LLM residual stream.
//! This is where retrieval becomes generation — the same physical process.

use crate::field::ContinuousField;
use crate::memory::SplatMemory;
use candle_core::{Result, Tensor};

pub struct NiodooEngine {
    field: ContinuousField,
    memory: SplatMemory,
    dt: f32,
    viscosity_scale: f32,
}

impl NiodooEngine {
    pub fn new(field: ContinuousField, memory: SplatMemory) -> Self {
        Self {
            field,
            memory,
            dt: 0.08,
            viscosity_scale: 0.6,
        }
    }

    /// Core steering: apply physics to LLM residual stream.
    ///
    /// `baseline_residual` must be shape `(1, D)` — single-batch residual.
    /// Returns the steered residual with the same shape `(1, D)`.
    ///
    /// steered = baseline + dt * (grad_force * viscosity + splat_force + goal_force)
    pub fn steer(&self, baseline_residual: &Tensor, goal_pos: &Tensor) -> Result<Tensor> {
        // Shape validation: require exactly (1, D)
        let dims = baseline_residual.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "steer: baseline_residual must be 2D (batch, dim), got {}D shape {:?}",
                dims.len(),
                dims
            )));
        }
        if dims[0] != 1 {
            return Err(candle_core::Error::Msg(format!(
                "steer: baseline_residual batch size must be 1, got {} (shape {:?}). \
                 Multi-batch steering is not supported in v1.",
                dims[0], dims
            )));
        }

        // Extract position vector: (1, D) -> (D,)
        let pos = baseline_residual.squeeze(0)?;

        // 1. Field gradient: ridge-running force
        let grad_force = self
            .field
            .probe_gradient(&pos)?
            .affine(self.viscosity_scale as f64, 0.0)?;

        // 2. Splat scar tissue force
        let splat_force = self.memory.query_force(&pos)?;

        // 3. Goal attractor
        let goal_force = (goal_pos - &pos)?;

        // Sum and scale by dt
        let total_force = ((&grad_force + &splat_force)? + &goal_force)?;
        // Force cap: prevent any single dimension from dominating (Variant 3)
        let total_force = total_force.clamp(-80f32, 80f32)?;
        let steering = total_force.affine(self.dt as f64, 0.0)?;

        // Restore batch dim: (D,) -> (1, D) and add to baseline
        let steering_2d = steering.unsqueeze(0)?;
        baseline_residual + &steering_2d
    }

    /// Get a reference to the memory for external updates.
    #[allow(dead_code)]
    pub fn memory(&self) -> &SplatMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory.
    #[allow(dead_code)]
    pub fn memory_mut(&mut self) -> &mut SplatMemory {
        &mut self.memory
    }
}
