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
    /// steered = baseline + dt * (grad_force * viscosity + splat_force + goal_force)
    pub fn steer(&self, baseline_residual: &Tensor, goal_pos: &Tensor) -> Result<Tensor> {
        // Current hidden position (simplified for v1: squeeze batch dim)
        let pos = baseline_residual.squeeze(0)?; // (1, 512) -> (512,)

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
        let steering = total_force.affine(self.dt as f64, 0.0)?;

        // Apply steering to residual (unsqueeze back to batch dim)
        let steering_2d = steering.unsqueeze(0)?; // (512,) -> (1, 512)
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
