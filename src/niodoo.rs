//! Niodoo Physics Steering Engine
//!
//! The core steering function: apply physics forces to the LLM residual stream.
//! Three forces act on the token trajectory each step:
//!   1. Field gradient (ridge-running): pulls toward high-density regions of the
//!      continuous Diderot embedding field. Scaled by viscosity.
//!   2. Splat scar tissue: accumulated Gaussian pleasure/pain scars pull/push
//!      the trajectory based on past generation experience.
//!   3. Goal attractor: linear pull toward the prompt's semantic goal position.
//!
//! The combined force is clamped per-element (force cap) to prevent runaway,
//! then scaled by dt and added to the residual.

use crate::field::ContinuousField;
use crate::memory::SplatMemory;
use candle_core::{Result, Tensor};

/// Result of a single steering step, including force telemetry.
pub struct SteerResult {
    pub steered: Tensor,
    pub grad_mag: f32,
    pub splat_mag: f32,
    pub goal_mag: f32,
}

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
            viscosity_scale: 0.35,
        }
    }

    /// Core steering: apply physics to LLM residual stream.
    ///
    /// `baseline_residual` must be shape `(1, D)` — single-batch residual.
    /// Returns the steered residual with the same shape `(1, D)`.
    ///
    /// steered = baseline + dt * (grad_force * viscosity + splat_force + goal_force)
    pub fn steer(
        &self,
        baseline_residual: &Tensor,
        goal_pos: &Tensor,
        _step: usize,
    ) -> Result<SteerResult> {
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

        // Force telemetry: capture magnitudes for JSONL logging
        let splat_mag: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let grad_mag: f32 = grad_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let goal_mag: f32 = goal_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Sum and scale by dt
        let total_force = ((&grad_force + &splat_force)? + &goal_force)?;
        // Force cap: prevent any single dimension from dominating (Variant 3)
        let total_force = total_force.clamp(-35f32, 35f32)?;
        let steering = total_force.affine(self.dt as f64, 0.0)?;

        // Restore batch dim: (D,) -> (1, D) and add to baseline
        let steering_2d = steering.unsqueeze(0)?;
        let steered = (baseline_residual + &steering_2d)?;

        Ok(SteerResult {
            steered,
            grad_mag,
            splat_mag,
            goal_mag,
        })
    }

    /// Get a reference to the memory for external queries.
    pub fn memory(&self) -> &SplatMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory for splat insertion.
    pub fn memory_mut(&mut self) -> &mut SplatMemory {
        &mut self.memory
    }

    /// Get a reference to the field's embedding positions for visualization.
    pub fn field_positions(&self) -> &Tensor {
        &self.field.positions
    }

    /// Get the embedding dimension.
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.field.dim
    }
}
