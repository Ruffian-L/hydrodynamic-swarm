//! Niodoo Physics Steering
//!
//! The core steering function that combines all forces and applies them
//! to the baseline residual stream. This is where retrieval becomes generation.

use crate::field::ContinuousField;
use crate::memory::SplatMemory;
use candle_core::{Result, Tensor};

/// Physics parameters for the steering loop.
pub struct NiodooParams {
    /// How much the field gradient influences steering
    pub viscosity_scale: f32,
    /// Integration timestep
    pub dt: f32,
    /// Momentum damping factor (0 = no momentum, 1 = full momentum)
    pub momentum_decay: f32,
    /// Noise scale for exploration (Langevin-style)
    pub noise_scale: f32,
}

impl Default for NiodooParams {
    fn default() -> Self {
        Self {
            viscosity_scale: 0.1,
            dt: 0.01,
            momentum_decay: 0.9,
            noise_scale: 0.001,
        }
    }
}

/// Compute the total steering force from all physics components.
///
/// total_force = grad_force + splat_force + goal_force + momentum + noise
pub fn compute_steering_force(
    field: &ContinuousField,
    memory: &SplatMemory,
    query_pos: &Tensor,
    goal_pos: &Tensor,
    momentum: &Tensor,
    params: &NiodooParams,
) -> Result<Tensor> {
    // Field gradient: the ridge-running force
    let grad_force = field
        .probe_gradient(query_pos)?
        .affine(params.viscosity_scale as f64, 0.0)?;

    // Splat scar tissue force
    let splat_force = memory.query_force(query_pos)?;

    // Goal-directed force: pull toward prompt embedding
    let goal_force = (goal_pos - query_pos)?;

    // Momentum (damped)
    let damped_momentum = momentum.affine(params.momentum_decay as f64, 0.0)?;

    // Langevin noise for exploration
    let noise = Tensor::randn(0.0f32, params.noise_scale, query_pos.dims(), field.device())?;

    // Sum all forces
    let total = ((grad_force + splat_force)? + goal_force)?;
    let total = (total + damped_momentum)?;
    let total = (total + noise)?;

    Ok(total)
}

/// Apply the steering force to a baseline residual.
///
/// steered_residual = baseline_residual + (total_force * dt)
pub fn steer_residual(baseline_residual: &Tensor, total_force: &Tensor, dt: f32) -> Result<Tensor> {
    let delta = total_force.affine(dt as f64, 0.0)?;
    baseline_residual + delta
}
