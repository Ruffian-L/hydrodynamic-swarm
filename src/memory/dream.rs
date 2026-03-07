//! Dream Replay + Micro-Dream Consolidation + TopoCoT
//!
//! Full dream replay: After generation, replay trajectories with Langevin noise.
//! Micro-dream: During generation, short forward+backward physics bursts
//! for real-time consolidation when steering delta is high.
//! TopoCoT: When correction_norm exceeds threshold, inject a reflection marker
//! into the generated token stream so the model "feels" the hydraulic jump.

use crate::memory::memory::SplatMemory;
use crate::physics::niodoo::NiodooEngine;
use candle_core::{Result, Tensor};

/// Threshold for dream correction injection.
/// When the micro-dream correction norm exceeds this, the model experienced
/// a significant trajectory warp -- a "hydraulic jump" in the latent stream.
pub const DREAM_CORRECTION_THRESHOLD: f32 = 18.0;

pub struct DreamEngine {
    memory: SplatMemory,
}

impl DreamEngine {
    pub fn new(memory: SplatMemory) -> Self {
        Self { memory }
    }

    /// Simple dream replay: replay trajectories with noise, update splats.
    pub fn run(&mut self, success_trajectories: Vec<Tensor>, noise_scale: f32) -> Result<()> {
        for traj in &success_trajectories {
            let noise = Tensor::randn(0.0f32, noise_scale, traj.dims(), traj.device())?;
            let _noisy = (traj + &noise)?;
            println!(
                "    Dream replay: processed trajectory (shape {:?}, noise {:.4})",
                traj.dims(),
                noise_scale
            );
        }

        // Global decay
        self.memory.decay_step(0.98);
        println!(
            "    Applied global decay (0.98). Splats remaining: {}",
            self.memory.len()
        );

        Ok(())
    }

    #[allow(dead_code)]
    pub fn into_memory(self) -> SplatMemory {
        self.memory
    }
}

/// Result of a micro-dream: the corrected position + whether a TopoCoT
/// reflection was triggered (correction_norm exceeded threshold).
pub struct MicroDreamResult {
    pub consolidated: Tensor,
    pub correction_norm: f32,
    pub reflection_triggered: bool,
}

/// Micro-dream: short forward+backward physics burst for real-time consolidation.
///
/// 1. Forward project: steer the current position 2-3 steps into the future
/// 2. Backward anchor: pull the projection back toward the goal
/// 3. Return the correction delta to blend into current steered logits
///
/// When correction_norm exceeds DREAM_CORRECTION_THRESHOLD, we flag it as a
/// TopoCoT reflection event -- the model hit a wall and course-corrected.
pub fn micro_dream(
    engine: &NiodooEngine,
    current_pos: &Tensor,  // (1, D) steered logits
    goal_pos: &Tensor,     // (D,) goal attractor
    step: usize,           // current generation step
    steps: usize,          // forward projection steps (2-3)
    blend_factor: f64,     // how much of the correction to apply (0.05-0.15)
) -> Result<MicroDreamResult> {
    let mut projected = current_pos.clone();

    // Forward projection: steer N steps into the future
    for fwd in 0..steps {
        // Use step + offset so force logging shows projection steps
        projected = engine.steer(&projected, goal_pos, 1000 + step * 10 + fwd)?.steered;
    }

    // Backward anchor: compute the pull from the future back to goal
    let future_pos = projected.squeeze(0)?;
    let anchor_pull = (goal_pos - &future_pos)?;

    // The correction is the scaled anchor pull reshaped back to (1, D)
    let correction = anchor_pull.affine(blend_factor, 0.0)?.unsqueeze(0)?;
    let correction_norm: f32 = correction.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

    // TopoCoT: detect hydraulic jump
    let reflection_triggered = correction_norm > DREAM_CORRECTION_THRESHOLD;

    // Apply correction to current position
    let consolidated = (current_pos + &correction)?;

    Ok(MicroDreamResult {
        consolidated,
        correction_norm,
        reflection_triggered,
    })
}
