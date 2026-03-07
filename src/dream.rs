//! Dream Replay + Micro-Dream Consolidation + TopoCoT
//!
//! Full dream replay: After generation, replay trajectories with Langevin noise.
//! Micro-dream: During generation, short forward+backward physics bursts
//! for real-time consolidation when steering delta is high.
//! TopoCoT: When correction_norm exceeds threshold, inject a reflection marker
//! into the generated token stream so the model "feels" the hydraulic jump.

use crate::memory::SplatMemory;
use crate::niodoo::NiodooEngine;
use candle_core::{Result, Tensor};

/// Threshold for dream correction injection.
/// When the micro-dream correction norm exceeds this, the model experienced
/// a significant trajectory warp -- a "hydraulic jump" in the latent stream.
pub const DREAM_CORRECTION_THRESHOLD: f32 = 6.0;

/// Full dream replay engine (Langevin + consolidate). Reserved for future use.
#[allow(dead_code)]
pub struct DreamEngine {
    memory: SplatMemory,
}

#[allow(dead_code)]
impl DreamEngine {
    pub fn new(memory: SplatMemory) -> Self {
        Self { memory }
    }

    /// Dream replay: add Langevin noise to trajectories and consolidate into memory.
    pub fn run(
        &mut self,
        trajectories: Vec<Tensor>,
        noise_scale: f32,
        sigma: f32,
        alpha_bonus: f32,
        min_dist: f32,
        decay_rate: f32,
    ) -> Result<()> {
        for traj in trajectories {
            let noise = Tensor::randn(0.0f32, noise_scale, traj.dims(), traj.device())?;
            let noisy = (&traj + &noise)?;
            let created =
                self.memory
                    .consolidate_trajectory(&noisy, sigma, alpha_bonus, min_dist, None)?;
            println!(
                "    Dream replay: {} points -> {} splats (noise {:.4})",
                traj.dim(0).unwrap_or(0),
                created,
                noise_scale
            );
        }

        self.memory.decay_step(decay_rate);
        println!(
            "    Applied decay ({:.3}). Splats: {}",
            decay_rate,
            self.memory.len()
        );
        Ok(())
    }

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
    pub reflection_tokens: Vec<u32>,
    pub reflection_text: String,
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
    current_pos: &Tensor, // (1, D) steered logits
    goal_pos: &Tensor,    // (D,) goal attractor
    step: usize,          // current generation step
    steps: usize,         // forward projection steps (2-3)
    blend_factor: f64,    // how much of the correction to apply (0.05-0.15)
) -> Result<MicroDreamResult> {
    let mut projected = current_pos.clone();

    // Forward projection: steer N steps into the future
    for fwd in 0..steps {
        // Use step + offset so force logging shows projection steps
        projected = engine
            .steer(&projected, goal_pos, 1000 + step * 10 + fwd)?
            .steered;
    }

    // Backward anchor: compute the pull from the future back to goal
    let future_pos = projected.squeeze(0)?;
    let anchor_pull = (goal_pos - &future_pos)?;

    // The correction is the scaled anchor pull reshaped back to (1, D)
    let correction = anchor_pull.affine(blend_factor, 0.0)?.unsqueeze(0)?;
    let correction_norm: f32 = correction.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

    // TopoCoT: detect hydraulic jump — clamp correction to prevent overshoot
    let reflection_triggered = correction_norm > DREAM_CORRECTION_THRESHOLD;
    let correction = if reflection_triggered && correction_norm > 0.0 {
        let scale = (DREAM_CORRECTION_THRESHOLD / correction_norm) as f64;
        correction.affine(scale, 0.0)?
    } else {
        correction
    };

    let consolidated = (current_pos + &correction)?;

    Ok(MicroDreamResult {
        consolidated,
        correction_norm,
        reflection_triggered,
        reflection_tokens: vec![],
        reflection_text: if reflection_triggered {
            format!(
                "Wait... the trajectory just jumped. The scar tissue from earlier is pulling harder than I expected. Recalibrating toward the original goal now...\n"
            )
        } else {
            String::new()
        },
    })
}
