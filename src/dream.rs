//! Dream Replay
//!
//! After generation, replay top-K successful trajectories with Langevin noise
//! to reinforce good paths and update splat opacities.

use crate::memory::SplatMemory;
use candle_core::{Result, Tensor};

pub struct DreamEngine {
    memory: SplatMemory,
}

impl DreamEngine {
    pub fn new(memory: SplatMemory) -> Self {
        Self { memory }
    }

    /// Simple dream replay: replay trajectories with noise, update splats.
    ///
    /// In full v1 this would project noisy trajectories back through the field
    /// and update splat alphas based on trajectory reward. For now we add
    /// Langevin noise to successful trajectories and apply global decay.
    pub fn run(&mut self, success_trajectories: Vec<Tensor>, noise_scale: f32) -> Result<()> {
        for traj in &success_trajectories {
            // Add Langevin noise to trajectory
            let noise = Tensor::randn(0.0f32, noise_scale, traj.dims(), traj.device())?;
            let _noisy = (traj + &noise)?;

            // TODO: Project noisy trajectory through field, update splat alphas
            // based on whether the perturbed path still reaches high-reward regions.
            println!(
                "    Dream replay: processed trajectory (shape {:?}, noise {:.4})",
                traj.dims(),
                noise_scale
            );
        }

        // Global decay — scars fade over time
        self.memory.decay_step(0.98);
        println!(
            "    Applied global decay (0.98). Splats remaining: {}",
            self.memory.len()
        );

        Ok(())
    }

    /// Access the memory after replay (for reinsertion into engine).
    #[allow(dead_code)]
    pub fn into_memory(self) -> SplatMemory {
        self.memory
    }
}
