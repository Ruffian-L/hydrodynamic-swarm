#![allow(dead_code)]
//! Dream Replay
//!
//! After generation, replay top-K successful trajectories with Langevin noise
//! to reinforce good paths and update splat opacities.
//!
//! Placeholder for Day 5 implementation.

use candle_core::{Result, Tensor};

/// A recorded trajectory through the field during generation.
pub struct Trajectory {
    /// Sequence of positions visited
    pub positions: Vec<Tensor>,
    /// Reward score for this trajectory
    pub reward: f32,
}

/// Dream replay state.
pub struct DreamReplay {
    /// Stored trajectories from recent generations
    pub trajectories: Vec<Trajectory>,
    /// How many top trajectories to replay
    pub top_k: usize,
    /// Noise scale for replay perturbation
    pub replay_noise: f32,
}

impl DreamReplay {
    pub fn new(top_k: usize, replay_noise: f32) -> Self {
        Self {
            trajectories: Vec::new(),
            top_k,
            replay_noise,
        }
    }

    /// Record a trajectory from a generation run.
    pub fn record(&mut self, trajectory: Trajectory) {
        self.trajectories.push(trajectory);
    }

    /// Run dream replay: replay top-K trajectories with noise.
    /// Returns updated splat opacity deltas.
    ///
    /// TODO: Full implementation on Day 5
    pub fn replay(&self) -> Result<Vec<(usize, f32)>> {
        // Sort by reward, take top-K
        let mut sorted: Vec<_> = self.trajectories.iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.reward.partial_cmp(&a.1.reward).unwrap());

        let top: Vec<_> = sorted
            .into_iter()
            .take(self.top_k)
            .map(|(idx, t)| (idx, t.reward))
            .collect();

        Ok(top)
    }

    /// Clear old trajectories.
    pub fn clear(&mut self) {
        self.trajectories.clear();
    }
}
