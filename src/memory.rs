#![allow(dead_code)]
//! SplatMemory — manages a collection of splats and computes aggregate forces.
//!
//! This is the "scar tissue" layer: accumulated experience that biases
//! the particle's trajectory through the field.

use crate::splat::Splat;
use candle_core::{Device, Result, Tensor};

pub struct SplatMemory {
    pub splats: Vec<Splat>,
    pub device: Device,
    /// Reward memories decay with this halflife
    pub reward_halflife: f32,
    /// Pain memories decay slower (asymmetric)
    pub pain_halflife: f32,
}

impl SplatMemory {
    pub fn new(device: &Device) -> Self {
        Self {
            splats: Vec::new(),
            device: device.clone(),
            reward_halflife: 100.0,
            pain_halflife: 500.0, // pain lasts 5x longer
        }
    }

    /// Add a new splat (memory scar).
    pub fn add_splat(&mut self, splat: Splat) {
        self.splats.push(splat);
    }

    /// Create and add a reward splat at a position.
    pub fn record_reward(&mut self, position: Tensor, sigma: f32, strength: f32, time: u64) {
        self.splats
            .push(Splat::new(position, sigma, strength, 1.0, time));
    }

    /// Create and add a pain splat at a position.
    pub fn record_pain(&mut self, position: Tensor, sigma: f32, strength: f32, time: u64) {
        self.splats
            .push(Splat::new(position, sigma, strength, -1.0, time));
    }

    /// Compute the total force from all splats on a query position.
    ///
    /// This is the scar tissue steering: accumulated reward/pain
    /// pulls the particle toward good regions and away from bad ones.
    pub fn query_force(&self, query_pos: &Tensor) -> Result<Tensor> {
        let dim = query_pos.dims()[0];
        let mut total = Tensor::zeros((dim,), candle_core::DType::F32, &self.device)?;

        for splat in &self.splats {
            if splat.alpha.abs() < 1e-6 {
                continue; // skip dead splats
            }
            let force = splat.force_on(query_pos)?;
            total = (total + force)?;
        }

        Ok(total)
    }

    /// Decay all splats (asymmetric: pain lasts longer).
    pub fn decay_all(&mut self, current_time: u64) {
        for splat in &mut self.splats {
            splat.decay(current_time, self.reward_halflife, self.pain_halflife);
        }
    }

    /// Prune dead splats (alpha below threshold).
    pub fn prune(&mut self, threshold: f32) {
        self.splats.retain(|s| s.alpha.abs() >= threshold);
    }

    /// Number of active splats.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splats.is_empty()
    }
}
