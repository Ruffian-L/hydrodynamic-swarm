#![allow(dead_code)]
//! Ridge-Running Loop
//!
//! A query particle slides down the continuous Diderot field,
//! steered by splat scar tissue, until it settles on a stable ridge.
//! This proves the physics works before we touch the LLM.

use crate::field::ContinuousField;
use crate::gpu::PhysicsBackend;
use crate::memory::SplatMemory;
use candle_core::{DType, Result, Tensor};

/// A particle that rides the field gradient toward high-density ridges.
pub struct QueryParticle {
    pub pos: Tensor,
    pub vel: Tensor,
    pub mass: f32,
}

impl QueryParticle {
    pub fn new(start_pos: Tensor) -> Result<Self> {
        let dims = start_pos.dims().to_vec();
        let device = start_pos.device().clone();
        let vel = Tensor::zeros(&dims[..], DType::F32, &device)?;
        Ok(Self {
            pos: start_pos,
            vel,
            mass: 1.0,
        })
    }

    /// L2 norm of current velocity.
    pub fn speed(&self) -> Result<f32> {
        let v2: f32 = self.vel.sqr()?.sum_all()?.to_scalar()?;
        Ok(v2.sqrt())
    }

    /// L2 norm of current position.
    pub fn pos_norm(&self) -> Result<f32> {
        let p2: f32 = self.pos.sqr()?.sum_all()?.to_scalar()?;
        Ok(p2.sqrt())
    }
}

/// Drives a particle through the field until it settles on a ridge.
pub struct RidgeRunner<'a> {
    field: &'a ContinuousField,
    splat_memory: &'a SplatMemory,
    backend: &'a dyn PhysicsBackend,
    dt: f32,
    viscosity_scale: f32,
    damping: f32,
    goal_pos: Tensor,
}

impl<'a> RidgeRunner<'a> {
    pub fn new(
        field: &'a ContinuousField,
        splat_memory: &'a SplatMemory,
        backend: &'a dyn PhysicsBackend,
        goal_pos: Tensor,
    ) -> Self {
        Self {
            field,
            splat_memory,
            backend,
            dt: 0.01,
            viscosity_scale: 0.5,
            damping: 0.95,
            goal_pos,
        }
    }

    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_viscosity(mut self, v: f32) -> Self {
        self.viscosity_scale = v;
        self
    }

    pub fn with_damping(mut self, d: f32) -> Self {
        self.damping = d;
        self
    }

    /// The core ridge-running loop.
    ///
    /// The particle integrates forces from:
    /// 1. Field gradient (ridge-running force) -- via PhysicsBackend
    /// 2. Splat memory (scar tissue pull/push) -- via PhysicsBackend
    /// 3. Goal attractor (prompt embedding)
    ///
    /// Stops when velocity drops below threshold (settled on ridge)
    /// or max_steps reached.
    pub fn run(
        &self,
        mut particle: QueryParticle,
        max_steps: usize,
        settle_threshold: f32,
    ) -> Result<(QueryParticle, RunStats)> {
        let mut stats = RunStats::default();

        for step in 0..max_steps {
            // 1. Field gradient: the ridge-running force (via backend)
            let grad_force = self
                .backend
                .field_gradient(self.field, &particle.pos)?
                .affine(self.viscosity_scale as f64, 0.0)?;

            // 2. Splat scar tissue force (via backend)
            let splat_force = self.backend.splat_force(self.splat_memory, &particle.pos)?;

            // 3. Goal attractor: pull toward prompt embedding
            let goal_force = (&self.goal_pos - &particle.pos)?;

            // Sum all forces
            let total_force = ((&grad_force + &splat_force)? + &goal_force)?;

            // Euler integration: a = F/m, v += a*dt, x += v*dt
            let accel = total_force.affine(1.0 / particle.mass as f64, 0.0)?;
            let dv = accel.affine(self.dt as f64, 0.0)?;
            particle.vel = (&particle.vel + &dv)?;

            // Apply damping to prevent runaway
            particle.vel = particle.vel.affine(self.damping as f64, 0.0)?;

            let dx = particle.vel.affine(self.dt as f64, 0.0)?;
            particle.pos = (&particle.pos + &dx)?;

            // Track stats
            let speed = particle.speed()?;
            let density: f32 = self.field.probe(&particle.pos)?.to_scalar()?;

            stats.steps = step + 1;
            stats.final_speed = speed;
            stats.final_density = density;

            // Log every 20 steps
            if step % 20 == 0 {
                let pos_norm = particle.pos_norm()?;
                println!(
                    "    step {:>4} | speed: {:.6} | density: {:.6e} | pos_norm: {:.4}",
                    step, speed, density, pos_norm,
                );
            }

            // Settled on ridge?
            if speed < settle_threshold {
                stats.settled = true;
                println!(
                    "    -> Particle settled on ridge after {} steps (speed={:.6})",
                    step, speed
                );
                break;
            }
        }

        if !stats.settled {
            println!(
                "    -> Max steps reached ({}) (speed={:.6})",
                max_steps, stats.final_speed
            );
        }

        Ok((particle, stats))
    }

    /// Simplified ridge loop for testing splat memory forces.
    #[allow(dead_code)]
    pub fn run_with_memory(
        &self,
        mut particle: QueryParticle,
        max_steps: usize,
    ) -> Result<QueryParticle> {
        for step in 0..max_steps {
            let grad_force = self
                .backend
                .field_gradient(self.field, &particle.pos)?
                .affine(self.viscosity_scale as f64, 0.0)?;
            let splat_force = self.backend.splat_force(self.splat_memory, &particle.pos)?;
            let goal_force = (&self.goal_pos - &particle.pos)?;

            let total_force = ((&grad_force + &splat_force)? + &goal_force)?;

            // Euler integration: a = F/m, v += a*dt, x += v*dt
            let accel = total_force.affine(1.0 / particle.mass as f64, 0.0)?;
            let dv = accel.affine(self.dt as f64, 0.0)?;
            particle.vel = (&particle.vel + &dv)?;
            let dx = particle.vel.affine(self.dt as f64, 0.0)?;
            particle.pos = (&particle.pos + &dx)?;

            let vel_norm = particle.speed()?;
            if vel_norm < 0.001 {
                println!("    Particle settled on ridge after {} steps", step);
                break;
            }

            if step % 30 == 0 {
                println!(
                    "    step {:>4} | speed: {:.6} | pos_norm: {:.4}",
                    step,
                    vel_norm,
                    particle.pos_norm()?
                );
            }
        }
        Ok(particle)
    }
}

/// Statistics from a ridge-running session.
#[derive(Debug, Default)]
pub struct RunStats {
    pub steps: usize,
    pub settled: bool,
    pub final_speed: f32,
    pub final_density: f32,
}

/// Vietoris-Rips H1 reflex check.
///
/// Given a sliding window of recent positions (hidden states), checks all
/// triples for zero-persistence H1: a 1-cycle that is born and immediately
/// killed when the 2-simplex appears at the same filtration radius.
///
/// Detection: for a triple (a, b, c) with sorted edge lengths d0 <= d1 <= d2,
/// an H1 cycle is born at radius d1 (when the last of the first two edges
/// appears) and killed at d2 (when the third edge + 2-simplex appear).
/// Persistence = d2 - d1.  Zero-persistence ≈ d2/d1 close to 1.
///
/// Returns true if any triple has persistence ratio < `threshold` (e.g. 1.05).
pub fn check_vr_h1_reflex(positions: &[Tensor], threshold: f32) -> Result<bool> {
    let n = positions.len();
    if n < 3 {
        return Ok(false);
    }

    let start = n.saturating_sub(8);
    for i in start..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let d01: f32 = (&positions[i] - &positions[j])?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                let d02: f32 = (&positions[i] - &positions[k])?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                let d12: f32 = (&positions[j] - &positions[k])?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();

                let mut edges = [d01, d02, d12];
                edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let d_mid = edges[1];
                let d_max = edges[2];

                if d_mid > 1e-8 && d_max / d_mid < threshold {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
