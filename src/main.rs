//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Retrieval and generation as a single physical process:
//! a particle sliding down a continuous Diderot field,
//! guided by splat scar tissue.

mod dream;
mod field;
mod memory;
mod niodoo;
mod ridge;
mod splat;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use dream::DreamEngine;
use field::ContinuousField;
use memory::SplatMemory;
use niodoo::NiodooEngine;
use ridge::{QueryParticle, RidgeRunner};
use splat::Splat;

fn main() -> Result<()> {
    println!("=== SplatRAG v1 — Hydrodynamic Swarm ===\n");

    // Use Metal GPU if available, fall back to CPU
    let device = match Device::new_metal(0) {
        Ok(d) => {
            println!("[*] Using Metal GPU");
            d
        }
        Err(_) => {
            println!("[*] Metal not available, using CPU");
            Device::Cpu
        }
    };

    // =========================================================
    // Day 1: Continuous Diderot Field
    // =========================================================
    println!("\n--- Day 1: Continuous Field ---");
    let field = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    println!(
        "[1] Field loaded: {} points × {} dims",
        field.n_points(),
        field.dim()
    );

    let probe_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let density: f32 = field.probe(&probe_pos)?.to_scalar()?;
    let grad = field.probe_gradient(&probe_pos)?;
    let grad_norm: f32 = grad.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!(
        "[1] Density: {:.6e}, Gradient norm: {:.6e}",
        density, grad_norm
    );

    // =========================================================
    // Day 2: Ridge-Running Loop
    // =========================================================
    println!("\n--- Day 2: Ridge-Running ---");
    let field2 = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    let memory2 = SplatMemory::new(device.clone());
    let goal2 = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let runner = RidgeRunner::new(field2, memory2, goal2)
        .with_dt(0.01)
        .with_viscosity(0.5)
        .with_damping(0.95);

    let start = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let particle = QueryParticle::new(start)?;
    println!("[2] Start pos norm: {:.4}", particle.pos_norm()?);

    let (final_p, stats) = runner.run(particle, 100, 0.001)?;
    println!(
        "[2] Steps: {} | Final speed: {:.4} | Final pos norm: {:.4}",
        stats.steps,
        stats.final_speed,
        final_p.pos_norm()?
    );

    // =========================================================
    // Day 3: Splat Memory (Scar Tissue)
    // =========================================================
    println!("\n--- Day 3: Splat Memory ---");
    let mut memory3 = SplatMemory::new(device.clone());

    let pleasure_pos = Tensor::zeros((512,), DType::F32, &device)?;
    memory3.add_splat(Splat::new(pleasure_pos, 0.15, 1.2));

    let pain_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    memory3.add_splat(Splat::new(pain_pos, 0.15, -0.8));
    println!("[3] Splats: {} (1 pleasure, 1 pain)", memory3.len());

    let test_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let force = memory3.query_force(&test_pos)?;
    let force_norm: f32 = force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!("[3] Splat force at random pos: {:.6e}", force_norm);

    // Test asymmetric decay
    let pre_decay_len = memory3.len();
    for _ in 0..5 {
        memory3.decay_step(0.9);
    }
    println!(
        "[3] After 5 decay steps: {} splats (pain persists longer)",
        memory3.len()
    );
    assert_eq!(pre_decay_len, memory3.len()); // splats don't vanish, just fade

    // =========================================================
    // Day 4: Niodoo Steering Hook
    // =========================================================
    println!("\n--- Day 4: Niodoo Steering ---");
    let field4 = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    let memory4 = SplatMemory::new(device.clone());
    let engine = NiodooEngine::new(field4, memory4);

    // Simulate a baseline residual (batch=1, dim=512)
    let baseline = Tensor::randn(0.0f32, 1.0, (1, 512), &device)?;
    let goal = Tensor::zeros((512,), DType::F32, &device)?;

    let steered = engine.steer(&baseline, &goal)?;

    let baseline_norm: f32 = baseline.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let steered_norm: f32 = steered.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let delta = (&steered - &baseline)?;
    let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

    println!("[4] Baseline norm: {:.6}", baseline_norm);
    println!("[4] Steered norm:  {:.6}", steered_norm);
    println!(
        "[4] Delta norm:    {:.6} (physics steering applied)",
        delta_norm
    );

    // =========================================================
    // Day 5: Dream Replay
    // =========================================================
    println!("\n--- Day 5: Dream Replay ---");
    let dream_memory = SplatMemory::new(device.clone());
    let mut dream = DreamEngine::new(dream_memory);

    let traj1 = Tensor::randn(0.0f32, 1.0, (10, 512), &device)?;
    let traj2 = Tensor::randn(0.0f32, 1.0, (8, 512), &device)?;
    dream.run(vec![traj1, traj2], 0.05)?;
    println!("[5] Dream replay complete.");

    // =========================================================
    // Summary
    // =========================================================
    println!("\n========================================");
    println!("  ✅ SplatRAG v1 — ALL PHASES COMPLETE");
    println!("========================================");
    println!("  [1] Continuous Diderot Field  ✓");
    println!("  [2] Ridge-Running Loop        ✓");
    println!("  [3] Splat Memory (Scar Tissue) ✓");
    println!("  [4] Niodoo Steering Hook      ✓");
    println!("  [5] Dream Replay              ✓");
    println!("========================================");
    println!("\nNext: Load real embeddings → hook to LLM residual stream → generate.");

    Ok(())
}
