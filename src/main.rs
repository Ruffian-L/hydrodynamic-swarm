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
use field::ContinuousField;
use memory::SplatMemory;
use niodoo::{compute_steering_force, steer_residual, NiodooParams};
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

    // Phase 1: Load continuous field from embeddings
    println!("[1] Loading continuous Diderot field...");
    let field = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    println!(
        "    Field: {} points × {} dims",
        field.n_points(),
        field.dim()
    );

    // Phase 2: Initialize empty splat memory
    println!("[2] Initializing splat memory...");
    let memory = SplatMemory::new(device.clone());
    println!("    Splats: {} (empty — no scars yet)", memory.len());

    // Phase 3: Demonstrate the physics
    println!("[3] Running single-step physics demo...\n");

    let params = NiodooParams::default();

    // Simulate a query position (would come from hidden state in real usage)
    let query_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    // Simulate a goal position (would come from prompt embedding)
    let goal_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    // Zero initial momentum
    let momentum = Tensor::zeros((512,), candle_core::DType::F32, &device)?;
    // Simulate a baseline residual
    let baseline_residual = Tensor::randn(0.0f32, 0.1, (512,), &device)?;

    // Probe the field
    let density = field.probe(&query_pos)?;
    let density_val: f32 = density.to_scalar()?;
    println!("    Field density at query: {:.6}", density_val);

    // Compute gradient
    let gradient = field.probe_gradient(&query_pos)?;
    let grad_norm: f32 = gradient.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!("    Gradient magnitude:     {:.6}", grad_norm);

    // Compute full steering force
    let total_force =
        compute_steering_force(&field, &memory, &query_pos, &goal_pos, &momentum, &params)?;
    let force_norm: f32 = total_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!("    Total force magnitude:  {:.6}", force_norm);

    // Apply steering to residual
    let steered = steer_residual(&baseline_residual, &total_force, params.dt)?;
    let steered_norm: f32 = steered.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let baseline_norm: f32 = baseline_residual
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt();
    println!("    Baseline residual norm: {:.6}", baseline_norm);
    println!("    Steered residual norm:  {:.6}", steered_norm);
    println!(
        "    Delta:                  {:.6}",
        (steered_norm - baseline_norm).abs()
    );

    println!("\n[✓] Day 1 physics pipeline working.");

    // === DAY 2: RIDGE-RUNNING DEMO ===
    println!("\n=== Day 2: Ridge-Running Demo ===");
    println!("[4] Building ridge runner...\n");

    let field2 = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    let splat_memory2 = SplatMemory::new(device.clone());
    let goal_pos2 = Tensor::randn(0.0f32, 1.0, (512,), &device)?;

    let runner = RidgeRunner::new(field2, splat_memory2, goal_pos2)
        .with_dt(0.01)
        .with_viscosity(0.5)
        .with_damping(0.95);

    let start_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let particle = QueryParticle::new(start_pos)?;

    println!("    Start position norm: {:.6}", particle.pos_norm()?);
    println!("    Running 200 steps...\n");

    let (final_particle, stats) = runner.run(particle, 200, 0.001)?;

    println!("\n    --- Results ---");
    println!("    Steps:              {}", stats.steps);
    println!("    Settled:            {}", stats.settled);
    println!("    Final speed:        {:.6}", stats.final_speed);
    println!("    Final density:      {:.6e}", stats.final_density);
    println!("    Final position norm: {:.6}", final_particle.pos_norm()?);

    println!("\n[✓] Day 2 ridge-running complete.");

    // === DAY 3: SPLAT MEMORY TEST ===
    println!("\n=== Day 3: Splat Memory Test ===");
    println!("[5] Testing pleasure + pain splats...\n");

    let mut memory3 = SplatMemory::new(device.clone());

    // Add test splats: pleasure at origin, pain at a random point
    let pleasure_pos = Tensor::zeros((512,), DType::F32, &device)?;
    memory3.add_splat(Splat::new(pleasure_pos, 0.15, 1.2)); // strong pleasure

    let pain_pos = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    memory3.add_splat(Splat::new(pain_pos, 0.15, -0.8)); // pain

    println!("    Splats: {} (1 pleasure + 1 pain)", memory3.len());

    let field3 = ContinuousField::load("data/universe_domain.safetensors", &device)?;
    let goal_pos3 = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let runner3 = RidgeRunner::new(field3, memory3, goal_pos3)
        .with_dt(0.01)
        .with_viscosity(0.5);

    let start3 = Tensor::randn(0.0f32, 1.0, (512,), &device)?;
    let particle3 = QueryParticle::new(start3)?;
    println!("    Start position norm: {:.6}", particle3.pos_norm()?);

    let final_p = runner3.run_with_memory(particle3, 150)?;

    println!(
        "\n    Splat force test complete. Final position norm: {:.4}",
        final_p.pos_norm()?
    );

    // Test asymmetric decay
    println!("\n[6] Testing asymmetric decay...");
    let mut decay_mem = SplatMemory::new(device.clone());
    let pos_a = Tensor::zeros((512,), DType::F32, &device)?;
    let pos_b = Tensor::zeros((512,), DType::F32, &device)?;
    decay_mem.add_splat(Splat::new(pos_a, 0.15, 1.0)); // pleasure α=1.0
    decay_mem.add_splat(Splat::new(pos_b, 0.15, -1.0)); // pain α=-1.0

    for i in 0..10 {
        decay_mem.decay_step(0.9);
        if i % 3 == 0 {
            println!("    decay step {:>2}", i);
        }
    }
    println!("    (Pain decays slower than pleasure — asymmetric scar tissue)");

    println!("\n[✓] Day 3 splat memory complete. Ready for Day 4: steering hook.");

    Ok(())
}
