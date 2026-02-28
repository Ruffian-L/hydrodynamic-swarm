//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Retrieval and generation as a single physical process:
//! a particle sliding down a continuous Diderot field,
//! guided by splat scar tissue.

mod dream;
mod field;
mod memory;
mod niodoo;
mod splat;

use anyhow::Result;
use candle_core::{Device, Tensor};
use field::ContinuousField;
use memory::SplatMemory;
use niodoo::{compute_steering_force, steer_residual, NiodooParams};

fn main() -> Result<()> {
    println!("=== SplatRAG v1 — Hydrodynamic Swarm ===\n");

    let device = Device::Cpu;

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
    let memory = SplatMemory::new(&device);
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

    println!("\n[✓] Physics pipeline working. Ready for Day 2: ridge-running loop.");

    Ok(())
}
