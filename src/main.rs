//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Orchestrator: parse CLI args, init hardware, then delegate to focused modules:
//!   - `loader`     — Phase 1: model + tokenizer
//!   - `field`      — Phase 2: Diderot embedding field
//!   - `niodoo`     — Phase 3: physics steering engine
//!   - `generation` — Phase 4: steered token loop
//!   - `session`    — Phase 5+6: splat scars, museum, summary

mod config;
mod dream;
mod field;
mod generation;
mod gpu;
mod llama;
mod loader;
mod logger;
mod memory;
mod model;
mod niodoo;
mod qwen35;
mod ridge;
mod session;
mod splat;
mod tui;
mod viz;
mod viz_metal;

use anyhow::Result;
use config::Config;
use field::ContinuousField;
use logger::{SessionConfig, SessionLogger};
use memory::SplatMemory;
use niodoo::NiodooEngine;
use std::path::Path;
use viz::VizCollector;

fn main() -> Result<()> {
    println!("=== SplatRAG v1 -- Hydrodynamic Swarm ===\n");

    // ── Config ────────────────────────────────────────────────────────────────
    let cfg = Config::load(Path::new("config.toml")).unwrap_or_else(|e| {
        eprintln!("    [CONFIG] {}, using defaults", e);
        Config::default()
    });

    // ── CLI args ──────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let clear_memory  = args.iter().any(|a| a == "--clear-memory");
    let test_mode     = args.iter().any(|a| a == "--test");
    let viz_enabled   = args.iter().any(|a| a == "--viz");
    let chat_mode     = args.iter().any(|a| a == "--chat");
    let cli_prompt    = args.iter().position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_model     = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());
    let max_tokens: usize = args.iter().position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
        .unwrap_or(cfg.generation.max_tokens);

    // ── Device ────────────────────────────────────────────────────────────────
    let device = match candle_core::Device::new_metal(0) {
        Ok(d) => { println!("[*] Using Metal GPU"); d }
        Err(_) => { println!("[*] Metal not available, using CPU"); candle_core::Device::Cpu }
    };

    // ── Phase 1: Model + Tokenizer ────────────────────────────────────────────
    let (mut model, tokenizer, arch) = loader::load(&cfg, &device)?;

    // ── Phase 2: Diderot Field ────────────────────────────────────────────────
    println!("\n--- Phase 2: Building Diderot Field ---");
    let field = ContinuousField::from_embeddings(model.token_embeddings(), &device)?;
    let dim = field.dim;

    // ── Phase 3: Niodoo Engine ────────────────────────────────────────────────
    println!("\n--- Phase 3: Niodoo Steering Engine ---");
    let memory = SplatMemory::new(device.clone());
    let backend = gpu::select_backend();
    let mut engine = NiodooEngine::new(
        field, memory, backend,
        cfg.physics.dt, cfg.physics.viscosity_scale, cfg.physics.force_cap,
    );
    if cfg.physics.gradient_topk > 0 {
        engine.set_gradient_topk(cfg.physics.gradient_topk);
        println!("    Engine ready (backend: {}, gradient Top-K: {})",
            engine.backend_name(), cfg.physics.gradient_topk);
    } else {
        println!("    Engine ready (backend: {}, exact gradient)", engine.backend_name());
    }

    let splat_file = Path::new("data/splat_memory.safetensors");
    if clear_memory && splat_file.exists() {
        std::fs::remove_file(splat_file)?;
        println!("    Cleared splat memory (--clear-memory)");
    }
    let loaded_count = engine.memory_mut().load(splat_file)?;
    if loaded_count == 0 && !clear_memory {
        println!("    No existing splat memory found (first run)");
    }

    // TUI chat mode
    if chat_mode {
        return tui::run_chat(
            &mut model, &tokenizer, &mut engine, &device,
            dim, max_tokens, &cfg, &arch,
        );
    }

    // ── Session logger ────────────────────────────────────────────────────────
    let model_variant = cli_model.as_deref().unwrap_or("default");
    let prompt = cli_prompt
        .as_deref()
        .unwrap_or(cfg.generation.default_prompt.as_str());

    let test_label = format!(
        "{}_v3-forcecap{}_T{}_s{}_a{}_d{}",
        model_variant,
        cfg.physics.force_cap as i32,
        cfg.generation.temperature,
        cfg.physics.splat_sigma as i32,
        cfg.physics.splat_alpha as i32,
        cfg.physics.min_splat_dist as i32,
    );
    let mut logger = SessionLogger::new(&test_label, model_variant)?;
    logger.log_config(SessionConfig {
        prompt: prompt.to_string(),
        dt: cfg.physics.dt,
        viscosity: cfg.physics.viscosity_scale,
        kernel_sigma: engine.field_kernel_sigma(),
        embedding_dim: dim,
        field_points: engine.field_n_points(),
        model: cfg.models.model_path.clone(),
        model_variant: model_variant.to_string(),
        backend: engine.backend_name().to_string(),
        splat_sigma: cfg.physics.splat_sigma,
        splat_alpha: cfg.physics.splat_alpha,
        force_cap: cfg.physics.force_cap,
        temperature: cfg.generation.temperature as f32,
        min_splat_dist: cfg.physics.min_splat_dist,
    })?;

    // ── Phase 4: Generation ───────────────────────────────────────────────────
    let mut viz_collector: Option<VizCollector> = if viz_enabled {
        match VizCollector::new(engine.field_positions(), &candle_core::Tensor::zeros(
            (dim,), candle_core::DType::F32, &device)?,
            prompt, dim) {
            Ok(c) => Some(c),
            Err(e) => { eprintln!("    [VIZ] Failed to init collector: {}", e); None }
        }
    } else {
        None
    };

    let result = generation::run(
        &mut model, &tokenizer, &mut engine, &device,
        prompt, &arch, model_variant, max_tokens,
        &cfg, &mut logger, &mut viz_collector,
    )?;

    // ── Phases 5+6: Splat scars, museum, summary ──────────────────────────────
    session::finish(
        &result, &mut engine, &mut logger, &device,
        model_variant, prompt, &cfg,
        splat_file, test_mode, viz_collector,
    )?;

    Ok(())
}
