//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Full Llama 3.1 + Niodoo physics steering with real tokenization.
//! Type a prompt → physics steers generation → decoded text output.

mod dream;
mod field;
mod logger;
mod memory;
mod niodoo;
mod ridge;
mod splat;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use dream::{DreamEngine, micro_dream};
use field::ContinuousField;
use logger::{SessionConfig, SessionLogger, SessionSummary, StepEntry};
use memory::SplatMemory;
use niodoo::NiodooEngine;
use splat::Splat;
use rand::Rng;
use std::io::BufReader;
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== SplatRAG v1 -- Hydrodynamic Swarm ===\n");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let clear_memory = args.iter().any(|a| a == "--clear-memory");
    let cli_prompt = args.iter().position(|a| a == "--prompt").map(|i| args[i + 1].clone());
    let cli_model = args.iter().position(|a| a == "--model").map(|i| args[i + 1].clone());

    // Use Metal GPU if available
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
    // Phase 1: Load real 4096d embeddings (Diderot field)
    // =========================================================
    println!("\n--- Phase 1: Loading Real Embeddings ---");
    let field = ContinuousField::load_real("data/universe_domain.safetensors", &device)?;
    let dim = field.dim;

    // =========================================================
    // Phase 2: Load Llama 3.1 GGUF + Tokenizer
    // =========================================================
    println!("\n--- Phase 2: Loading Llama 3.1 + Tokenizer ---");

    // Find GGUF model
    let llama_path = find_file(
        "data/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "/Users/j/Desktop/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    );
    println!("    Model: {}", llama_path);

    let mut file = std::fs::File::open(&llama_path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;
    let mut llama = ModelWeights::from_gguf(ct, &mut reader, &device)?;
    println!("    ✓ Llama 3.1 loaded");

    // Find tokenizer
    let tokenizer_path = find_file(
        "data/tokenizer.json",
        "/Users/j/Desktop/models/tokenizer.json",
    );
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    println!("    ✓ Tokenizer loaded ({})", tokenizer_path);

    // =========================================================
    // Phase 3: Niodoo Engine
    // =========================================================
    println!("\n--- Phase 3: Niodoo Steering Engine ---");
    let memory = SplatMemory::new(device.clone());
    let mut engine = NiodooEngine::new(field, memory);
    println!("    Engine ready");

    // Load persistent splat memory if it exists
    let splat_file = Path::new("data/splat_memory.safetensors");
    if clear_memory && splat_file.exists() {
        std::fs::remove_file(splat_file)?;
        println!("    Cleared splat memory (--clear-memory)");
    }
    let loaded_count = engine.memory_mut().load(splat_file)?;
    if loaded_count == 0 && !clear_memory {
        println!("    No existing splat memory found (first run)");
    }

    // Initialize telemetry logger
    let model_variant = cli_model.as_deref().unwrap_or("unsloth");
    let prompt = cli_prompt.as_deref().unwrap_or("Explain the Physics of Friendship in one paragraph.");
    let test_label = format!("{}_v3-forcecap80_T0.9_s150_a2_d100", model_variant);
    let mut logger = SessionLogger::new(&test_label, model_variant)?;
    logger.log_config(SessionConfig {
        prompt: prompt.to_string(),
        dt: 0.08,
        viscosity: 0.6,
        kernel_sigma: 2.24,
        embedding_dim: dim,
        field_points: 128256,
        model: "Llama-3.1-8B-Instruct-Q5_K_M".to_string(),
        model_variant: model_variant.to_string(),
        backend: if device.is_metal() {
            "Metal GPU"
        } else {
            "CPU"
        }
        .to_string(),
        splat_sigma: 150.0,
        splat_alpha: 2.0,
        force_cap: 80.0,
        temperature: 0.9,
        min_splat_dist: 100.0,
    })?;

    // =========================================================
    // Phase 4: Real Prompt -> Physics-Steered Generation
    // =========================================================
    println!("\n--- Phase 4: Physics-Steered Generation ---");
    println!("    Prompt: \"{}\"", prompt);

    // Encode prompt
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    println!("    Prompt tokens: {} IDs", prompt_ids.len());

    // Feed prompt through Llama (prefill) to get context-aware goal attractor
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("    Prefilling {} prompt tokens...", prompt_ids.len());
    let prefill_logits = llama.forward(&prompt_tensor, 0)?;
    let mut index_pos = prompt_ids.len();

    // Goal attractor: first D logits from the prefill pass.
    // This is the model's context-aware response to the prompt (in logit space),
    // so Niodoo will steer generation toward what the model "naturally" wants to say.
    // Much more meaningful than raw vocab mean which cancels to ~zero.
    let goal_pos = if prefill_logits.dim(1)? >= dim {
        prefill_logits.narrow(1, 0, dim)?.squeeze(0)? // (dim,)
    } else {
        prefill_logits.squeeze(0)?
    };
    let goal_norm: f32 = goal_pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!(
        "    Goal attractor norm: {:.4} (context-aware, from prefill logits)",
        goal_norm
    );

    // Now start generating from prefill logits
    let mut raw_logits = prefill_logits;

    // Collect generated tokens
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Track last steered position for splat creation
    let mut last_steered_pos: Option<Tensor> = None;

    println!("\n    === Generation (physics-steered) ===\n");

    for step in 0..60 {
        // Steer the logits with Niodoo physics
        let logit_slice = if raw_logits.dim(1)? >= dim {
            raw_logits.narrow(1, 0, dim)?
        } else {
            raw_logits.clone()
        };

        let steered_slice = engine.steer(&logit_slice, &goal_pos, step)?;
        last_steered_pos = Some(steered_slice.clone());

        // Micro-dream consolidation: forward+backward physics burst on the 4096d slice
        // Triggers every 8 steps after warmup
        let steered_slice = if step > 5 && (step % 8 == 0) {
            let result = micro_dream(&engine, &steered_slice, &goal_pos, step, 2, 0.10)?;
            if step <= 15 || step % 10 == 0 {
                println!(
                    "    [MICRO-DREAM] step {} | correction_norm: {:.2}{}",
                    step, result.correction_norm,
                    if result.reflection_triggered { " ** HYDRAULIC JUMP **" } else { "" }
                );
            }
            if result.reflection_triggered {
                // TopoCoT: inject reflection marker into token stream
                println!(
                    "    [TOPO-COT] step {} | *recalibrating latent path* (correction: {:.2})",
                    step, result.correction_norm
                );
            }
            result.consolidated
        } else {
            steered_slice
        };

        // Reconstruct full logits
        let steered_logits = if raw_logits.dim(1)? > dim {
            let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
            Tensor::cat(&[&steered_slice, &rest], 1)?
        } else {
            steered_slice
        };

        // Temperature sampling -- softmax over scaled logits, then sample
        let temperature: f64 = 0.9;
        let scaled_logits = (&steered_logits / temperature)?;
        let probs = candle_nn::ops::softmax(&scaled_logits, 1)?;
        let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;
        let mut rng = rand::thread_rng();
        let roll: f32 = rng.gen();
        let mut cumsum = 0.0f32;
        let mut next_token: u32 = 0;
        for (i, p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if roll < cumsum {
                next_token = i as u32;
                break;
            }
        }

        // Steering delta
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Online splat update -- add scar mid-generation based on steering strength
        if step > 5 && delta_norm > 12.0 {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?; // (1, D) -> (D,)

                // Diagnostic: how far is the current position from the last splat?
                let splat_force = engine.memory().query_force(&current_pos)?;
                let splat_force_norm: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                if step <= 15 || step % 10 == 0 {
                    println!(
                        "    [DIAG] step {} | splat_force_norm: {:.6e} | splats: {}",
                        step, splat_force_norm, engine.memory().len()
                    );
                }

                // Variant 2: min distance check -- prevent stacking
                let too_close = engine.memory().has_nearby(&current_pos, 100.0)?;
                if !too_close {
                    engine.memory_mut().add_splat(Splat::new(
                        current_pos,
                        150.0,
                        2.0, // pleasure -- soft pull
                    ));
                    println!(
                        "    [ONLINE] Added pleasure splat at step {} (delta {:.2}, splats: {})",
                        step, delta_norm, engine.memory().len()
                    );
                } else if step <= 15 || step % 10 == 0 {
                    println!(
                        "    [SKIP] step {} -- too close to existing splat (delta {:.2})",
                        step, delta_norm
                    );
                }
            }
        }

        generated_tokens.push(next_token);

        // Decode and print
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));

        if step < 15 || step % 10 == 0 {
            println!(
                "    step {:>2} | token {:>6} | delta: {:>7.2} | \"{}\"",
                step, next_token, delta_norm, decoded
            );
        }

        // Log every step to JSONL
        let residual_norm: f32 = steered_logits.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        logger.log_step(StepEntry {
            step,
            token_id: next_token,
            token_text: decoded,
            steering_delta: delta_norm,
            residual_norm,
        })?;

        // Stop on EOS tokens
        if next_token == 128009 || next_token == 128001 {
            println!("    → EOS at step {}", step);
            break;
        }

        // Feed next token
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        raw_logits = llama.forward(&next_input, index_pos)?;
        index_pos += 1;
    }

    // =========================================================
    // Decode full output
    // =========================================================
    println!("\n    === Full Decoded Output ===\n");
    let full_text = tokenizer
        .decode(&generated_tokens, true)
        .unwrap_or_else(|_| "[decode error]".to_string());
    println!("    {}", full_text);

    // =========================================================
    // Populate real splats from this generation
    // =========================================================
    println!("\n--- Phase 5: Splat Scar Tissue ---");
    if let Some(final_pos) = last_steered_pos {
        let pos_1d = final_pos.squeeze(0)?; // (1, D) -> (D,)
        if generated_tokens.len() > 15 {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d, 150.0, // sigma
                1.8,  // positive scar (pleasure)
            ));
            println!(
                "    ✓ Added PLEASURE splat (generation succeeded: {} tokens)",
                generated_tokens.len()
            );
        } else {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d, 150.0, // sigma
                -0.9, // negative scar (pain)
            ));
            println!(
                "    ✗ Added PAIN splat (generation too short: {} tokens)",
                generated_tokens.len()
            );
        }
        println!("    Splats in memory: {}", engine.memory().len());
    }

    // Save persistent splat memory to disk (before dream decay wipes them)
    engine.memory().save(splat_file)?;

    // =========================================================
    // Phase 6: Dream Replay
    // =========================================================
    println!("\n--- Phase 6: Dream Replay ---");
    let dream_memory = SplatMemory::new(device.clone());
    let mut dream = DreamEngine::new(dream_memory);
    let traj = Tensor::randn(0.0f32, 1.0, (20, dim), &device)?;
    dream.run(vec![traj], 0.05)?;

    // =========================================================
    // Summary
    // =========================================================
    let splat_type = if generated_tokens.len() > 15 {
        "pleasure"
    } else {
        "pain"
    };
    logger.log_summary(SessionSummary {
        prompt: prompt.to_string(),
        prompt_token_count: prompt_ids.len(),
        generated_token_count: generated_tokens.len(),
        goal_attractor_norm: goal_norm,
        splat_count_before: engine.memory().len(), // includes online splats from generation
        splat_count_after: engine.memory().len(),
        splat_type_added: splat_type.to_string(),
        decoded_output: full_text.clone(),
        delta_min: 0.0, // filled by log_summary
        delta_max: 0.0,
        delta_mean: 0.0,
    })?;

    println!("\n========================================");
    println!("  SplatRAG v1.1 -- OPERATIONAL");
    println!("========================================");
    println!("  Model:    Llama 3.1 8B Instruct (Q5_K_M)");
    println!("  Variant:  {}", model_variant);
    println!("  Prompt:   \"{}\"", prompt);
    println!("  Tokens:   {} generated", generated_tokens.len());
    println!("  Log:      {}", logger.path().display());
    println!("  Backend:  Metal GPU + Niodoo physics");
    println!("========================================");

    // Append to human-readable log
    {
        use std::io::Write;
        let readable_path = Path::new("logs/readable.txt");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(readable_path)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let secs_per_day: u64 = 86400;
        let days = now / secs_per_day;
        let day_secs = now % secs_per_day;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        // Approximate date from Unix days
        let mut y = 1970i64;
        let mut remaining = days as i64;
        loop {
            let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
            if remaining < days_in_year { break; }
            remaining -= days_in_year;
            y += 1;
        }
        let month_days = [31, if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut m = 1;
        for md in &month_days {
            if remaining < *md as i64 { break; }
            remaining -= *md as i64;
            m += 1;
        }
        let d = remaining + 1;
        writeln!(f, "=== Run: {}-{:02}-{:02} {:02}:{:02} UTC ===", y, m, d, hours, minutes)?;
        writeln!(f, "Model: {} | Tokens: {} | Splats: {}", model_variant, generated_tokens.len(), engine.memory().len())?;
        writeln!(f, "Prompt: \"{}\"", prompt)?;
        writeln!(f, "")?;
        writeln!(f, "{}", full_text)?;
        writeln!(f, "")?;
        writeln!(f, "---")?;
        writeln!(f, "")?;
    }

    Ok(())
}

/// Find a file at primary path, fallback to secondary.
fn find_file(primary: &str, fallback: &str) -> String {
    if std::path::Path::new(primary).exists() {
        primary.to_string()
    } else if std::path::Path::new(fallback).exists() {
        fallback.to_string()
    } else {
        eprintln!("ERROR: File not found at {} or {}", primary, fallback);
        std::process::exit(1);
    }
}
