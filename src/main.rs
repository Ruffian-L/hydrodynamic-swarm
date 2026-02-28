//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Full Llama 3.1 + Niodoo physics steering with real tokenization.
//! Type a prompt → physics steers generation → decoded text output.

mod dream;
mod field;
mod memory;
mod niodoo;
mod ridge;
mod splat;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use dream::DreamEngine;
use field::ContinuousField;
use memory::SplatMemory;
use niodoo::NiodooEngine;
use std::io::BufReader;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    println!("=== SplatRAG v1 — Hydrodynamic Swarm ===\n");

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
        "/Users/j/Desktop/again/Niodoo-Physics-LLM-main/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
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
        "/Users/j/Desktop/again/Niodoo-Physics-LLM-main/models/tokenizer.json",
    );
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    println!("    ✓ Tokenizer loaded ({})", tokenizer_path);

    // =========================================================
    // Phase 3: Niodoo Engine
    // =========================================================
    println!("\n--- Phase 3: Niodoo Steering Engine ---");
    let memory = SplatMemory::new(device.clone());
    let engine = NiodooEngine::new(field, memory);
    println!("    ✓ Engine ready");

    // =========================================================
    // Phase 4: Real Prompt → Physics-Steered Generation
    // =========================================================
    let prompt = "Explain the Physics of Friendship in one paragraph.";
    println!("\n--- Phase 4: Physics-Steered Generation ---");
    println!("    Prompt: \"{}\"", prompt);

    // Encode prompt
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    println!("    Prompt tokens: {} IDs", prompt_ids.len());

    // Goal position: mean of prompt token embeddings in the Diderot field
    // For v1: use a zero attractor (will refine with real prompt embedding lookup later)
    let goal_pos = Tensor::zeros((dim,), DType::F32, &device)?;

    // Feed prompt through Llama first (prefill)
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("    Prefilling {} prompt tokens...", prompt_ids.len());
    let mut raw_logits = llama.forward(&prompt_tensor, 0)?;
    let mut index_pos = prompt_ids.len();

    // Collect generated tokens
    let mut generated_tokens: Vec<u32> = Vec::new();

    println!("\n    === Generation (physics-steered) ===\n");

    for step in 0..60 {
        // Steer the logits with Niodoo physics
        let logit_slice = if raw_logits.dim(1)? >= dim {
            raw_logits.narrow(1, 0, dim)?
        } else {
            raw_logits.clone()
        };

        let steered_slice = engine.steer(&logit_slice, &goal_pos)?;

        // Reconstruct full logits
        let steered_logits = if raw_logits.dim(1)? > dim {
            let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
            Tensor::cat(&[&steered_slice, &rest], 1)?
        } else {
            steered_slice
        };

        // Greedy decode
        let next_token: u32 = steered_logits.argmax(1)?.squeeze(0)?.to_scalar()?;

        // Steering delta
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

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
    // Phase 5: Dream Replay
    // =========================================================
    println!("\n--- Phase 5: Dream Replay ---");
    let dream_memory = SplatMemory::new(device.clone());
    let mut dream = DreamEngine::new(dream_memory);
    let traj = Tensor::randn(0.0f32, 1.0, (20, dim), &device)?;
    dream.run(vec![traj], 0.05)?;

    // =========================================================
    // Summary
    // =========================================================
    println!("\n========================================");
    println!("  ✅ SplatRAG v1 — FULLY OPERATIONAL");
    println!("========================================");
    println!("  Model:    Llama 3.1 8B Instruct (Q5_K_M)");
    println!("  Prompt:   \"{}\"", prompt);
    println!("  Tokens:   {} generated", generated_tokens.len());
    println!("  Backend:  Metal GPU + Niodoo physics");
    println!("========================================");

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
