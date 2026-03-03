//! Chat-style TUI for one-shot prompt demonstration.
//!
//! Usage: `cargo run -- --chat`
//! Shows a styled prompt, takes user input, runs physics-steered generation
//! with live token streaming, then exits.

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::io::{self, Write};
use tokenizers::Tokenizer;

use crate::dream::micro_dream;
use crate::niodoo::NiodooEngine;
use crate::splat::Splat;

// ANSI color codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GOLD: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const GRAY: &str = "\x1b[90m";

/// Run the chat TUI -- one-shot mode.
pub fn run_chat(
    llama: &mut ModelWeights,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
) -> Result<()> {
    // Clear screen and show banner
    print!("\x1b[2J\x1b[H");
    println!();
    println!("  {BOLD}{CYAN}============================================{RESET}");
    println!("  {BOLD}{CYAN}   HYDRODYNAMIC SWARM  --  SplatRAG v1.1{RESET}");
    println!("  {BOLD}{CYAN}============================================{RESET}");
    println!("  {DIM}{GRAY}Physics-steered LLM generation engine{RESET}");
    println!("  {DIM}{GRAY}Llama 3.1 8B + Niodoo field dynamics{RESET}");
    println!();
    println!(
        "  {DIM}{GRAY}Splats in memory: {}{RESET}",
        engine.memory().len()
    );
    println!("  {DIM}{GRAY}Max tokens: {}{RESET}", max_tokens);
    println!();

    // Prompt input
    print!("  {BOLD}{GREEN}>{RESET} ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let prompt = input.trim();

    if prompt.is_empty() {
        println!("  {DIM}{GRAY}(empty prompt, exiting){RESET}");
        return Ok(());
    }

    println!();
    println!("  {DIM}{GRAY}--- Encoding prompt ---{RESET}");

    // Encode prompt
    let encoded = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();

    println!("  {DIM}{GRAY}{} tokens{RESET}", prompt_ids.len());

    // Prefill
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;
    let prefill_logits = llama.forward(&prompt_tensor, 0)?;

    // Goal attractor from prefill
    let goal_pos = if prefill_logits.dim(1)? >= dim {
        prefill_logits.narrow(1, 0, dim)?.squeeze(0)?
    } else {
        prefill_logits.squeeze(0)?
    };

    println!("  {DIM}{GRAY}--- Generating ---{RESET}");
    println!();
    print!("  {BOLD}{GOLD}");
    io::stdout().flush()?;

    // Generation loop
    let mut generated_tokens: Vec<u32> = Vec::new();
    #[allow(unused_assignments)]
    let mut last_steered_pos: Option<Tensor> = None;
    for (step, index_pos) in (0..max_tokens).zip(prompt_ids.len()..) {
        // Build input tensor
        let input_ids = if step == 0 {
            // First step: use last prompt token
            let last_id = *prompt_ids.last().unwrap_or(&1);
            Tensor::new(&[last_id], device)?.unsqueeze(0)?
        } else {
            let last_token = *generated_tokens.last().unwrap_or(&1);
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };

        // Forward pass
        let raw_logits = llama.forward(&input_ids, index_pos)?;

        // Physics steering
        let raw_slice = raw_logits.narrow(1, 0, dim)?;
        let steer_result = engine.steer(&raw_slice, &goal_pos, step)?;
        last_steered_pos = Some(steer_result.steered.clone());
        let steered_slice = steer_result.steered;

        // Micro-dream (adaptive)
        let steered_slice = if step > 3 {
            let raw_probs_temp = candle_nn::ops::softmax(&raw_logits, 1)?;
            let raw_probs_flat: Vec<f32> = raw_probs_temp.squeeze(0)?.to_vec1()?;
            let sample_n = raw_probs_flat.len().min(dim);
            let entropy: f32 = raw_probs_flat[..sample_n]
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|p| -p * p.ln())
                .sum();

            let should_dream = (step % 25 == 0) || (entropy > 3.0 && step % 8 == 0);
            if should_dream {
                let dream_steps = if entropy > 4.0 {
                    4
                } else if entropy > 3.0 {
                    3
                } else {
                    2
                };
                let blend = if entropy > 3.0 { 0.15 } else { 0.10 };
                let result =
                    micro_dream(engine, &steered_slice, &goal_pos, step, dream_steps, blend)?;
                result.consolidated
            } else {
                steered_slice
            }
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

        // Temperature sampling
        let temperature: f64 = 0.9;
        let scaled_logits = (&steered_logits / temperature)?;
        let probs = candle_nn::ops::softmax(&scaled_logits, 1)?;
        let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        let mut rng = rand::thread_rng();
        use rand::Rng;
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

        // Online splat creation
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        if step > 5 && delta_norm > 12.0 {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine.memory().has_nearby(&current_pos, 100.0)?;
                if !too_close {
                    let splat_sigma = if delta_norm > 30.0 {
                        70.0
                    } else if delta_norm > 20.0 {
                        50.0
                    } else {
                        35.0
                    };
                    let splat_alpha = (delta_norm / 10.0).clamp(1.0, 5.0);
                    engine.memory_mut().add_splat(Splat::new(
                        current_pos,
                        splat_sigma,
                        splat_alpha,
                    ));
                }
            }
        }

        generated_tokens.push(next_token);

        // Stream token
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));
        print!("{}", decoded);
        io::stdout().flush()?;

        // Stop on EOS
        if next_token == 128001 || next_token == 128009 {
            break;
        }
    }

    // Clean up
    print!("{RESET}");
    println!();
    println!();
    println!(
        "  {DIM}{GRAY}--- Done: {} tokens generated ---{RESET}",
        generated_tokens.len()
    );
    println!(
        "  {DIM}{GRAY}Splats in memory: {}{RESET}",
        engine.memory().len()
    );
    println!();

    Ok(())
}
