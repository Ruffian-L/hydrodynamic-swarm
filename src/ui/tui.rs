//! Chat-style TUI for one-shot prompt demonstration.
//!
//! Usage: `cargo run -- --chat`
//! Shows a styled prompt, takes user input, runs physics-steered generation
//! with live token streaming, then exits.

use anyhow::Result;
use candle_core::Tensor;
use crate::model::llama::ModelWeights;
use std::io::{self, Write};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::memory::dream::micro_dream;
use crate::physics::niodoo::NiodooEngine;
use crate::memory::splat::Splat;

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
    cfg: &Config,
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

    // Generation loop -- matches main.rs pattern:
    // Use prefill logits for step 0, call forward at END of each step for next iteration.
    let mut raw_logits = prefill_logits;
    let mut index_pos = prompt_ids.len();
    let mut generated_tokens: Vec<u32> = Vec::new();
    #[allow(unused_assignments)]
    let mut last_steered_pos: Option<Tensor> = None;
    #[allow(clippy::explicit_counter_loop)]
    for step in 0..max_tokens {

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

            let should_dream = (step % cfg.micro_dream.fixed_interval == 0)
                || (entropy > cfg.micro_dream.entropy_threshold && step % cfg.micro_dream.adaptive_interval == 0);
            if should_dream {
                let dream_steps = if entropy > 4.0 {
                    4
                } else if entropy > 3.0 {
                    3
                } else {
                    2
                };
                let blend = if entropy > cfg.micro_dream.entropy_threshold {
                    cfg.micro_dream.blend_high_entropy
                } else {
                    cfg.micro_dream.blend_normal
                };
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
        let temperature: f64 = cfg.generation.temperature;
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

        if step > 5 && delta_norm > cfg.physics.splat_delta_threshold {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine.memory().has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
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
        if cfg.generation.eos_token_ids.contains(&next_token) {
            break;
        }

        // Feed next token to get logits for the next step
        let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        raw_logits = llama.forward(&next_input, index_pos)?;
        index_pos += 1;
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
