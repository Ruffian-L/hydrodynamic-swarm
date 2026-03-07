//! Physics-Steered Token Generation (Phase 4)
//!
//! This module owns the entire generation loop:
//!   - Chat template encoding (Llama / Qwen3.5 ChatML)
//!   - Prefill + goal attractor extraction
//!   - Per-step: steering → micro-dream → sampling → repetition penalty
//!   - Live token stream (`logs/live.txt`)
//!   - EOS detection
//!   - Full decoded output
//!
//! Returns a `GenerationResult` consumed by `session::finish`.

use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::Rng;
use std::io::Write;
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::dream::micro_dream;
use crate::logger::{SessionLogger, StepEntry};
use crate::model::ModelBackend;
use crate::niodoo::{NiodooEngine, SteerResult};
use crate::session::GenerationResult;
use crate::splat::Splat;
use crate::viz::VizCollector;

/// Run the full physics-steered generation loop.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &Device,
    prompt: &str,
    arch: &str,
    model_variant: &str,
    max_tokens: usize,
    cfg: &Config,
    logger: &mut SessionLogger,
    viz_collector: &mut Option<VizCollector>,
) -> Result<GenerationResult> {
    println!("\n--- Phase 4: Physics-Steered Generation ---");
    println!("    Prompt: \"{}\"", prompt);

    let dim = engine.field_dim();

    // ── Chat template encoding ────────────────────────────────────────────────
    let prompt_ids: Vec<u32> = encode_prompt(tokenizer, prompt, arch)?;
    println!("    Prompt tokens: {} IDs", prompt_ids.len());

    // ── Prefill ───────────────────────────────────────────────────────────────
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;
    println!("    Prefilling {} prompt tokens...", prompt_ids.len());

    let (prefill_logits, prefill_hidden) = if cfg.physics.steer_hidden {
        let (logits, hidden) = model.forward_with_hidden(&prompt_tensor, 0)?;
        (logits, Some(hidden))
    } else {
        let logits = model.forward(&prompt_tensor, 0)?;
        (logits, None)
    };
    let mut index_pos = prompt_ids.len();

    // ── Goal attractor ────────────────────────────────────────────────────────
    let goal_pos = if let Some(ref hidden) = prefill_hidden {
        let h = hidden.squeeze(0)?;
        println!(
            "    Goal attractor: from hidden state (D={}, steer_hidden=true)",
            h.dim(0)?
        );
        h
    } else {
        let g = if prefill_logits.dim(1)? >= dim {
            prefill_logits.narrow(1, 0, dim)?.squeeze(0)?
        } else {
            prefill_logits.squeeze(0)?
        };
        println!("    Goal attractor: from logit space (steer_hidden=false)");
        g
    };
    let goal_norm: f32 = goal_pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!("    Goal attractor norm: {:.4}", goal_norm);

    // ── Debug: top-5 prefill tokens ───────────────────────────────────────────
    {
        let logits_1d = if prefill_logits.dims().len() > 1 {
            prefill_logits.squeeze(0)?
        } else {
            prefill_logits.clone()
        };
        let logits_vec: Vec<f32> = logits_1d.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> =
            logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprint!("    [DBG] Top-5 after prefill:");
        for (id, val) in indexed[..5].iter() {
            let tok = tokenizer.decode(&[*id as u32], false).unwrap_or_default();
            eprint!(" {}:{:.2}(\"{}\")", id, val, tok.trim());
        }
        eprintln!(" | min={:.2} max={:.2}", indexed.last().unwrap().1, indexed[0].1);
    }

    // ── Viz collector ─────────────────────────────────────────────────────────
    if viz_collector.is_none() {
        // already initialised (or not) by caller; nothing to do
    }

    // ── Generation state ──────────────────────────────────────────────────────
    let mut raw_logits = prefill_logits;
    let mut raw_hidden: Option<Tensor> = prefill_hidden;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut last_steered_pos: Option<Tensor> = None;

    // Live-stream file (tail -f logs/live.txt in another terminal)
    let live_path = std::path::Path::new("logs/live.txt");
    let mut live_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(live_path)?;
    writeln!(live_file, "\n=== [{}] \"{}\" ===", model_variant, prompt)?;
    live_file.flush()?;

    println!("\n    === Generation ({} tokens, physics-steered) ===\n", max_tokens);

    #[allow(clippy::explicit_counter_loop)]
    for step in 0..max_tokens {
        // ── Steering input: hidden state (preferred) or logit slice ───────────
        let (steer_input, is_hidden_steer) = if cfg.physics.steer_hidden {
            if let Some(ref h) = raw_hidden {
                (h.clone(), true)
            } else {
                let s = if raw_logits.dim(1)? >= dim {
                    raw_logits.narrow(1, 0, dim)?
                } else {
                    raw_logits.clone()
                };
                (s, false)
            }
        } else {
            let s = if raw_logits.dim(1)? >= dim {
                raw_logits.narrow(1, 0, dim)?
            } else {
                raw_logits.clone()
            };
            (s, false)
        };

        let SteerResult {
            steered: steered_slice,
            grad_mag,
            splat_mag,
            goal_mag,
            pos_norm: step_pos_norm,
            raw_grad_mag,
            cos_sim_goal,
        } = engine.steer(&steer_input, &goal_pos, step)?;
        last_steered_pos = Some(steered_slice.clone());

        // ── Micro-dream consolidation ─────────────────────────────────────────
        let steered_slice = if step > 10 {
            let raw_probs_slice = candle_nn::ops::softmax(&raw_logits, 1)?;
            let raw_probs_flat: Vec<f32> = raw_probs_slice.squeeze(0)?.to_vec1()?;
            let sample_n = raw_probs_flat.len().min(1000);
            let entropy: f32 = raw_probs_flat[..sample_n]
                .iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum::<f32>()
                .max(0.0);

            let should_dream = entropy > cfg.micro_dream.entropy_threshold
                || step % cfg.micro_dream.fixed_interval == 0;

            if should_dream {
                let dream_steps =
                    if entropy > 4.0 { 4 } else if entropy > 3.0 { 3 } else { 2 };
                let blend = if entropy > cfg.micro_dream.entropy_threshold {
                    cfg.micro_dream.blend_high_entropy
                } else {
                    cfg.micro_dream.blend_normal
                };
                let result =
                    micro_dream(engine, &steered_slice, &goal_pos, step, dream_steps, blend)?;
                if step <= 15 || step % 10 == 0 {
                    println!(
                        "    [MICRO-DREAM] step {} | correction: {:.2} | entropy: {:.2} | depth: {}{}",
                        step,
                        result.correction_norm,
                        entropy,
                        dream_steps,
                        if result.reflection_triggered { " ** HYDRAULIC JUMP **" } else { "" }
                    );
                }
                if result.reflection_triggered {
                    println!(
                        "    [TOPO-COT] step {} | *recalibrating latent path* (correction: {:.2})",
                        step, result.correction_norm
                    );
                }
                result.consolidated
            } else {
                steered_slice
            }
        } else {
            steered_slice
        };

        // ── Reconstruct full vocab logits ─────────────────────────────────────
        let steered_logits = if is_hidden_steer {
            model.project_to_logits(&steered_slice)?
        } else if raw_logits.dim(1)? > dim {
            let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
            Tensor::cat(&[&steered_slice, &rest], 1)?
        } else {
            steered_slice
        };

        // ── Temperature sampling ──────────────────────────────────────────────
        let scaled_logits = (&steered_logits / cfg.generation.temperature)?;
        let probs = candle_nn::ops::softmax(&scaled_logits, 1)?;
        let mut probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        // ── Repetition penalty (dynamic bigram) ───────────────────────────────
        let base_rep_penalty: f32 = 1.18;
        let recent_window = 64usize;
        let recent_toks: Vec<u32> = generated_tokens
            .iter()
            .rev()
            .take(recent_window)
            .cloned()
            .collect();
        let rep_score = if recent_toks.len() >= 4 {
            let mut bigram_count = std::collections::HashMap::new();
            for pair in recent_toks.windows(2) {
                *bigram_count.entry((pair[0], pair[1])).or_insert(0u32) += 1;
            }
            let total_bigrams = recent_toks.len() - 1;
            let repeated: u32 = bigram_count.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
            repeated as f32 / total_bigrams as f32
        } else {
            0.0
        };
        let rep_penalty = if rep_score > 0.35 { 1.35f32 } else { base_rep_penalty };
        let recent: std::collections::HashSet<u32> = recent_toks.iter().cloned().collect();
        for (i, p) in probs_vec.iter_mut().enumerate() {
            if recent.contains(&(i as u32)) {
                *p = p.powf(rep_penalty);
            }
        }
        let prob_sum: f32 = probs_vec.iter().sum();
        if prob_sum > 0.0 {
            for p in probs_vec.iter_mut() {
                *p /= prob_sum;
            }
        }

        // ── Sample next token ─────────────────────────────────────────────────
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

        // ── Steering delta + online splat ─────────────────────────────────────
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        if cfg.physics.enable_online_splats
            && step > 5
            && delta_norm > cfg.physics.splat_delta_threshold
        {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine
                    .memory()
                    .has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
                if !too_close {
                    let splat_alpha = (delta_norm / 10.0).clamp(1.0, 5.0);
                    engine.memory_mut().add_splat(Splat::with_scale(
                        current_pos,
                        cfg.physics.splat_sigma,
                        splat_alpha,
                        delta_norm,
                    ));
                }
            }
        }

        generated_tokens.push(next_token);

        // ── Decode + stream ───────────────────────────────────────────────────
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));

        // Viz snapshot
        if let Some(ref mut collector) = viz_collector {
            let neighbors = if step % 5 == 0 {
                let mut prob_indices: Vec<(u32, f32)> = probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as u32, p))
                    .collect();
                if prob_indices.len() > 5 {
                    prob_indices.select_nth_unstable_by(4, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    prob_indices.truncate(5);
                }
                prob_indices
                    .iter()
                    .take(5)
                    .map(|&(tid, prob)| {
                        let text = tokenizer
                            .decode(&[tid], false)
                            .unwrap_or_else(|_| format!("[{}]", tid));
                        (tid, text, prob)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let _ = collector.snapshot(step, next_token, &decoded, &steered_logits, delta_norm, neighbors);
        }

        print!("{}", decoded);
        std::io::stdout().flush().ok();
        write!(live_file, "{}", decoded).ok();
        live_file.flush().ok();

        if step > 0 && step % 50 == 0 {
            println!("  [{}/{}]", step, max_tokens);
        }

        // ── Per-step telemetry ────────────────────────────────────────────────
        let step_entropy: f32 = {
            let sample_n = probs_vec.len().min(1000);
            probs_vec[..sample_n]
                .iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum::<f32>()
                .max(0.0)
        };
        let residual_norm: f32 = steered_logits.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        logger.log_step(StepEntry {
            step,
            token_id: next_token,
            token_text: decoded,
            steering_delta: delta_norm,
            residual_norm,
            grad_force_mag: grad_mag,
            splat_force_mag: splat_mag,
            goal_force_mag: goal_mag,
            pos_norm: step_pos_norm,
            raw_grad_mag,
            cos_sim_goal,
            entropy: step_entropy,
        })?;

        // ── EOS check ─────────────────────────────────────────────────────────
        if cfg.generation.eos_token_ids.contains(&next_token) {
            println!("    → EOS at step {}", step);
            break;
        }

        // ── Feed next token ───────────────────────────────────────────────────
        let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        if cfg.physics.steer_hidden {
            let (logits, hidden) = model.forward_with_hidden(&next_input, index_pos)?;
            raw_logits = logits;
            raw_hidden = Some(hidden);
        } else {
            raw_logits = model.forward(&next_input, index_pos)?;
            raw_hidden = None;
        }
        index_pos += 1;
    }

    // ── Full decode ───────────────────────────────────────────────────────────
    println!("\n    === Full Decoded Output ===\n");
    let full_text = tokenizer
        .decode(&generated_tokens, true)
        .unwrap_or_else(|_| "[decode error]".to_string());
    println!("    {}", full_text);

    Ok(GenerationResult {
        tokens: generated_tokens,
        full_text,
        goal_norm,
        prompt_ids,
        last_steered_pos,
    })
}

/// Encode a prompt with the correct chat template for the given architecture.
fn encode_prompt(tokenizer: &Tokenizer, prompt: &str, arch: &str) -> Result<Vec<u32>> {
    if arch == "qwen35" {
        // Qwen3.5-Instruct ChatML:
        //   <|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n
        const IM_START: u32 = 248045;
        const IM_END: u32 = 248046;

        let encode = |s: &str| -> Result<Vec<u32>> {
            tokenizer
                .encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("encode: {}", e))
        };

        let mut ids = Vec::new();
        ids.push(IM_START);
        ids.extend(encode("user")?);
        ids.extend(encode("\n")?);
        ids.extend(encode(prompt)?);
        ids.extend(encode("\n")?);
        ids.push(IM_END);
        ids.extend(encode("\n")?);
        ids.push(IM_START);
        ids.extend(encode("assistant")?);
        ids.extend(encode("\n")?);
        Ok(ids)
    } else {
        // Llama / default: BOS + raw prompt
        tokenizer
            .encode(prompt, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("encode prompt: {}", e))
    }
}
