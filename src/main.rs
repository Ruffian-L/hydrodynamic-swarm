//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Full Llama 3.1 + Niodoo physics steering with real tokenization.
//! Type a prompt → physics steers generation → decoded text output.

#[allow(dead_code, unused_imports, unused_variables)]
mod concourse;
mod config;
mod dream;
mod field;
mod gemma;
mod gpu;
mod llama;
mod logger;
mod memory;
mod niodoo;
mod ridge;
mod splat;
mod tui;
mod viz;
// mod viz_metal; // removed: XSS-vulnerable HTML viewer (security audit 2026-03-07)

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use config::Config;
use dream::micro_dream;
use field::ContinuousField;
use logger::{SessionConfig, SessionLogger, SessionSummary, StepEntry};
use memory::SplatMemory;
use niodoo::{NiodooEngine, SteerResult};
use rand::Rng;
use splat::Splat;
use std::io::BufReader;
use std::path::Path;
use tokenizers::Tokenizer;
use viz::VizCollector;

// ═══════════════════════════════════════════════════════════════════════════════
// Model: dispatch enum wrapping Llama and Gemma for physics steering
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified model interface for the Niodoo physics engine.
/// Both variants expose the same 4 methods used by the generation loop.
enum Model {
    Llama(llama::ModelWeights),
    Gemma(gemma::ModelWeights),
}

impl Model {
    fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(tokens, index_pos),
            Model::Gemma(m) => m.forward(tokens, index_pos),
        }
    }

    fn forward_with_hidden(
        &mut self,
        tokens: &Tensor,
        index_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Model::Llama(m) => m.forward_with_hidden(tokens, index_pos),
            Model::Gemma(m) => m.forward_with_hidden(tokens, index_pos),
        }
    }

    fn project_to_logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Model::Llama(m) => m.project_to_logits(hidden),
            Model::Gemma(m) => m.project_to_logits(hidden),
        }
    }

    fn token_embeddings(&self) -> &Tensor {
        match self {
            Model::Llama(m) => m.token_embeddings(),
            Model::Gemma(m) => m.token_embeddings(),
        }
    }

    fn variant_name(&self) -> &'static str {
        match self {
            Model::Llama(_) => "llama3.1",
            Model::Gemma(_) => "gemma27b",
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== SplatRAG v1 -- Hydrodynamic Swarm ===\n");

    // Load configuration (falls back to defaults if no config.toml)
    let cfg = Config::load(Path::new("config.toml")).unwrap_or_else(|e| {
        eprintln!("    [CONFIG] {}, using defaults", e);
        Config::default()
    });

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let clear_memory = args.iter().any(|a| a == "--clear-memory");
    let cli_prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());
    let max_tokens: usize = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
        .unwrap_or(cfg.generation.max_tokens)
        .min(50_000); // security: cap to prevent DoS-level resource exhaustion
    let viz_enabled = args.iter().any(|a| a == "--viz");
    let chat_mode = args.iter().any(|a| a == "--chat");

    // Force full NVIDIA CUDA for all Candle ops + physics (post-upgrade)
    let device = Device::new_cuda(0).expect("CUDA GPU required - nvidia-smi shows GB10. Fix: export CUDA_VISIBLE_DEVICES=0 && sudo apt install nvidia-cuda-toolkit");
    println!("[*] Using CUDA GPU (forced - all tensors/physics on NVIDIA)");

    // =========================================================
    // Phase 1: Load Model (Llama or Gemma) + Tokenizer
    // =========================================================
    let use_gemma = cli_model.as_deref() == Some("gemma27b") || cli_model.as_deref() == Some("gemma");

    let (mut model, model_path, tokenizer_path) = if use_gemma {
        println!("\n--- Phase 1: Loading Gemma 3 27B + Tokenizer ---");
        let gemma_path = find_file(
            "data/google/gemma-3-27b-it-Q8_0.gguf",
            "data/google/gemma-3-27b-it-q8_0.gguf",
        )?;
        println!("    Model: {}", gemma_path);

        let mut file = std::fs::File::open(&gemma_path)?;
        let mut reader = BufReader::new(&mut file);
        let ct = gguf_file::Content::read(&mut reader)?;
        let weights = gemma::ModelWeights::from_gguf(ct, &mut reader, &device)?;
        println!("    Gemma 3 27B loaded (hidden_dim={})", weights.hidden_dim);

        let tok_path = find_file(
            "data/google/tokenizer.json",
            "data/google/tokenizer_gemma.json",
        )?;

        (Model::Gemma(weights), gemma_path, tok_path)
    } else {
        println!("\n--- Phase 1: Loading Llama 3.1 + Tokenizer ---");
        let llama_path = find_file(
            "data/bartowski/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            "data/llama3.1/Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        )?;
        println!("    Model: {}", llama_path);

        let mut file = std::fs::File::open(&llama_path)?;
        let mut reader = BufReader::new(&mut file);
        let ct = gguf_file::Content::read(&mut reader)?;
        let weights = llama::ModelWeights::from_gguf(ct, &mut reader, &device)?;
        println!("    Llama 3.1 loaded");

        let tok_path = find_file(
            "data/bartowski/tokenizer_official.json",
            "data/bartowski/tokenizer_nous.json",
        )?;

        (Model::Llama(weights), llama_path, tok_path)
    };

    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    println!("    Tokenizer loaded ({})", tokenizer_path);

    // =========================================================
    // Phase 2: Build live Diderot field from model embeddings
    // =========================================================
    println!("\n--- Phase 2: Building Diderot Field ---");
    let field = ContinuousField::from_embeddings(model.token_embeddings(), &device)?;
    let dim = field.dim;

    // =========================================================
    // Phase 3: Niodoo Engine
    // =========================================================
    println!("\n--- Phase 3: Niodoo Steering Engine ---");
    let memory = SplatMemory::new(device.clone());
    let backend = gpu::select_backend();
    let mut engine = NiodooEngine::new(
        field,
        memory,
        backend,
        cfg.physics.dt,
        cfg.physics.viscosity_scale,
        cfg.physics.force_cap,
    );
    if cfg.physics.gradient_topk > 0 {
        engine.set_gradient_topk(cfg.physics.gradient_topk);
        println!(
            "    Engine ready (backend: {}, gradient Top-K: {})",
            engine.backend_name(),
            cfg.physics.gradient_topk
        );
    } else {
        println!(
            "    Engine ready (backend: {}, exact gradient)",
            engine.backend_name()
        );
    }

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

    // =========================================================
    // Chat TUI mode (--chat)
    // =========================================================
    if chat_mode {
        // TUI is stub — extract inner Llama for now
        if let Model::Llama(ref mut llama) = model {
            return tui::run_chat(
                llama,
                &tokenizer,
                &mut engine,
                &device,
                dim,
                max_tokens,
                &cfg,
            );
        } else {
            eprintln!("    [TUI] Chat mode not yet supported for Gemma — use --prompt instead");
            return Ok(());
        }
    }

    // Initialize telemetry logger
    let model_variant = model.variant_name();
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
        model: model_path.clone(),
        model_variant: model_variant.to_string(),
        backend: engine.backend_name().to_string(),
        splat_sigma: cfg.physics.splat_sigma,
        splat_alpha: cfg.physics.splat_alpha,
        force_cap: cfg.physics.force_cap,
        temperature: cfg.generation.temperature as f32,
        min_splat_dist: cfg.physics.min_splat_dist,
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

    // Feed prompt through Llama (prefill)
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("    Prefilling {} prompt tokens...", prompt_ids.len());

    // Use forward_with_hidden when steer_hidden is enabled
    let (prefill_logits, prefill_hidden) = if cfg.physics.steer_hidden {
        let (logits, hidden) = model.forward_with_hidden(&prompt_tensor, 0)?;
        (logits, Some(hidden))
    } else {
        let logits = model.forward(&prompt_tensor, 0)?;
        (logits, None)
    };
    let mut index_pos = prompt_ids.len();

    // Goal attractor: from hidden state (steer_hidden) or logit space (fallback)
    let goal_pos = if let Some(ref hidden) = prefill_hidden {
        // Hidden state is already (1, D) -- squeeze to (D,)
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

    // Visualization collector (only when --viz is passed)
    let mut viz_collector: Option<VizCollector> = if viz_enabled {
        match VizCollector::new(engine.field_positions(), &goal_pos, prompt, dim) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("    [VIZ] Failed to init collector: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Now start generating from prefill
    let mut raw_logits = prefill_logits;
    let mut raw_hidden: Option<Tensor> = prefill_hidden;

    // Collect generated tokens
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Track last steered position for splat creation
    let mut last_steered_pos: Option<Tensor> = None;

    // Sliding window of recent hidden states for VR H1 reflex
    let mut recent_hidden: Vec<Tensor> = Vec::new();
    let mut last_reflex_step: usize = 0;

    // Full generation trajectory (real hidden states for dream replay)
    // trajectory_masses: per-token weight (1 - prob) — surprise = high mass
    let mut generation_trajectory: Vec<Tensor> = Vec::new();
    let mut trajectory_masses: Vec<f32> = Vec::new();

    println!(
        "\n    === Generation ({} tokens, physics-steered) ===\n",
        max_tokens
    );

    // Live stream file: per-token output for tail -f viewing
    use std::io::Write;
    let live_path = std::path::Path::new("logs/live.txt");
    let mut live_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(live_path)?;
    writeln!(live_file, "\n=== [{}] \"{}\" ===", model_variant, prompt)?;
    live_file.flush()?;

    #[allow(clippy::explicit_counter_loop)]
    for step in 0..max_tokens {
        // Steer: hidden state (steer_hidden=true) or logit slice (fallback)
        let (steer_input, is_hidden_steer) = if cfg.physics.steer_hidden {
            if let Some(ref h) = raw_hidden {
                (h.clone(), true) // already (1, D) from forward_with_hidden
            } else {
                // Fallback if hidden state unavailable
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
            steered: mut steered_slice,
            grad_mag,
            splat_mag,
            goal_mag,
        } = engine.steer(&steer_input, &goal_pos, step)?;

        // Manifold safety: blend steered state back toward baseline each step
        // Prevents cumulative drift off the Llama manifold
        if cfg.physics.manifold_pullback > 0.0 {
            let pb = cfg.physics.manifold_pullback as f64;
            steered_slice =
                (&steered_slice.affine(1.0 - pb, 0.0)? + &steer_input.affine(pb, 0.0)?)?;
        }

        // Bundle stress: collective force from nearby splats (emergent fluid structure)
        if engine.memory().len() > 3 {
            let pos = steered_slice.squeeze(0)?;
            let bundle = engine.memory().query_bundle_force(&pos, 8)?;
            let bundle_2d = bundle.unsqueeze(0)?;
            steered_slice = (&steered_slice + &bundle_2d.affine(0.01, 0.0)?)?;
        }

        last_steered_pos = Some(steered_slice.clone());

        // VR H1 reflex: track recent hidden states, check for zero-persistence cycles
        // On detection: blend steered slice 30% back toward baseline (collapse correction)
        if let Some(ref h) = raw_hidden {
            let h_flat = h.squeeze(0)?;
            recent_hidden.push(h_flat);
            if recent_hidden.len() > 12 {
                recent_hidden.remove(0);
            }
            if step > 50 && step % 100 == 0 && (step - last_reflex_step) >= 100 {
                if let Ok(true) = ridge::check_vr_h1_reflex(&recent_hidden, 2.0) {
                    last_reflex_step = step;
                    steered_slice =
                        (&steered_slice.affine(0.7, 0.0)? + &steer_input.affine(0.3, 0.0)?)?;
                    println!(
                        "    [REFLEX] step {} | VR H1 collapse -> corrective blend applied",
                        step
                    );
                }
            }
        }

        // === Micro-dream: entropy-adaptive steering consolidation ===
        let steered_slice = if step > 12 {
            let raw_probs_slice = candle_nn::ops::softmax(&raw_logits, 1)?;
            let raw_probs_flat: Vec<f32> = raw_probs_slice.squeeze(0)?.to_vec1()?;
            let sample_n = raw_probs_flat.len().min(1000);
            let entropy: f32 = raw_probs_flat[..sample_n]
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|p| -p * p.ln())
                .sum();

            let dream_steps = if entropy > 4.0 { 4 } else if entropy > 3.0 { 3 } else { 2 };
            let blend = if entropy > 2.5 { 0.12 } else { 0.07 };

            let result = micro_dream(&engine, &steered_slice, &goal_pos, step, dream_steps, blend)?;
            result.consolidated
        } else {
            steered_slice
        };

        // Reconstruct full logits for sampling
        let steered_logits = if is_hidden_steer {
            // Project steered hidden state through lm_head to get full vocab logits
            model.project_to_logits(&steered_slice)?
        } else {
            // Logit-space steering: cat steered slice with remaining logits
            if raw_logits.dim(1)? > dim {
                let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
                Tensor::cat(&[&steered_slice, &rest], 1)?
            } else {
                steered_slice
            }
        };

        // Repetition penalty: penalize tokens already generated
        let rep_penalty = cfg.generation.rep_penalty;
        let steered_logits = {
            let mut logits_vec: Vec<f32> = steered_logits.squeeze(0)?.to_vec1()?;
            for &tid in prompt_ids.iter().chain(generated_tokens.iter()) {
                if (tid as usize) < logits_vec.len() {
                    let l = &mut logits_vec[tid as usize];
                    if *l > 0.0 {
                        *l /= rep_penalty;
                    } else {
                        *l *= rep_penalty;
                    }
                }
            }
            Tensor::from_vec(logits_vec, steered_logits.dim(1)?, steered_logits.device())?
                .unsqueeze(0)?
        };

        // Temperature sampling -- softmax over scaled logits, then sample
        let temperature: f64 = cfg.generation.temperature;
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

        // Online splat update -- multi-scale creation based on steering strength
        if step > 5 && delta_norm > cfg.physics.splat_delta_threshold {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine
                    .memory()
                    .has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
                if !too_close {
                    // Alpha proportional to steering delta (advantage signal)
                    let splat_alpha = (delta_norm / 10.0).clamp(1.0, 5.0);
                    // Multi-scale: large deltas get coarse sigma, small deltas get fine sigma
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

        // Decode and stream to console
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));

        // Viz snapshot with nearest token attractors (zero cost when --viz not passed)
        if let Some(ref mut collector) = viz_collector {
            // Find top-5 highest probability tokens every 5 steps as attractors
            let neighbors = if step % 5 == 0 {
                // Use softmax probs to find what the model is attracted to
                // Partial sort: only find top-5 without fully sorting 128K items
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
            let _ = collector.snapshot(
                step,
                next_token,
                &decoded,
                &steered_logits,
                delta_norm,
                neighbors,
            );
        }

        // Stream tokens live -- print without newline for flowing text
        print!("{}", decoded);
        std::io::stdout().flush().ok();

        // Write to live stream file (for tail -f in separate terminal)
        write!(live_file, "{}", decoded).ok();
        live_file.flush().ok();

        // Milestone markers every 50 steps
        if step > 0 && step % 50 == 0 {
            println!("  [{}/{}]", step, max_tokens);
        }

        // Log every step to JSONL
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
        })?;

        // Stop on EOS tokens
        if cfg.generation.eos_token_ids.contains(&next_token) {
            println!("    → EOS at step {}", step);
            break;
        }

        // Feed next token
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        if cfg.physics.steer_hidden {
            let (logits, hidden) = model.forward_with_hidden(&next_input, index_pos)?;
            raw_logits = logits;
            raw_hidden = Some(hidden);
        } else {
            raw_logits = model.forward(&next_input, index_pos)?;
            raw_hidden = None;
        }
        index_pos += 1;

        // Collect hidden state for dream replay — AFTER forward pass so
        // trajectory[N] = state that produced token[N] (correct alignment)
        // Token mass: weight by surprise (low prob = high mass = stronger splat)
        if let Some(ref h) = raw_hidden {
            let mass = 1.0_f32 - probs_vec[next_token as usize].min(1.0);
            generation_trajectory.push(h.squeeze(0)?);
            trajectory_masses.push(mass);
        }
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
        let pos_1d = final_pos.squeeze(0)?;
        if generated_tokens.len() > cfg.generation.min_success_tokens {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                cfg.generation.pleasure_alpha,
            ));
            println!(
                "    + Added PLEASURE splat (generation succeeded: {} tokens)",
                generated_tokens.len()
            );
        } else {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                cfg.generation.pain_alpha,
            ));
            println!(
                "    x Added PAIN splat (generation too short: {} tokens)",
                generated_tokens.len()
            );
        }
        println!("    Splats in memory: {}", engine.memory().len());
    }

    // Evaporation: time-based decay + cull dead splats
    engine.memory_mut().decay_step(cfg.memory.decay_rate);
    let culled = engine.memory_mut().cull(cfg.memory.prune_threshold);
    if culled > 0 {
        println!("    [EVAPORATE] Culled {} dead splats", culled);
    }

    // Consolidate and cap splat memory before saving
    let _ = engine
        .memory_mut()
        .consolidate(cfg.memory.consolidation_dist);
    engine.memory_mut().prune_to_limit(cfg.memory.max_splats);

    // TODO: re-enable splat persistence + museum once steering is stable
    println!(
        "    Splats in memory: {} (persistence disabled)",
        engine.memory().len()
    );

    // =========================================================
    // Phase 6: Dream Replay (REAL — replays actual generation trajectory)
    // =========================================================
    println!("\n--- Phase 6: Dream Replay ---");
    let splat_count_before = engine.memory().len();
    if !generation_trajectory.is_empty() {
        let traj_refs: Vec<&Tensor> = generation_trajectory.iter().collect();
        let traj_stack = Tensor::stack(&traj_refs, 0)?;
        let noise = Tensor::randn(0.0f32, 0.05, traj_stack.dims(), &device)?;
        let noisy_traj = (&traj_stack + &noise)?;
        let replay_bonus = 1.25_f32;
        let masses_ref = if trajectory_masses.is_empty() {
            None
        } else {
            Some(trajectory_masses.as_slice())
        };
        let replay_count = engine.memory_mut().consolidate_trajectory(
            &noisy_traj,
            cfg.physics.splat_sigma,
            replay_bonus,
            cfg.physics.min_splat_dist,
            masses_ref,
        )?;
        let avg_mass = if trajectory_masses.is_empty() {
            1.0
        } else {
            trajectory_masses.iter().sum::<f32>() / trajectory_masses.len() as f32
        };
        println!(
            "    Dream replay: {} points -> {} splats (avg mass {:.3}, bonus {:.2})",
            generation_trajectory.len(),
            replay_count,
            avg_mass,
            replay_bonus,
        );
    } else {
        println!("    No hidden trajectory collected (steer_hidden disabled?)");
    }
    engine.memory_mut().decay_step(cfg.memory.decay_rate);
    println!(
        "    Applied decay ({:.3}). Splats remaining: {}",
        cfg.memory.decay_rate,
        engine.memory().len(),
    );

    // =========================================================
    // Summary
    // =========================================================
    let splat_type = if generated_tokens.len() > cfg.generation.min_success_tokens {
        "pleasure"
    } else {
        "pain"
    };
    let splat_count_after = engine.memory().len();
    logger.log_summary(SessionSummary {
        prompt: prompt.to_string(),
        prompt_token_count: prompt_ids.len(),
        generated_token_count: generated_tokens.len(),
        goal_attractor_norm: goal_norm,
        splat_count_before,
        splat_count_after,
        splat_type_added: splat_type.to_string(),
        decoded_output: full_text.clone(),
        delta_min: 0.0, // filled by log_summary
        delta_max: 0.0,
        delta_mean: 0.0,
    })?;

    println!("\n========================================");
    println!("  SplatRAG v1.1 -- OPERATIONAL");
    println!("========================================");
    println!("  Model:    {}", model_path);
    println!("  Variant:  {}", model_variant);
    println!("  Prompt:   \"{}\"", prompt);
    println!("  Tokens:   {} generated", generated_tokens.len());
    println!("  Log:      {}", logger.path().display());
    println!("  TACO:     {}", logger.taco_stats());
    println!("  Backend:  {} + Niodoo physics", engine.backend_name());
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
        let days = now / 86400;
        let day_secs = now % 86400;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let (y, m, d) = logger::days_to_date(days);
        writeln!(
            f,
            "=== Run: {}-{:02}-{:02} {:02}:{:02} UTC ===",
            y, m, d, hours, minutes
        )?;
        writeln!(
            f,
            "Model: {} | Tokens: {} | Splats: {}",
            model_variant,
            generated_tokens.len(),
            engine.memory().len()
        )?;
        writeln!(f, "Prompt: \"{}\"", prompt)?;
        writeln!(f)?;
        writeln!(f, "{}", full_text)?;
        writeln!(f)?;
        writeln!(f, "---")?;
        writeln!(f)?;
    }

    // =========================================================
    // Visualization export (JSON only — HTML viewer removed)
    // =========================================================
    if let Some(mut collector) = viz_collector {
        // Load real splat scar data from engine memory
        collector.load_splats(engine.memory());

        // Export JSON snapshot data
        let viz_path = logger.path().with_extension("viz.json");
        let _ = collector.export_json(&viz_path);
    }

    Ok(())
}

/// Find a file at primary path, fallback to secondary.
fn find_file(primary: &str, fallback: &str) -> Result<String> {
    if std::path::Path::new(primary).exists() {
        Ok(primary.to_string())
    } else if std::path::Path::new(fallback).exists() {
        Ok(fallback.to_string())
    } else {
        Err(anyhow::anyhow!(
            "Required file not found.\n  Tried: {}\n  Tried: {}\n  \
             Please ensure the data files are in the data/ directory.",
            primary,
            fallback
        ))
    }
}
