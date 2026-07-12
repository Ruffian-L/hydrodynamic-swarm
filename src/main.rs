//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Physics-steered generation over Llama 3.1 / Gemma 3 with shared-ocean multi-mind forces.
//! Type a prompt → physics steers generation → decoded text output.
//!
//! ## Licenses & attributions
//!
//! - Our code: MIT-0 (LICENSE)
//! - Candle loader code: Apache-2.0 OR MIT (NOT the same as model weights)
//! - Llama 3.1 weights: Llama 3.1 Community License — "Built with Llama" in README
//! - Gemma 3 weights: Gemma Terms of Use (NOT Apache; Gemma 4 is Apache)
//! - GGUF quants: bartowski, Unsloth (on top of Meta/Google terms)
//!
//! See NOTICE in the repo root.

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
mod ocean;
mod quality;
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
use niodoo::{FieldWakeConfig, FieldWakeMode, NiodooEngine, SteerResult};
use ocean::{MindId, OceanConfig, SharedOcean};
use quality::{alpha_for, classify, score_token, QualityThresholds, SplatKind};
use rand::RngExt;
use splat::Splat;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use viz::VizCollector;

// ═══════════════════════════════════════════════════════════════════════════════
// Model: dispatch enum wrapping Llama and Gemma for physics steering
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified model interface for the Niodoo physics engine.
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
            Model::Gemma(_) => "gemma3",
        }
    }

    fn is_gemma(&self) -> bool {
        matches!(self, Model::Gemma(_))
    }
}

/// GGUF architecture string from metadata (lowercase).
fn gguf_architecture(ct: &gguf_file::Content) -> String {
    ct.metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_lowercase())
        .unwrap_or_default()
}

/// Heuristic when metadata is missing: path name.
fn path_looks_like_gemma(path: &str) -> bool {
    let p = path.to_lowercase();
    p.contains("gemma") && !p.contains("llama")
}

/// Wrap a raw user prompt in Gemma 3 IT chat turns when needed.
///
/// Answer-only framing cuts the common IT failure mode of restating/rephrasing
/// the user request ("explain friendship from a physics perspective…") instead
/// of producing the paragraph. Raw prompts that already contain turn markers
/// are left unchanged.
fn format_prompt_for_model(raw: &str, is_gemma: bool) -> String {
    if !is_gemma {
        return raw.to_string();
    }
    if raw.contains("<start_of_turn>") {
        return raw.to_string();
    }
    // Keep the user turn short — long “do not restate” preambles were not
    // helping 27B and bloated the J-space prefill (goal norm collapsed).
    format!(
        "<start_of_turn>user\n\
         Answer in one short paragraph only.\n\n\
         {}\n\
         <end_of_turn>\n\
         <start_of_turn>model\n",
        raw.trim()
    )
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
    let cli_tokenizer = args
        .iter()
        .position(|a| a == "--tokenizer")
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
    // Phase 1: Load GGUF (Gemma 3 or Llama) + Tokenizer
    // =========================================================
    println!("\n--- Phase 1: Loading Model + Tokenizer ---");

    // Prefer --model, then Gemma 3 sizes that the gemma3 loader can open, then Llama.
    let model_path = cli_model
        .filter(|path| Path::new(path).exists())
        .or_else(|| {
            find_existing_file(&[
                // 4B first for fast size-scaled iteration (see docs/MODEL_SIZE_PHYSICS_SCALING.md)
                "data/google/gemma-3-4b-it-Q4_K_M.gguf",
                "data/google/gemma-3-27b-it-Q4_K_M.gguf",
                "data/google/gemma-3-27b-it-Q8_0.gguf",
                "data/gemma-3-27b-it-Q8_0.gguf",
                // gemma3n / gemma4 need dedicated loaders — do not list until wired
                "data/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                "/home/ruffianl/projects/niodoo-live/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            ])
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Required model file not found. Pass --model /path/to/model.gguf or put it in data/."
            )
        })?;
    println!("    Model: {}", model_path);

    let mut file = std::fs::File::open(&model_path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;
    let arch = gguf_architecture(&ct);
    let load_gemma = arch.contains("gemma3")
        || (arch.is_empty() && path_looks_like_gemma(&model_path))
        || (arch.contains("gemma") && !arch.contains("gemma4"));

    if arch.contains("gemma4") {
        anyhow::bail!(
            "GGUF architecture is '{arch}' (Gemma 4). This harness loads Gemma 3 via gemma.rs. \
             Use data/google/gemma-3-27b-it-Q4_K_M.gguf (fast) or Q8_0, or Llama 3.1."
        );
    }
    if arch.contains("gemma3n") {
        anyhow::bail!(
            "GGUF architecture is '{arch}' (Gemma 3n E2B/E4B). Needs a dedicated loader \
             (AltUp + Laurel + per-layer emb). File is fine at data/google/google_gemma-3n-E4B-it-Q5_K_M.gguf \
             — not wired yet. Use gemma-3-27b-it-Q4_K_M.gguf for fast iteration today."
        );
    }

    let mut model = if load_gemma {
        println!("    Architecture: {} → Gemma 3 loader", if arch.is_empty() { "path-heuristic".into() } else { arch.clone() });
        let m = gemma::ModelWeights::from_gguf(ct, &mut reader, &device)?;
        println!("    Gemma 3 loaded (hidden_dim={})", m.hidden_dim);
        Model::Gemma(m)
    } else {
        println!("    Architecture: {} → Llama loader", if arch.is_empty() { "default".into() } else { arch.clone() });
        let m = llama::ModelWeights::from_gguf(ct, &mut reader, &device)?;
        println!("    Llama loaded");
        Model::Llama(m)
    };

    // Find tokenizer. Prefer --tokenizer, then next to model, then fallbacks.
    let tokenizer_path = cli_tokenizer
        .filter(|path| Path::new(path).exists())
        .or_else(|| tokenizer_next_to_model(&model_path))
        .or_else(|| {
            find_existing_file(&[
                "data/google/tokenizer.json",
                "data/tokenizer.json",
                "/home/ruffianl/projects/niodoo-live/model/tokenizer.json",
                "/home/ruff/projects/Homernd/team_build/niodoo/model/tokenizer.json",
            ])
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Required tokenizer file not found. Pass --tokenizer /path/to/tokenizer.json or put it next to the model."
            )
        })?;
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
    // Phase 3: Niodoo Engine + Shared Ocean (Lane C)
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
    }
    engine.set_splat_force_limits(
        cfg.physics.splat_force_scale,
        cfg.physics.splat_force_max,
    );
    engine.set_goal_force_limits(
        cfg.physics.goal_force_scale,
        cfg.physics.goal_force_max,
    );
    engine.set_goal_late_attenuate(
        cfg.physics.goal_late_start,
        cfg.physics.goal_late_span,
        cfg.physics.goal_late_end,
    );
    let wake_mode = FieldWakeMode::parse(&cfg.physics.field_wake_mode);
    engine.set_field_wake(FieldWakeConfig {
        mode: wake_mode,
        k: cfg.physics.field_wake_k.max(1),
        scale: cfg.physics.field_wake_scale,
        max_mag: cfg.physics.field_wake_max,
        grad_blend: cfg.physics.field_grad_blend,
        dist_tau: cfg.physics.field_wake_dist_tau,
    });
    engine.set_force_ramp(cfg.physics.force_ramp_tokens, cfg.physics.force_ramp_start);
    println!(
        "    Engine ready (backend: {}, Top-K: {}, F_s={}/{}, F_a={}/{})",
        engine.backend_name(),
        cfg.physics.gradient_topk,
        cfg.physics.splat_force_scale,
        cfg.physics.splat_force_max,
        cfg.physics.goal_force_scale,
        cfg.physics.goal_force_max
    );
    if cfg.physics.force_ramp_tokens > 0 {
        println!(
            "    Force ramp: first {} tokens from {:.2} → 1.0 (J-space respect)",
            cfg.physics.force_ramp_tokens, cfg.physics.force_ramp_start
        );
    }
    if cfg.physics.goal_late_start > 0 {
        println!(
            "    Late F_a attenuate: after step {} → ×{:.2} over {} tok (early goal intact)",
            cfg.physics.goal_late_start,
            cfg.physics.goal_late_end,
            cfg.physics.goal_late_span
        );
    }
    if cfg.physics.targeted_splat_only {
        println!("    Targeted splats: ON (high-δ or strong quality only)");
    }
    println!(
        "    Field wake: mode={} k={} scale={} max={} blend={} τ={}",
        wake_mode.as_str(),
        cfg.physics.field_wake_k,
        cfg.physics.field_wake_scale,
        cfg.physics.field_wake_max,
        cfg.physics.field_grad_blend,
        cfg.physics.field_wake_dist_tau
    );
    if cfg.physics.field_logit_alpha > 0.0 {
        println!(
            "    Field logit bias: α={}  (z += α·norm(E û_g) pre-softmax)",
            cfg.physics.field_logit_alpha
        );
    } else {
        println!("    Field logit bias: off");
    }

    // Shared ocean: multi-mind field packets (starts with host deposits).
    // On 27B (~5376-d) single-host ocean was soft-yanking (F_ocean ~15–24) while
    // scar force stayed quiet — disable multi-mind ocean until multi-mind or
    // config knobs land. 4B keeps a light ocean for Lane C experiments.
    let mut ocean_cfg = OceanConfig::default();
    if dim >= 4096 {
        ocean_cfg.enabled = false;
    }
    let ocean = SharedOcean::new(dim, device.clone(), ocean_cfg.clone());
    engine.set_ocean(ocean);
    if ocean_cfg.enabled {
        println!(
            "    Shared Ocean online (dim={}, deposit every {}, force_scale={})",
            dim, ocean_cfg.deposit_interval, ocean_cfg.force_scale
        );
    } else {
        println!(
            "    Shared Ocean: OFF for dim={} (single-host soft-yank guard)",
            dim
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
    let is_gemma = model.is_gemma();
    let raw_prompt = cli_prompt
        .as_deref()
        .unwrap_or(cfg.generation.default_prompt.as_str());
    let prompt = format_prompt_for_model(raw_prompt, is_gemma);
    // Gemma IT: <eos>=1, <end_of_turn>=106. Llama 3: 128009/128001.
    let eos_token_ids: Vec<u32> = if is_gemma {
        vec![1, 106]
    } else {
        cfg.generation.eos_token_ids.clone()
    };
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
        prompt: raw_prompt.to_string(),
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
    println!("    Prompt: \"{}\"", raw_prompt);
    if is_gemma {
        println!("    Chat template: Gemma 3 IT turns applied");
    }

    // Encode prompt
    let encoded = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    println!("    Prompt tokens: {} IDs", prompt_ids.len());

    // Prefill
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
    // This prefill hidden is the "J-space" / pre-verbal image of the prompt.
    let goal_pos = if let Some(ref hidden) = prefill_hidden {
        // Hidden state is already (1, D) -- squeeze to (D,)
        let h = hidden.squeeze(0)?;
        println!(
            "    Goal attractor (J-space): from prefill hidden (D={}, steer_hidden=true)",
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

    // Optional reflective micro-dream on prefill (variant D) — work with J-space, not yank it
    let mut prefill_hidden = prefill_hidden;
    if cfg.physics.prefill_micro_dream {
        if let Some(ref h) = prefill_hidden {
            let r = micro_dream(&mut engine, h, &goal_pos, 0, 3, 0.08)?;
            println!(
                "    Prefill micro-dream (J-space): ||corr||={:.3} reflection={}",
                r.correction_norm, r.reflection_triggered
            );
            prefill_hidden = Some(r.consolidated);
        }
    }

    // Visualization collector (only when --viz is passed)
    let mut viz_collector: Option<VizCollector> = if viz_enabled {
        match VizCollector::new(engine.field_positions(), &goal_pos, raw_prompt, dim) {
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
    let mut last_online_splat_step: isize = -999;

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
            ocean_mag,
            field_dir,
        } = engine.steer(&steer_input, &goal_pos, step)?;

        // Ocean deposit moved to *after* token quality scoring (quality-gated).
        // Depositing every 4 steps without quality was crystallizing late garbage.

        // Manifold safety: blend steered state back toward baseline each step
        // Prevents cumulative drift off the model manifold
        if cfg.physics.manifold_pullback > 0.0 {
            let pb = cfg.physics.manifold_pullback as f64;
            steered_slice =
                (&steered_slice.affine(1.0 - pb, 0.0)? + &steer_input.affine(pb, 0.0)?)?;
        }

        // Bundle stress: light K-NN scar pull AFTER main steer.
        // NOTE: this path intentionally bypasses niodoo force_cap/ramp (applied
        // inside engine.steer). Keep the 0.01 scale tiny; do not raise without
        // folding bundle into the capped total_force sum in niodoo.rs.
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
            // Threshold was 2.0 (almost always true). True near-zero persistence ~1.05–1.15.
            // Also require a real late-run stress signal: high splat force or many recent pain-ish steps.
            if step > 80 && step % 100 == 0 && (step - last_reflex_step) >= 100 {
                let stress = splat_mag > 35.0 || grad_mag + splat_mag + goal_mag > 100.0;
                if stress {
                    if let Ok(true) = ridge::check_vr_h1_reflex(&recent_hidden, 1.12) {
                        last_reflex_step = step;
                        steered_slice =
                            (&steered_slice.affine(0.7, 0.0)? + &steer_input.affine(0.3, 0.0)?)?;
                        println!(
                            "    [REFLEX] step {} | tight H1 (thr=1.12) + stress F_s={:.1} -> blend",
                            step, splat_mag
                        );
                    }
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

            let dream_steps = if entropy > 4.0 {
                4
            } else if entropy > 3.0 {
                3
            } else {
                2
            };
            let blend = if entropy > 2.5 { 0.12 } else { 0.07 };

            let result = micro_dream(&mut engine, &steered_slice, &goal_pos, step, dream_steps, blend)?;
            result.consolidated
        } else {
            steered_slice
        };

        // Reconstruct full logits for sampling
        let mut steered_logits = if is_hidden_steer {
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

        // ── Surface field logit bias (Gemini-style bridge) ─────────────────
        // z_final = z + α · ŝ ,  ŝ = normalize(E û_g)
        // û_g = unit F_g direction from residual steer (same D as token emb).
        // Does not replace residual physics — tips vocab toward field-aligned tokens.
        if cfg.physics.field_logit_alpha > 0.0 && grad_mag > 1e-8 {
            let emb = model.token_embeddings(); // (V, D)
            let v = field_dir.to_dtype(emb.dtype())?.to_device(emb.device())?;
            // scores: (V,) = E @ û_g
            let scores = emb.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
            // Peak-normalize so α is a comparable logit-scale knob across steps
            let peak: f32 = scores
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                .max(1e-8);
            let bias = scores
                .affine((cfg.physics.field_logit_alpha / peak) as f64, 0.0)?
                .unsqueeze(0)?; // (1, V)
            steered_logits = (&steered_logits + &bias)?;
        }

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
        let mut rng = rand::rng();
        let roll: f32 = rng.random();
        let mut cumsum = 0.0f32;
        let mut next_token: u32 = 0;
        for (i, p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if roll < cumsum {
                next_token = i as u32;
                break;
            }
        }

        // Steering delta (telemetry / multi-scale only — NOT the definition of "good")
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Decode first so quality scoring can see the surface form
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));

        // ── Semantic splat: "good" = confident non-spam, "bad" = surprise/loop ──
        // Quantified from P(token), top-k entropy, recent repeats — not δ.
        let q_thr = QualityThresholds::default();
        let quality = score_token(
            &probs_vec,
            next_token,
            &decoded,
            &generated_tokens,
            &q_thr,
        );
        let kind = classify(&quality, &q_thr);
        let interval = cfg.physics.online_splat_interval.max(1);
        let rate_ok = step > 4
            && (step as isize - last_online_splat_step) >= interval as isize;
        // High-signal: steering event OR pain OR strong pleasure (original Niodoo targeting)
        let high_delta = delta_norm > cfg.physics.splat_delta_threshold;
        let high_signal = high_delta
            || kind == SplatKind::Pain
            || (kind == SplatKind::Pleasure && quality.p_chosen >= 0.25);
        let splat_ok = if cfg.physics.targeted_splat_only {
            rate_ok && kind != SplatKind::Skip && high_signal
        } else {
            rate_ok && kind != SplatKind::Skip
        };
        if splat_ok {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine
                    .memory()
                    .has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
                if !too_close {
                    let splat_alpha = alpha_for(
                        kind,
                        &quality,
                        cfg.generation.pleasure_alpha,
                        cfg.generation.pain_alpha,
                    );
                    // Hierarchical width relative to deposit threshold (not absolute 20/30)
                    engine.memory_mut().add_splat(Splat::with_scale_ref_lambda(
                        current_pos,
                        cfg.physics.splat_sigma,
                        splat_alpha,
                        delta_norm,
                        cfg.physics.splat_delta_threshold,
                        cfg.physics.splat_lambda_default,
                    ));
                    // Cap during generation — prune_to_limit used to run only in Phase 5
                    // after the full loop, so 1000-tok runs could grow memory unbounded
                    // and F_s latched even with 1/√n damp + force caps.
                    engine
                        .memory_mut()
                        .prune_to_limit(cfg.memory.max_splats);
                    last_online_splat_step = step as isize;
                    if step % 20 == 0 || kind == SplatKind::Pain || high_delta {
                        println!(
                            "    [SPLAT {:?}] p={:.3} H≈{:.2} δ={:.1} α={:.2} «{}»",
                            kind,
                            quality.p_chosen,
                            quality.topk_entropy,
                            delta_norm,
                            splat_alpha,
                            decoded.replace('\n', "⏎")
                        );
                    }
                }
            }
        }

        // Lane C ocean: quality-gated deposits (original: not every token)
        if let Some(ocean) = engine.ocean_mut() {
            if step > 0 && step % ocean.config.deposit_interval == 0 {
                let host_vec = steer_input.squeeze(0)?;
                let mind = if is_gemma {
                    MindId::Gemma
                } else {
                    MindId::Host
                };
                match kind {
                    SplatKind::Pleasure if high_signal || !cfg.physics.targeted_splat_only => {
                        let w = quality.p_chosen.clamp(0.3, 1.0);
                        let noise = (0.55 - 0.3 * quality.p_chosen).clamp(0.15, 0.55);
                        ocean.deposit(mind, &host_vec, w, noise)?;
                    }
                    SplatKind::Pain => {
                        if cfg.physics.pain_recovery_ocean {
                            // Variant E: recovery anchor — stronger corrective packet
                            ocean.deposit(mind, &host_vec, 0.85, 0.35)?;
                            if step % 10 == 0 {
                                println!(
                                    "    [OCEAN recovery] pain packet p={:.3} δ={:.1}",
                                    quality.p_chosen, delta_norm
                                );
                            }
                        } else {
                            ocean.deposit(mind, &host_vec, 0.15, 0.92)?;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Mid-run F_s control: per-token scar alpha decay (not wall-clock decay_step).
        if cfg.memory.online_decay_rate < 1.0 && engine.memory().len() > 0 {
            engine.memory_mut().decay_per_token(
                cfg.memory.online_decay_rate,
                cfg.physics.pain_decay_factor,
            );
            if step > 0 && step % 25 == 0 {
                let _ = engine.memory_mut().cull(cfg.memory.prune_threshold);
            }
        }

        generated_tokens.push(next_token);

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
            let ocean_info = engine
                .ocean()
                .map(|o| {
                    format!(
                        " ocean_n={} noise={:.2} F_ocean={:.2}",
                        o.len(),
                        o.mean_noise(),
                        ocean_mag
                    )
                })
                .unwrap_or_default();
            println!(
                "  [{}/{}] δ={:.1} F_g={:.1} F_s={:.1} F_a={:.1}{}",
                step, max_tokens, delta_norm, grad_mag, splat_mag, goal_mag, ocean_info
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
            grad_force_mag: grad_mag,
            splat_force_mag: splat_mag,
            goal_force_mag: goal_mag,
        })?;

        // Stop on EOS tokens
        if eos_token_ids.contains(&next_token) {
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
        prompt: raw_prompt.to_string(),
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

    let ocean_summary = engine
        .ocean()
        .map(|o| {
            format!(
                "  Ocean:    {} packets | deposits={} | mean_noise={:.3}",
                o.len(),
                o.total_deposits(),
                o.mean_noise()
            )
        })
        .unwrap_or_else(|| "  Ocean:    offline".into());

    println!("\n========================================");
    println!("  SplatRAG v1.1 -- OPERATIONAL");
    println!("========================================");
    println!("  Model:    {}", model_path);
    println!("  Variant:  {}", model_variant);
    println!("  Prompt:   \"{}\"", raw_prompt);
    println!("  Tokens:   {} generated", generated_tokens.len());
    println!("{}", ocean_summary);
    println!("  Log:      {}", logger.path().display());
    println!("  TACO:     {}", logger.taco_stats());
    println!("  Backend:  {} + Niodoo physics + Shared Ocean", engine.backend_name());
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
            .unwrap_or_default()
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
        writeln!(f, "Prompt: \"{}\"", raw_prompt)?;
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

fn find_existing_file(paths: &[&str]) -> Option<String> {
    paths
        .iter()
        .find(|path| Path::new(path).exists())
        .map(|path| (*path).to_string())
}

fn tokenizer_next_to_model(model_path: &str) -> Option<String> {
    let tokenizer_path: PathBuf = Path::new(model_path).parent()?.join("tokenizer.json");
    tokenizer_path
        .exists()
        .then(|| tokenizer_path.display().to_string())
}
