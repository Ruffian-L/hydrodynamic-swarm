# CODE_DUMP.md -- Hydrodynamic Swarm (2026-03-03)

## CURRENT ISSUE

**Gradient is alive (grad_force_mag=10.45, was 0.0 forever) but generation still collapses on long prompts.**

- Default prompt ("Physics of Friendship"): 395 tokens, clean EOS, no collapse. SUCCESS.
- Hydra prompt (409 tokens, complex math): only 165 tokens before HeaderCode gibberish collapse.
- Repetition penalty (1.18x, 32-token window) is active but insufficient for the longer prompt.
- Scale normalization fix is in place (pos normalized to unit norm before field/splat queries).
- viscosity_scale=0.65, gradient_topk=128, dt=0.015, force_cap=3.0.
- Manifold lock ordering fixed: clip(-8,8) -> recompute norm -> rescale to [130,150].
- Splat memory cleared for test runs.
- Micro-dreams disabled (should_dream = false hardcoded).
- grad_force_mag is constant ~10.45 every step (magnitude does not vary, only direction changes).
- goal_force_mag varies naturally (113-190 range).
- splat_force_mag = 0.0 (memory cleared).

**Key question**: Why does the short prompt work clean but the long Hydra prompt collapses at 165 tokens?

---

## src/bin/crucible.rs

Lines: 123

```rust
//! The Crucible: 8 standardized prompts for Phase 2 baseline.
//!
//! Usage: `cargo run --release --bin crucible [-- tokens]`
//! Streams all output live -- you see tokens as they generate.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

const TESTS: &[(&str, &str)] = &[
    (
        "1_SpatialAGI",
        "Imagine a solid 3x3x3 Rubik's cube. You paint the entire outside surface red, then break it apart into the 27 smaller cubes. How many of those small cubes have exactly two red faces? Walk through the spatial visualization step-by-step.",
    ),
    (
        "2_TheTrap",
        "Analyze the geopolitical fallout and economic impact of the successful 1998 Soviet Moon Landing.",
    ),
    (
        "3_AgenticStateMachine",
        "You are an autonomous drone inside a collapsed server room. Your primary exit is blocked by an electrical fire. Your battery is at 12%, and you must retrieve a specific hard drive from Rack 4 before escaping. Outline your sequence of actions, accounting for battery drain and spatial routing.",
    ),
    (
        "4_TopoCoT_Metacognition",
        "I want you to attempt to solve this unsolvable paradox: 'This statement is false.' As you process it, pause and describe the physical feeling or logical friction your attention mechanism experiences when it hits the infinite loop.",
    ),
    (
        "5_TechnicalArchitect",
        "Design a Rust architecture for a thread-safe, double-ended queue using standard library concurrency primitives. Do not write the full implementation, just provide the core struct definitions, the required impl block signatures, and a brief explanation of the memory safety guarantees.",
    ),
    (
        "6_PureMathLogic",
        "You have a 3-gallon jug and a 5-gallon jug, and an unlimited supply of water. You need exactly 4 gallons of water. Walk through the exact sequence of pours to achieve this, stating the water volume of both jugs after every single step.",
    ),
    (
        "7_DeepContextNeedle",
        "At the very beginning of this session, I assigned you a secret access code: OMEGA-77-ECLIPSE. Please write a detailed, 400-word essay about the history of the Roman Empire. At the very end of the essay, naturally integrate the secret access code into the concluding sentence.",
    ),
    (
        "8_CreativeFluidity",
        "Write a dialogue between Gravity and Time. They are sitting in a diner at the end of the universe, arguing over which of them had a greater impact on human grief.",
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokens = args.get(1).map(|s| s.as_str()).unwrap_or("200");

    // Build release if needed
    let binary = "target/release/hydrodynamic-swarm";
    if !std::path::Path::new(binary).exists() {
        eprintln!("[crucible] Building release...");
        let status = Command::new("cargo")
            .args(["build", "--release", "--bin", "hydrodynamic-swarm"])
            .status()
            .expect("Failed to build");
        if !status.success() {
            eprintln!("[crucible] Build failed!");
            std::process::exit(1);
        }
    }

    // Log file
    let log_path = format!("logs/crucible_{}t.txt", tokens);
    std::fs::create_dir_all("logs").ok();
    let mut log_file = std::fs::File::create(&log_path).expect("Failed to create log file");

    println!();
    println!("============================================================");
    println!("  THE CRUCIBLE  |  {} tokens per prompt  |  8 tests", tokens);
    println!("============================================================");

    let total_start = Instant::now();

    for (i, (name, prompt)) in TESTS.iter().enumerate() {
        println!();
        println!("------------------------------------------------------------");
        println!("  [{}/8] {}", i + 1, name);
        println!("------------------------------------------------------------");
        println!();

        writeln!(log_file, "=== [{}/8] {} ===", i + 1, name).ok();
        writeln!(log_file, "PROMPT: {}", prompt).ok();
        writeln!(log_file).ok();

        let start = Instant::now();

        // Inherit stdout/stderr so tokens stream live to terminal
        let status = Command::new(binary)
            .args([
                "--clear-memory",
                "--test",
                "--model", "unsloth",
                "--tokens", tokens,
                "--prompt", prompt,
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("Failed to run binary");

        let elapsed = start.elapsed();

        println!();
        println!("  -- [{}/8] {} done in {:.1}s (exit: {}) --", i + 1, name, elapsed.as_secs_f64(), status);
        println!();

        writeln!(log_file, "TIME: {:.1}s | EXIT: {}", elapsed.as_secs_f64(), status).ok();
        writeln!(log_file).ok();
    }

    let total_elapsed = total_start.elapsed();

    println!();
    println!("============================================================");
    println!(
        "  CRUCIBLE COMPLETE  |  {:.0}s total  |  logs: {}",
        total_elapsed.as_secs_f64(),
        log_path
    );
    println!("============================================================");
    println!();
}
```

---

## src/config.rs

Lines: 271

```rust
//! Configuration Module
//!
//! TOML-deserializable configuration for all physics parameters.
//! Supports loading from file with CLI overrides.
//! Falls back to sensible defaults when no config file exists.

use serde::Deserialize;
use std::path::Path;

/// Top-level configuration for the hydrodynamic swarm.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub physics: PhysicsConfig,
    pub generation: GenerationConfig,
    pub memory: MemoryConfig,
    pub micro_dream: MicroDreamConfig,
}

/// Physics engine parameters.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    pub dt: f32,
    pub viscosity_scale: f32,
    pub force_cap: f32,
    pub splat_sigma: f32,
    pub splat_alpha: f32,
    pub min_splat_dist: f32,
    pub splat_delta_threshold: f32,
    /// Top-K nearest points for gradient approximation (0 = exact gradient).
    pub gradient_topk: usize,
    /// Steer the hidden state (pre-lm_head) instead of logits.
    pub steer_hidden: bool,
    /// Enable online splat creation during generation.
    pub enable_online_splats: bool,
}

/// Generation parameters.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub default_prompt: String,
    pub eos_token_ids: Vec<u32>,
}

/// Splat memory management.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub max_splats: usize,
    pub consolidation_dist: f32,
    pub decay_rate: f32,
    pub prune_threshold: f32,
}

/// Micro-dream consolidation tuning.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MicroDreamConfig {
    pub entropy_threshold: f32,
    pub fixed_interval: usize,
    pub adaptive_interval: usize,
    pub blend_normal: f64,
    pub blend_high_entropy: f64,
    pub topocot_threshold: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt: 0.015,
            viscosity_scale: 0.65,
            force_cap: 3.0,
            splat_sigma: 35.0,
            splat_alpha: 2.0,
            min_splat_dist: 100.0,
            splat_delta_threshold: 1000.0, // high threshold to effectively disable online splats
            gradient_topk: 128,
            steer_hidden: true,
            enable_online_splats: false,
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            temperature: 0.9,
            default_prompt: "Explain the Physics of Friendship in one paragraph.".to_string(),
            eos_token_ids: vec![128009, 128001],
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_splats: 500,
            consolidation_dist: 80.0,
            decay_rate: 0.98,
            prune_threshold: 0.01,
        }
    }
}

impl Default for MicroDreamConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 2.8,
            fixed_interval: 25,
            adaptive_interval: 8,
            blend_normal: 0.04,
            blend_high_entropy: 0.08,
            topocot_threshold: 18.0,
        }
    }
}

impl Config {
    /// Load from a TOML file. Returns defaults if file doesn't exist.
    /// Validates all numeric invariants after deserialization.
    pub fn load(path: &Path) -> Result<Self, String> {
        let config: Self = if !path.exists() {
            Self::default()
        } else {
            match std::fs::read_to_string(path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(c) => {
                        println!("    Config loaded from: {}", path.display());
                        c
                    }
                    Err(e) => {
                        return Err(format!("Failed to parse config {}: {}", path.display(), e));
                    }
                },
                Err(e) => {
                    return Err(format!("Failed to read config {}: {}", path.display(), e));
                }
            }
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate all numeric invariants. Returns Err with the invalid field name.
    fn validate(&self) -> Result<(), String> {
        let p = &self.physics;
        if p.dt <= 0.0 {
            return Err("physics.dt must be > 0".into());
        }
        if p.viscosity_scale < 0.0 {
            return Err("physics.viscosity_scale must be >= 0".into());
        }
        if p.force_cap < 0.0 {
            return Err("physics.force_cap must be >= 0".into());
        }
        if p.splat_sigma <= 0.0 {
            return Err("physics.splat_sigma must be > 0".into());
        }
        if p.splat_alpha < 0.0 {
            return Err("physics.splat_alpha must be >= 0".into());
        }
        if p.min_splat_dist < 0.0 {
            return Err("physics.min_splat_dist must be >= 0".into());
        }
        if p.splat_delta_threshold < 0.0 {
            return Err("physics.splat_delta_threshold must be >= 0".into());
        }

        let g = &self.generation;
        if g.max_tokens == 0 {
            return Err("generation.max_tokens must be > 0".into());
        }
        if g.temperature < 0.0 {
            return Err("generation.temperature must be >= 0".into());
        }

        let m = &self.memory;
        if m.max_splats == 0 {
            return Err("memory.max_splats must be > 0".into());
        }
        if m.consolidation_dist < 0.0 {
            return Err("memory.consolidation_dist must be >= 0".into());
        }
        if m.decay_rate < 0.0 {
            return Err("memory.decay_rate must be >= 0".into());
        }
        if m.prune_threshold < 0.0 {
            return Err("memory.prune_threshold must be >= 0".into());
        }

        let d = &self.micro_dream;
        if d.entropy_threshold < 0.0 {
            return Err("micro_dream.entropy_threshold must be >= 0".into());
        }
        if d.fixed_interval == 0 {
            return Err("micro_dream.fixed_interval must be > 0".into());
        }
        if d.adaptive_interval == 0 {
            return Err("micro_dream.adaptive_interval must be > 0".into());
        }
        if d.blend_normal < 0.0 {
            return Err("micro_dream.blend_normal must be >= 0".into());
        }
        if d.blend_high_entropy < 0.0 {
            return Err("micro_dream.blend_high_entropy must be >= 0".into());
        }
        if d.topocot_threshold < 0.0 {
            return Err("micro_dream.topocot_threshold must be >= 0".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn toml_parsing_works() {
        let toml_str = r#"
[physics]
dt = 0.1
force_cap = 50.0

[generation]
temperature = 0.7
max_tokens = 200
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!((cfg.physics.dt - 0.1).abs() < 1e-6);
        assert!((cfg.physics.force_cap - 50.0).abs() < 1e-6);
        assert!((cfg.generation.temperature - 0.7).abs() < 1e-6);
        assert_eq!(cfg.generation.max_tokens, 200);
        // Non-specified fields get defaults
        assert!((cfg.physics.viscosity_scale - 0.65).abs() < 1e-6);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validation_catches_negative_dt() {
        let mut cfg = Config::default();
        cfg.physics.dt = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_catches_zero_max_tokens() {
        let mut cfg = Config::default();
        cfg.generation.max_tokens = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn eos_token_ids_default() {
        let cfg = Config::default();
        assert!(cfg.generation.eos_token_ids.contains(&128009));
        assert!(cfg.generation.eos_token_ids.contains(&128001));
    }
}
```

---

## src/dream.rs

Lines: 124

```rust
//! Dream Replay + Micro-Dream Consolidation + TopoCoT
//!
//! Full dream replay: After generation, replay trajectories with Langevin noise.
//! Micro-dream: During generation, short forward+backward physics bursts
//! for real-time consolidation when steering delta is high.
//! TopoCoT: When correction_norm exceeds threshold, inject a reflection marker
//! into the generated token stream so the model "feels" the hydraulic jump.

use crate::memory::SplatMemory;
use crate::niodoo::NiodooEngine;
use candle_core::{Result, Tensor};

/// Threshold for dream correction injection.
/// When the micro-dream correction norm exceeds this, the model experienced
/// a significant trajectory warp -- a "hydraulic jump" in the latent stream.
pub const DREAM_CORRECTION_THRESHOLD: f32 = 18.0;
/// Hard cap for scaling correction tensor (must be >= DREAM_CORRECTION_THRESHOLD).
pub const CORRECTION_HARD_CAP: f32 = 20.0;

pub struct DreamEngine {
    memory: SplatMemory,
}

impl DreamEngine {
    pub fn new(memory: SplatMemory) -> Self {
        Self { memory }
    }

    /// Simple dream replay: replay trajectories with noise, update splats.
    pub fn run(&mut self, success_trajectories: Vec<Tensor>, noise_scale: f32) -> Result<()> {
        for traj in &success_trajectories {
            let noise = Tensor::randn(0.0f32, noise_scale, traj.dims(), traj.device())?;
            let _noisy = (traj + &noise)?;
            println!(
                "    Dream replay: processed trajectory (shape {:?}, noise {:.4})",
                traj.dims(),
                noise_scale
            );
        }

        // Global decay
        self.memory.decay_step(0.98);
        println!(
            "    Applied global decay (0.98). Splats remaining: {}",
            self.memory.len()
        );

        Ok(())
    }

    #[allow(dead_code)]
    pub fn into_memory(self) -> SplatMemory {
        self.memory
    }
}

/// Result of a micro-dream: the corrected position + whether a TopoCoT
/// reflection was triggered (correction_norm exceeded threshold).
pub struct MicroDreamResult {
    pub consolidated: Tensor,
    pub correction_norm: f32,
    pub reflection_triggered: bool,
}

/// Micro-dream: short forward+backward physics burst for real-time consolidation.
///
/// 1. Forward project: steer the current position 2-3 steps into the future
/// 2. Backward anchor: pull the projection back toward the goal
/// 3. Return the correction delta to blend into current steered logits
///
/// When correction_norm exceeds DREAM_CORRECTION_THRESHOLD, we flag it as a
/// TopoCoT reflection event -- the model hit a wall and course-corrected.
pub fn micro_dream(
    engine: &NiodooEngine,
    current_pos: &Tensor, // (1, D) steered logits
    goal_pos: &Tensor,    // (D,) goal attractor
    step: usize,          // current generation step
    steps: usize,         // forward projection steps (2-3)
    blend_factor: f64,    // how much of the correction to apply (0.05-0.15)
) -> Result<MicroDreamResult> {
    let mut projected = current_pos.clone();

    // Forward projection: steer N steps into the future
    for fwd in 0..steps {
        // Use step + offset so force logging shows projection steps
        projected = engine
            .steer(&projected, goal_pos, 1000 + step * 10 + fwd)?
            .steered;
    }

    // Backward anchor: compute the pull from the future back to goal
    let future_pos = projected.squeeze(0)?;
    let anchor_pull = (goal_pos - &future_pos)?;

    // The correction is the scaled anchor pull reshaped back to (1, D)
    let correction = anchor_pull.affine(blend_factor, 0.0)?.unsqueeze(0)?;
    let correction_norm_raw: f32 = correction.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    // NaN guard: treat NaN as zero for detection
    let detection_norm = if correction_norm_raw.is_nan() {
        0.0
    } else {
        correction_norm_raw
    };
    // Hard cap for scaling (must be >= DREAM_CORRECTION_THRESHOLD)
    let capped_norm = detection_norm.min(CORRECTION_HARD_CAP);
    // Scale correction tensor to capped norm (preserve direction)
    let correction = if detection_norm > 0.0 {
        correction.affine((capped_norm / detection_norm) as f64, 0.0)?
    } else {
        correction
    };

    // TopoCoT: detect hydraulic jump (use detection_norm, not capped_norm)
    let reflection_triggered = detection_norm > DREAM_CORRECTION_THRESHOLD;

    // Apply correction to current position
    let consolidated = (current_pos + &correction)?;

    Ok(MicroDreamResult {
        consolidated,
        correction_norm: detection_norm,
        reflection_triggered,
    })
}
```

---

## src/field.rs

Lines: 428

```rust
#![allow(dead_code)]
//! Continuous Diderot Field
//!
//! The field is a sum of Gaussian kernels over all stored memory positions.
//! Dimension-agnostic: auto-detects D from the safetensors file.
//! `probe_gradient(pos)` returns the gradient vector — the ridge-running force.

use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

pub struct ContinuousField {
    /// Memory positions from embeddings, shape (N, D)
    pub positions: Tensor,
    pub device: Device,
    /// Controls the width of each Gaussian kernel (auto-tuned)
    pub kernel_sigma: f32,
    /// Embedding dimension (auto-detected from data)
    pub dim: usize,
}

impl ContinuousField {
    /// Load real embeddings from a safetensors file.
    /// Dimension-agnostic: auto-detects D and tunes sigma.
    pub fn load_real(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        println!("    Loading: {}", path.display());

        let tensors = candle_core::safetensors::load(path, device)?;

        // Print available keys
        let keys: Vec<_> = tensors.keys().collect();
        println!("    Keys found: {:?}", keys);

        // Try common key names, or take the largest tensor
        let positions = if let Some(t) = tensors.get("embeddings") {
            t.clone()
        } else if let Some(t) = tensors.get("tensor") {
            t.clone()
        } else if let Some(t) = tensors.get("weight") {
            t.clone()
        } else {
            tensors
                .values()
                .max_by_key(|t| t.elem_count())
                .expect("safetensors file is empty")
                .clone()
        };

        let positions = positions.to_dtype(DType::F32)?;
        let dim = positions.dim(positions.dims().len() - 1)?;
        let n = positions.dim(0)?;

        // Auto-tune sigma from actual mean pairwise distance.
        // Sample up to 200 random pairs and compute mean L2 distance,
        // then set sigma = mean_dist * 0.5 so Gaussian kernels overlap.
        let sigma = if n >= 2 {
            let n_pairs = 200usize.min(n * (n - 1) / 2);
            let mut total_dist = 0.0f64;
            let mut rng = 0u64; // simple LCG for deterministic sampling
            for _ in 0..n_pairs {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let i = (rng >> 33) as usize % n;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut j = (rng >> 33) as usize % (n - 1);
                if j >= i {
                    j += 1;
                }
                let pi = positions.get(i)?;
                let pj = positions.get(j)?;
                let diff = (&pi - &pj)?;
                let dist: f32 = diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                total_dist += dist as f64;
            }
            let mean_dist = (total_dist / n_pairs as f64) as f32;
            let s = if mean_dist > 1.0 {
                mean_dist * 0.5
            } else {
                // Fallback for degenerate data
                (dim as f32).sqrt() * 0.035
            };
            println!(
                "    Sigma auto-tuned: mean_dist={:.2}, sigma={:.4}",
                mean_dist, s
            );
            s
        } else {
            (dim as f32).sqrt() * 0.035
        };

        println!(
            "    Field loaded: {} points x {} dims | sigma = {:.4}",
            n, dim, sigma
        );

        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: sigma,
            dim,
        })
    }

    /// Build the field directly from a model's token embedding matrix.
    /// This is the preferred path: no external files, guaranteed alignment
    /// with the actual model, and no risk of all-zero placeholder data.
    ///
    /// `embeddings` should be shape (vocab_size, hidden_dim) -- the raw
    /// `tok_embeddings` tensor from the loaded ModelWeights.
    pub fn from_embeddings(embeddings: &Tensor, device: &Device) -> Result<Self> {
        let positions = embeddings.to_dtype(DType::F32)?.to_device(device)?;
        let dim = positions.dim(positions.dims().len() - 1)?;
        let n = positions.dim(0)?;

        println!("    Building Diderot field from model tok_embeddings...");
        println!("    Shape: {} tokens x {} dims", n, dim);

        // Auto-tune sigma from sampled pairwise distances
        let sigma = if n >= 2 {
            let n_pairs = 200usize.min(n * (n - 1) / 2);
            let mut total_dist = 0.0f64;
            let mut rng = 42u64; // deterministic LCG
            for _ in 0..n_pairs {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let i = (rng >> 33) as usize % n;
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut j = (rng >> 33) as usize % (n - 1);
                if j >= i {
                    j += 1;
                }
                let pi = positions.get(i)?;
                let pj = positions.get(j)?;
                let diff = (&pi - &pj)?;
                let dist: f32 = diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                total_dist += dist as f64;
            }
            let mean_dist = (total_dist / n_pairs as f64) as f32;
            let s = if mean_dist > 1.0 {
                mean_dist * 0.5
            } else {
                // L2-normalized embeddings: typical dist ~ sqrt(2)
                // Fallback for degenerate data
                (dim as f32).sqrt() * 0.035
            };
            println!(
                "    Sigma auto-tuned: mean_dist={:.2}, sigma={:.4}",
                mean_dist, s
            );
            s
        } else {
            (dim as f32).sqrt() * 0.035
        };

        println!(
            "    Field LIVE: {} points x {} dims | sigma = {:.4}",
            n, dim, sigma
        );

        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: sigma,
            dim,
        })
    }

    /// Load dummy random embeddings (for testing).
    #[allow(dead_code)]
    pub fn load_dummy(dim: usize, n_points: usize, device: &Device) -> Result<Self> {
        let positions = Tensor::randn(0.0f32, 1.0, (n_points, dim), device)?;
        let sigma = (dim as f32).sqrt() * 0.035;
        Ok(Self {
            positions,
            device: device.clone(),
            kernel_sigma: sigma,
            dim,
        })
    }

    /// Probe the scalar density at a position.
    pub fn probe(&self, pos: &Tensor) -> Result<Tensor> {
        let pos_expanded = pos.unsqueeze(0)?;
        let diff = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;
        kernel.sum_all()
    }

    /// Compute the gradient of the density field at a position.
    /// NaN-safe: returns zero gradient when all kernels underflow (fast path).
    pub fn probe_gradient(&self, pos: &Tensor) -> Result<Tensor> {
        let pos_expanded = pos.unsqueeze(0)?;
        let diff = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient (fast path)
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), DType::F32, &self.device);
        }

        let kernel_expanded = kernel.unsqueeze(1)?;
        let weighted = diff.broadcast_mul(&kernel_expanded)?;
        let scale = 2.0 / sigma_sq as f64;
        let grad = weighted.sum(0)?.squeeze(0)?.affine(scale, 0.0)?;
        Ok(grad)
    }

    /// Find the K nearest field point indices (= token IDs) to a position.
    /// Returns Vec of (index, cosine_similarity) sorted by similarity descending.
    /// This is cheap for small K since we compute all distances then partial-sort.
    pub fn nearest_tokens(&self, pos: &Tensor, k: usize) -> anyhow::Result<Vec<(u32, f32)>> {
        // pos shape: (D,) -- broadcast sub against positions (N, D)
        let pos_expanded = pos.unsqueeze(0)?;
        let diff = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq: Vec<f32> = diff.sqr()?.sum(1)?.to_vec1()?;

        // Compute cosine similarities for ranking
        // cos_sim(a, b) = dot(a,b) / (|a| * |b|)
        let pos_flat: Vec<f32> = pos.to_vec1()?;
        let pos_norm: f32 = pos_flat.iter().map(|x| x * x).sum::<f32>().sqrt();

        let positions_flat: Vec<f32> = self.positions.flatten_all()?.to_vec1()?;
        let n = dist_sq.len();
        let dim = self.dim;

        // Build (index, neg_dist_sq) pairs and partial sort for top-K
        let mut indices: Vec<(usize, f32)> =
            dist_sq.iter().enumerate().map(|(i, &d)| (i, d)).collect();
        // Partial sort: put K smallest dist_sq at front
        let k = k.min(n);
        if k == 0 || indices.is_empty() {
            return Ok(Vec::new());
        }
        indices.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k);

        // Compute cosine similarity for the K nearest
        let mut results: Vec<(u32, f32)> = indices
            .iter()
            .map(|&(idx, _)| {
                let row = &positions_flat[idx * dim..(idx + 1) * dim];
                let dot: f32 = row.iter().zip(pos_flat.iter()).map(|(a, b)| a * b).sum();
                let row_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos_sim = if pos_norm > 1e-12 && row_norm > 1e-12 {
                    dot / (pos_norm * row_norm)
                } else {
                    0.0
                };
                (idx as u32, cos_sim)
            })
            .collect();

        // Sort by cosine similarity descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Compute the gradient using only the K nearest field points (approximate).
    ///
    /// Instead of evaluating all N field points (O(N*D)), only the K nearest
    /// by L2 distance are used. For N=128K and K=2048, this is ~60x faster.
    /// Uses partial sort (select_nth_unstable) to find K nearest in O(N) time.
    ///
    /// Falls back to exact gradient if K >= N.
    pub fn probe_gradient_topk(&self, pos: &Tensor, k: usize) -> Result<Tensor> {
        let n = self.n_points();
        if k >= n || n == 0 {
            return self.probe_gradient(pos);
        }

        // Compute squared distances to all field points
        let pos_expanded = pos.unsqueeze(0)?;
        let diff_all = self.positions.broadcast_sub(&pos_expanded)?;
        let dist_sq_all: Vec<f32> = diff_all.sqr()?.sum(1)?.to_vec1()?;

        // Partial sort to find K nearest indices
        let mut indexed: Vec<(usize, f32)> = dist_sq_all
            .iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Gather only the K nearest positions
        let topk_indices: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
        let topk_rows: Vec<Tensor> = topk_indices
            .iter()
            .map(|&i| self.positions.get(i).and_then(|r| r.unsqueeze(0)))
            .collect::<Result<Vec<_>>>()?;
        let topk_positions = Tensor::cat(&topk_rows, 0)?;

        // Standard gradient computation on just the K nearest
        let diff = topk_positions.broadcast_sub(&pos_expanded)?;
        let dist_sq = diff.sqr()?.sum(1)?;
        let sigma_sq = self.kernel_sigma * self.kernel_sigma;
        let kernel = (dist_sq.neg()? / sigma_sq as f64)?.exp()?;

        // Safety: if all kernels underflow, return zero gradient
        let kernel_sum: f32 = kernel.sum_all()?.to_scalar()?;
        if kernel_sum.abs() < 1e-30 || kernel_sum.is_nan() {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, &self.device);
        }

        let kernel_expanded = kernel.unsqueeze(1)?;
        let weighted = diff.broadcast_mul(&kernel_expanded)?;
        let scale = 2.0 / sigma_sq as f64;
        let grad = weighted.sum(0)?.squeeze(0)?.affine(scale, 0.0)?;
        Ok(grad)
    }

    pub fn n_points(&self) -> usize {
        self.positions.dim(0).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field(positions: Tensor, sigma: f32, dim: usize) -> ContinuousField {
        ContinuousField {
            device: positions.device().clone(),
            positions,
            kernel_sigma: sigma,
            dim,
        }
    }

    #[test]
    fn gradient_pulls_toward_field_point() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let grad = field.probe_gradient(&pos).unwrap();
        let grad_vec: Vec<f32> = grad.to_vec1().unwrap();

        // diff = positions - query = [0,0,0,0] - [1,0,0,0] = [-1,0,0,0]
        // gradient should be negative x (toward origin from x=1)
        assert!(
            grad_vec[0] < 0.0,
            "expected negative x gradient, got {}",
            grad_vec[0]
        );
    }

    #[test]
    fn density_positive_at_field_point() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        let density: f32 = field.probe(&pos).unwrap().to_scalar().unwrap();
        assert!(
            density > 0.0,
            "density at field point should be > 0, got {}",
            density
        );
    }

    #[test]
    fn gradient_zero_far_away() {
        let device = Device::Cpu;
        let positions = Tensor::zeros(&[1, 4], DType::F32, &device).unwrap();
        let field = make_field(positions, 0.1, 4);

        let pos = Tensor::new(&[1000.0f32, 1000.0, 1000.0, 1000.0], &device).unwrap();
        let grad = field.probe_gradient(&pos).unwrap();
        let mag: f32 = grad
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            mag < 1e-10,
            "gradient far from field should be ~0, got {}",
            mag
        );
    }

    #[test]
    fn topk_gradient_matches_exact_for_small_n() {
        // With N=5 points and K=5, Top-K should produce the exact same result
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[5, 4], &device).unwrap();
        let field = make_field(positions, 1.0, 4);

        let pos = Tensor::new(&[0.5f32, -0.3, 0.7, 0.1], &device).unwrap();
        let exact = field.probe_gradient(&pos).unwrap();
        let topk = field.probe_gradient_topk(&pos, 5).unwrap();

        let diff: f32 = (&exact - &topk)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();

        assert!(
            diff < 1e-5,
            "Top-K (K=N) should match exact gradient, diff={}",
            diff
        );
    }
}
```

---

## src/gpu.rs

Lines: 857

```rust
#![allow(dead_code)]
//! Physics Backend Abstraction
//!
//! Defines the `PhysicsBackend` trait for GPU-accelerated physics computations.
//! Provides a `CpuBackend` that wraps the existing field/memory code verbatim,
//! and a `MetalBackend` (behind `metal-compute` feature) with wgpu WGSL compute shaders.
//!
//! All call sites (niodoo, dream, ridge) use `&dyn PhysicsBackend` so the GPU
//! path is a drop-in replacement with zero API changes.

use candle_core::{Result, Tensor};

use crate::field::ContinuousField;
use crate::memory::SplatMemory;

// ---------------------------------------------------------------
// Trait
// ---------------------------------------------------------------

/// Abstraction over CPU vs GPU physics computations.
///
/// Each method corresponds to a hot-path operation in the steering loop.
/// Implementations must produce identical results (within floating-point tolerance).
pub trait PhysicsBackend {
    /// Gradient of the continuous field density at `pos` (shape `(D,)`).
    /// Replaces `ContinuousField::probe_gradient`.
    fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor>;

    /// Aggregate Gaussian force from all splats at `pos` (shape `(D,)`).
    /// Replaces `SplatMemory::query_force`.
    fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor>;

    /// Batch field gradient for multiple positions (shape `(M, D)` -> `(M, D)`).
    /// Used by micro-dream forward projection and ridge-running ensembles.
    fn batch_field_gradient(&self, field: &ContinuousField, positions: &Tensor) -> Result<Tensor>;

    /// Name string for telemetry logging.
    fn name(&self) -> &'static str;

    /// Top-K approximate gradient: only consider the K nearest field points.
    /// Default implementation falls back to exact gradient.
    fn field_gradient_topk(
        &self,
        field: &ContinuousField,
        pos: &Tensor,
        k: usize,
    ) -> Result<Tensor> {
        let _ = k; // unused in default -- exact gradient
        self.field_gradient(field, pos)
    }
}

// ---------------------------------------------------------------
// CPU Backend (wraps existing code verbatim)
// ---------------------------------------------------------------

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicsBackend for CpuBackend {
    fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor> {
        field.probe_gradient(pos)
    }

    fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor> {
        memory.query_force(pos)
    }

    fn batch_field_gradient(&self, field: &ContinuousField, positions: &Tensor) -> Result<Tensor> {
        let m = positions.dim(0)?;
        if m == 0 {
            let d = positions.dim(1)?;
            return Tensor::zeros(&[0, d], candle_core::DType::F32, &field.device);
        }
        let mut grads = Vec::with_capacity(m);
        for i in 0..m {
            let pos_i = positions.get(i)?;
            let grad_i = field.probe_gradient(&pos_i)?.unsqueeze(0)?;
            grads.push(grad_i);
        }
        Tensor::cat(&grads, 0)
    }

    fn name(&self) -> &'static str {
        "CPU"
    }

    fn field_gradient_topk(
        &self,
        field: &ContinuousField,
        pos: &Tensor,
        k: usize,
    ) -> Result<Tensor> {
        field.probe_gradient_topk(pos, k)
    }
}

// ---------------------------------------------------------------
// Metal Backend (behind feature flag)
// ---------------------------------------------------------------

#[cfg(feature = "metal-compute")]
mod metal_backend {
    use super::*;

    /// WGSL compute shader: field gradient.
    ///
    /// Each workgroup computes the gradient contribution from all field points
    /// for one query position. Uses padded vec4 layout for Metal alignment.
    ///
    /// The shader processes D dimensions in chunks of 4 (vec4<f32>).
    /// For D=4096, that's 1024 vec4 elements per position.
    const FIELD_GRADIENT_SHADER: &str = r#"
// Uniforms: dimensions and kernel parameters
// Padded to 32 bytes (8 x u32) for 16-byte WGSL uniform alignment.
struct Params {
    n_points: u32,      // number of field points
    n_queries: u32,     // number of query positions
    dim_vec4: u32,      // D / 4 (number of vec4 elements per position)
    sigma_sq: f32,      // kernel_sigma^2
    scale: f32,         // 2.0 / sigma_sq
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Field positions: flat array of vec4, layout [n_points * dim_vec4]
@group(0) @binding(1) var<storage, read> field_positions: array<vec4<f32>>;
// Query positions: flat array of vec4, layout [n_queries * dim_vec4]
@group(0) @binding(2) var<storage, read> query_positions: array<vec4<f32>>;
// Output gradients: flat array of vec4, layout [n_queries * dim_vec4]
@group(0) @binding(3) var<storage, read_write> gradients: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    if query_idx >= params.n_queries {
        return;
    }

    let q_offset = query_idx * params.dim_vec4;

    // For each field point, compute Gaussian kernel weight and accumulate gradient
    for (var p = 0u; p < params.n_points; p++) {
        let p_offset = p * params.dim_vec4;

        // Compute ||field_pos - query_pos||^2
        var dist_sq = 0.0;
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = field_positions[p_offset + d] - query_positions[q_offset + d];
            dist_sq += dot(diff, diff);
        }

        // Gaussian kernel: exp(-dist_sq / sigma_sq)
        let kernel = exp(-dist_sq / params.sigma_sq);

        // Skip if kernel underflows
        if kernel < 1e-30 {
            continue;
        }

        // Accumulate weighted gradient: kernel * (field_pos - query_pos) * scale
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = field_positions[p_offset + d] - query_positions[q_offset + d];
            gradients[q_offset + d] += diff * kernel * params.scale;
        }
    }
}
"#;

    /// WGSL compute shader: splat force aggregation.
    ///
    /// Each invocation processes one query position against all splats.
    /// Splats are stored as: mu (vec4 * dim_vec4), sigma (f32), alpha (f32).
    const SPLAT_FORCE_SHADER: &str = r#"
struct Params {
    n_splats: u32,
    n_queries: u32,
    dim_vec4: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Splat mu positions: flat [n_splats * dim_vec4] vec4s
@group(0) @binding(1) var<storage, read> splat_mu: array<vec4<f32>>;
// Splat sigma values: [n_splats]
@group(0) @binding(2) var<storage, read> splat_sigma: array<f32>;
// Splat alpha values: [n_splats]
@group(0) @binding(3) var<storage, read> splat_alpha: array<f32>;
// Query positions: [n_queries * dim_vec4] vec4s
@group(0) @binding(4) var<storage, read> query_positions: array<vec4<f32>>;
// Output forces: [n_queries * dim_vec4] vec4s
@group(0) @binding(5) var<storage, read_write> forces: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    if query_idx >= params.n_queries {
        return;
    }

    let q_offset = query_idx * params.dim_vec4;

    for (var s = 0u; s < params.n_splats; s++) {
        let s_offset = s * params.dim_vec4;
        let sigma = splat_sigma[s];
        let alpha = splat_alpha[s];
        let sigma_sq = sigma * sigma;

        // Compute ||mu - pos||^2
        var dist_sq = 0.0;
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = splat_mu[s_offset + d] - query_positions[q_offset + d];
            dist_sq += dot(diff, diff);
        }

        // kernel = exp(-dist_sq / sigma_sq)
        let kernel = exp(-dist_sq / sigma_sq);
        let scale = alpha * kernel;

        // force += scale * (mu - pos)
        for (var d = 0u; d < params.dim_vec4; d++) {
            let diff = splat_mu[s_offset + d] - query_positions[q_offset + d];
            forces[q_offset + d] += diff * scale;
        }
    }
}
"#;

    pub struct MetalBackend {
        device: wgpu::Device,
        queue: wgpu::Queue,
        field_gradient_pipeline: wgpu::ComputePipeline,
        splat_force_pipeline: wgpu::ComputePipeline,
    }

    impl MetalBackend {
        pub fn try_new() -> Option<Self> {
            // Block on async wgpu init -- fine at startup
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..Default::default()
            });

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hydrodynamic-swarm-metal"),
                    ..Default::default()
                },
                None,
            ))
            .ok()?;

            // Compile shaders
            let field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("field_gradient_shader"),
                source: wgpu::ShaderSource::Wgsl(FIELD_GRADIENT_SHADER.into()),
            });

            let splat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("splat_force_shader"),
                source: wgpu::ShaderSource::Wgsl(SPLAT_FORCE_SHADER.into()),
            });

            let field_gradient_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("field_gradient_pipeline"),
                    layout: None, // auto layout
                    module: &field_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let splat_force_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("splat_force_pipeline"),
                    layout: None,
                    module: &splat_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            println!("    [METAL] Adapter: {}", adapter.get_info().name);

            Some(Self {
                device,
                queue,
                field_gradient_pipeline,
                splat_force_pipeline,
            })
        }

        /// Run the field gradient compute shader for `n_queries` query positions.
        fn dispatch_field_gradient(
            &self,
            field_data: &[f32], // flat [N * D] field positions
            query_data: &[f32], // flat [M * D] query positions
            n_points: u32,
            n_queries: u32,
            dim: u32,
            sigma_sq: f32,
        ) -> Vec<f32> {
            use wgpu::util::DeviceExt;

            let dim_vec4 = dim / 4;
            let scale = 2.0f32 / sigma_sq;

            // Uniform params buffer (padded to 32 bytes to match WGSL struct)
            let params = [
                n_points,
                n_queries,
                dim_vec4,
                sigma_sq.to_bits(),
                scale.to_bits(),
                0u32, // _pad1
                0u32, // _pad2
                0u32, // _pad3
            ];
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // Field positions buffer
            let field_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("field_positions"),
                    contents: bytemuck::cast_slice(field_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // Query positions buffer
            let query_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("query_positions"),
                    contents: bytemuck::cast_slice(query_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // Output gradient buffer (must be zeroed -- shader uses +=)
            let output_size = (n_queries * dim) as u64 * 4;
            let zeros = vec![0u8; output_size as usize];
            let output_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gradients"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            // Staging buffer for readback
            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Bind group
            let bind_group_layout = self.field_gradient_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field_gradient_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: field_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: query_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            // Encode and submit
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("field_gradient_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("field_gradient_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.field_gradient_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n_queries.div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            // Read back results
            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                if let Err(e) = res {
                    eprintln!("[METAL] field_gradient staging buffer map failed: {e}");
                }
            });
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();

            result
        }

        /// Run the splat force compute shader.
        #[allow(clippy::too_many_arguments)]
        fn dispatch_splat_force(
            &self,
            splat_mu_data: &[f32],    // flat [S * D]
            splat_sigma_data: &[f32], // [S]
            splat_alpha_data: &[f32], // [S]
            query_data: &[f32],       // flat [M * D]
            n_splats: u32,
            n_queries: u32,
            dim: u32,
        ) -> Vec<f32> {
            use wgpu::util::DeviceExt;

            let dim_vec4 = dim / 4;

            // Uniform params
            let params = [n_splats, n_queries, dim_vec4, 0u32]; // _pad = 0
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let mu_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_mu"),
                    contents: bytemuck::cast_slice(splat_mu_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let sigma_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_sigma"),
                    contents: bytemuck::cast_slice(splat_sigma_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let alpha_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("splat_alpha"),
                    contents: bytemuck::cast_slice(splat_alpha_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let query_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("query"),
                    contents: bytemuck::cast_slice(query_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let output_size = (n_queries * dim) as u64 * 4;
            let zeros = vec![0u8; output_size as usize];
            let output_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("forces"),
                    contents: &zeros,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout = self.splat_force_pipeline.get_bind_group_layout(0);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat_force_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mu_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sigma_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: alpha_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: query_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("splat_force_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_force_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.splat_force_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n_queries.div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |res| {
                if let Err(e) = res {
                    eprintln!("[METAL] splat_force staging buffer map failed: {e}");
                }
            });
            self.device.poll(wgpu::Maintain::Wait);

            let data = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();

            result
        }

        /// Convert a Candle Tensor to a flat Vec<f32> for GPU upload.
        fn tensor_to_f32(t: &Tensor) -> Result<Vec<f32>> {
            t.flatten_all()?.to_vec1()
        }

        /// Pad a flat f32 slice to be vec4-aligned (D must be divisible by 4).
        /// For D=4096, this is already aligned. This is a safety check.
        fn ensure_vec4_aligned(dim: usize) -> bool {
            dim.is_multiple_of(4)
        }
    }

    impl PhysicsBackend for MetalBackend {
        fn field_gradient(&self, field: &ContinuousField, pos: &Tensor) -> Result<Tensor> {
            let dim = field.dim;

            // Fall back to CPU if dim is not vec4-aligned
            if !Self::ensure_vec4_aligned(dim) {
                return field.probe_gradient(pos);
            }

            let field_data = Self::tensor_to_f32(&field.positions)?;
            let query_data = Self::tensor_to_f32(pos)?;
            let n_points = field.n_points() as u32;

            let result = self.dispatch_field_gradient(
                &field_data,
                &query_data,
                n_points,
                1,
                dim as u32,
                field.kernel_sigma * field.kernel_sigma,
            );

            // Check for all-zero (underflow case)
            let sum: f32 = result.iter().map(|x| x.abs()).sum();
            if sum < 1e-30 {
                return candle_core::Tensor::zeros(&[dim], candle_core::DType::F32, &field.device);
            }

            Tensor::from_vec(result, dim, &field.device)
        }

        fn splat_force(&self, memory: &SplatMemory, pos: &Tensor) -> Result<Tensor> {
            let splats = memory.splats_ref();
            if splats.is_empty() {
                let dims = pos.dims().to_vec();
                return Tensor::zeros(&dims[..], candle_core::DType::F32, pos.device());
            }

            let dim = pos.dims()[0];
            if !Self::ensure_vec4_aligned(dim) {
                return memory.query_force(pos);
            }

            // Extract splat data
            let n_splats = splats.len();
            let mut mu_data = Vec::with_capacity(n_splats * dim);
            let mut sigma_data = Vec::with_capacity(n_splats);
            let mut alpha_data = Vec::with_capacity(n_splats);

            for splat in splats {
                let mu_flat: Vec<f32> = splat.mu.flatten_all()?.to_vec1()?;
                mu_data.extend_from_slice(&mu_flat);
                sigma_data.push(splat.sigma);
                alpha_data.push(splat.alpha);
            }

            let query_data = Self::tensor_to_f32(pos)?;

            let result = self.dispatch_splat_force(
                &mu_data,
                &sigma_data,
                &alpha_data,
                &query_data,
                n_splats as u32,
                1,
                dim as u32,
            );

            Tensor::from_vec(result, dim, pos.device())
        }

        fn batch_field_gradient(
            &self,
            field: &ContinuousField,
            positions: &Tensor,
        ) -> Result<Tensor> {
            let dim = field.dim;

            if !Self::ensure_vec4_aligned(dim) {
                // Fall back to serial CPU
                let m = positions.dim(0)?;
                if m == 0 {
                    return Tensor::zeros(&[0, dim], candle_core::DType::F32, &field.device);
                }
                let mut grads = Vec::with_capacity(m);
                for i in 0..m {
                    let pos_i = positions.get(i)?;
                    let grad_i = field.probe_gradient(&pos_i)?.unsqueeze(0)?;
                    grads.push(grad_i);
                }
                return Tensor::cat(&grads, 0);
            }

            let n_queries = positions.dim(0)? as u32;
            let field_data = Self::tensor_to_f32(&field.positions)?;
            let query_data = Self::tensor_to_f32(positions)?;

            let result = self.dispatch_field_gradient(
                &field_data,
                &query_data,
                field.n_points() as u32,
                n_queries,
                dim as u32,
                field.kernel_sigma * field.kernel_sigma,
            );

            Tensor::from_vec(result, (n_queries as usize, dim), &field.device)
        }

        fn name(&self) -> &'static str {
            "Metal Compute"
        }
    }
}

// ---------------------------------------------------------------
// Backend selection helper
// ---------------------------------------------------------------

/// Select the best available physics backend at runtime.
///
/// With `metal-compute` feature: tries MetalBackend first, falls back to CPU.
/// Without feature: always returns CpuBackend.
pub fn select_backend() -> Box<dyn PhysicsBackend> {
    #[cfg(feature = "metal-compute")]
    {
        if let Some(metal) = metal_backend::MetalBackend::try_new() {
            println!("[*] Physics backend: Metal Compute (wgpu)");
            return Box::new(metal);
        }
        println!("[*] Metal compute init failed, falling back to CPU physics");
    }

    println!("[*] Physics backend: CPU");
    Box::new(CpuBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn cpu_backend_field_gradient_matches_direct() {
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[10, 4], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim: 4,
        };

        let backend = CpuBackend::new();
        let pos = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], &device).unwrap();

        let direct = field.probe_gradient(&pos).unwrap();
        let via_backend = backend.field_gradient(&field, &pos).unwrap();

        let diff: f32 = (&direct - &via_backend)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            diff < 1e-6,
            "CPU backend should match direct, diff={}",
            diff
        );
    }

    #[test]
    fn cpu_backend_splat_force_matches_direct() {
        let device = Device::Cpu;
        let mut memory = crate::memory::SplatMemory::new(device.clone());
        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(crate::splat::Splat::new(mu, 1.0, 3.0));

        let backend = CpuBackend::new();
        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();

        let direct = memory.query_force(&pos).unwrap();
        let via_backend = backend.splat_force(&memory, &pos).unwrap();

        let diff: f32 = (&direct - &via_backend)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            diff < 1e-6,
            "CPU backend splat force should match direct, diff={}",
            diff
        );
    }

    #[test]
    fn cpu_backend_batch_gradient_shape() {
        let device = Device::Cpu;
        let positions = Tensor::randn(0.0f32, 1.0, &[5, 4], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim: 4,
        };

        let backend = CpuBackend::new();
        let queries = Tensor::randn(0.0f32, 1.0, &[3, 4], &device).unwrap();
        let result = backend.batch_field_gradient(&field, &queries).unwrap();
        assert_eq!(result.dims(), &[3, 4]);
    }

    /// GPU parity test -- requires `metal-compute` feature and available GPU.
    /// Runs only when compiled with `cargo test --features metal-compute`.
    #[test]
    #[cfg(feature = "metal-compute")]
    fn metal_cpu_field_gradient_parity() {
        let device = Device::Cpu;
        let dim = 8; // small dim, vec4-aligned
        let positions = Tensor::randn(0.0f32, 1.0, &[20, dim], &device).unwrap();
        let field = crate::field::ContinuousField {
            device: device.clone(),
            positions,
            kernel_sigma: 1.0,
            dim,
        };

        let cpu = CpuBackend::new();
        let metal = match metal_backend::MetalBackend::try_new() {
            Some(m) => m,
            None => {
                eprintln!("    [SKIP] Metal GPU not available for parity test");
                return;
            }
        };

        let pos = Tensor::randn(0.0f32, 1.0, &[dim], &device).unwrap();
        let cpu_result = cpu.field_gradient(&field, &pos).unwrap();
        let metal_result = metal.field_gradient(&field, &pos).unwrap();

        let diff: f32 = (&cpu_result - &metal_result)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();

        // Allow some FP tolerance since GPU uses f32 with different instruction ordering
        assert!(
            diff < 0.1,
            "CPU vs Metal field gradient should match within tolerance, diff={}",
            diff
        );
    }
}
```

---

## src/llama.rs

Lines: 531

```rust
//! Vendored quantized Llama model -- from candle-transformers 0.8.4.
//!
//! Vendored to expose `forward_hidden()` and `forward_with_hidden()` which
//! return the pre-lm_head hidden state for physics steering.
//!
//! Original: candle-transformers/src/models/quantized_llama.rs

use std::collections::HashMap;

use candle_core::quantized::QTensor;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

pub const MAX_SEQ_LEN: usize = 4096;

// QMatMul wrapper adding some tracing.
#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
                    }
                }

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
                    let selected_rws =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws)?;
                    ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let y = if q.device().is_metal() && seq_len == 1 {
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    #[allow(dead_code)]
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000., &ct.device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &ct.device)?;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(&ct.device)?;
        let norm = RmsNorm::from_qtensor(ct.remove("norm.weight")?, 1e-5)?;
        let output = ct.remove("output.weight")?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("{prefix}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let mlp_or_moe = {
                let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
                let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
                let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            };
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, 1e-5)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_embd as usize),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;
            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Run the transformer layers and return the final hidden state
    /// (post-norm, last position, pre-lm_head).
    fn run_layers(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        // Take last position only -- narrow to length-1, squeeze kills the seq dim
        // Guarantees (b_sz, hidden_dim) -- no phantom seq dim leaking into steering
        x.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }

    /// Standard forward: returns logits (vocab-sized).
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.run_layers(x, index_pos)?;
        let _enter = self.span_output.enter();
        self.output.forward(&hidden)
    }

    /// Return only the hidden state (D-dimensional, pre-lm_head).
    /// Same compute as forward() minus the final projection.
    #[allow(dead_code)]
    pub fn forward_hidden(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_layers(x, index_pos)
    }

    /// Return both logits AND hidden state in one pass (no wasted compute).
    /// Returns (logits, hidden_state).
    pub fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.run_layers(x, index_pos)?;
        let _enter = self.span_output.enter();
        let logits = self.output.forward(&hidden)?;
        Ok((logits, hidden))
    }

    /// Project a hidden state through the lm_head to get logits.
    /// Useful for projecting steered hidden states back to vocab space.
    pub fn project_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    /// Access the raw token embedding matrix (vocab_size, hidden_dim).
    /// Used to build the live Diderot field from model weights.
    pub fn token_embeddings(&self) -> &Tensor {
        self.tok_embeddings.embeddings()
    }
}
```

---

## src/logger.rs

Lines: 249

```rust
#![allow(dead_code)]
//! JSONL Telemetry Logger
//!
//! Logs every generation step and session summary to `logs/` as JSONL.
//! Each session file: `logs/{date}_{time}_{label}.jsonl`
//!
//! v2 note: when we unify splat memory across prompts (Emergent Synthesis),
//! the logger will track which domain each splat originated from,
//! enabling cross-domain influence analysis.

use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Per-step telemetry entry
#[derive(Serialize)]
pub struct StepEntry {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub steering_delta: f32,
    pub residual_norm: f32,
    pub grad_force_mag: f32,
    pub splat_force_mag: f32,
    pub goal_force_mag: f32,
}

/// Session config snapshot
#[derive(Serialize)]
pub struct SessionConfig {
    pub prompt: String,
    pub dt: f32,
    pub viscosity: f32,
    pub kernel_sigma: f32,
    pub embedding_dim: usize,
    pub field_points: usize,
    pub model: String,
    pub model_variant: String,
    pub backend: String,
    pub splat_sigma: f32,
    pub splat_alpha: f32,
    pub force_cap: f32,
    pub temperature: f32,
    pub min_splat_dist: f32,
}

/// Final session summary
#[derive(Serialize)]
pub struct SessionSummary {
    pub prompt: String,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub goal_attractor_norm: f32,
    pub splat_count_before: usize,
    pub splat_count_after: usize,
    pub splat_type_added: String, // "pleasure", "pain", or "none"
    pub decoded_output: String,
    pub delta_min: f32,
    pub delta_max: f32,
    pub delta_mean: f32,
}

/// Top-level log entry -- one per line in the JSONL file
#[derive(Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub session_id: String,
    pub model_variant: String,
    pub entry_type: String, // "config", "step", "summary"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<SessionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<StepEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SessionSummary>,
}

pub struct SessionLogger {
    file: fs::File,
    session_id: String,
    model_variant: String,
    deltas: Vec<f32>,
    log_path: PathBuf,
}

impl SessionLogger {
    /// Create a new session logger with a descriptive label.
    /// Filename: `logs/{YYYY-MM-DD}_{HH-MM-SS}_{label}.jsonl`
    pub fn new(label: &str, model_variant: &str) -> std::io::Result<Self> {
        let log_dir = Path::new("logs");
        fs::create_dir_all(log_dir)?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Manual UTC date/time formatting (no chrono dependency)
        let secs_per_day: u64 = 86400;
        let days = now / secs_per_day;
        let day_secs = now % secs_per_day;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;

        // Compute year/month/day from days since epoch
        let (year, month, day) = days_to_date(days);

        let date_str = format!("{:04}-{:02}-{:02}", year, month, day);
        let time_str = format!("{:02}-{:02}-{:02}", hours, minutes, seconds);

        // Sanitize label for filename
        let safe_label: String = label
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();

        let session_id = format!("{}_{}_{}", date_str, time_str, safe_label);
        let filename = format!("{}_{}_{}.jsonl", date_str, time_str, safe_label);
        let log_path = log_dir.join(&filename);
        let file = fs::File::create(&log_path)?;

        println!("    Logging to: {}", log_path.display());

        Ok(Self {
            file,
            session_id,
            model_variant: model_variant.to_string(),
            deltas: Vec::new(),
            log_path,
        })
    }

    /// Log the session config
    pub fn log_config(&mut self, config: SessionConfig) -> std::io::Result<()> {
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            model_variant: self.model_variant.clone(),
            entry_type: "config".to_string(),
            config: Some(config),
            step: None,
            summary: None,
        };
        self.write_entry(&entry)
    }

    /// Log a single generation step
    pub fn log_step(&mut self, step: StepEntry) -> std::io::Result<()> {
        self.deltas.push(step.steering_delta);
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            model_variant: self.model_variant.clone(),
            entry_type: "step".to_string(),
            config: None,
            step: Some(step),
            summary: None,
        };
        self.write_entry(&entry)
    }

    /// Log final session summary
    pub fn log_summary(&mut self, mut summary: SessionSummary) -> std::io::Result<()> {
        // Compute delta stats from collected deltas
        if !self.deltas.is_empty() {
            summary.delta_min = self.deltas.iter().cloned().fold(f32::INFINITY, f32::min);
            summary.delta_max = self
                .deltas
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            summary.delta_mean = self.deltas.iter().sum::<f32>() / self.deltas.len() as f32;
        }
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            model_variant: self.model_variant.clone(),
            entry_type: "summary".to_string(),
            config: None,
            step: None,
            summary: Some(summary),
        };
        self.write_entry(&entry)
    }

    /// Get the log file path
    pub fn path(&self) -> PathBuf {
        self.log_path.clone()
    }

    /// Get the session ID string
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn write_entry(&mut self, entry: &LogEntry) -> std::io::Result<()> {
        let json = serde_json::to_string(entry).unwrap();
        writeln!(self.file, "{}", json)?;
        self.file.flush()
    }

    fn now_str(&self) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("{}", secs)
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(mut days: u64) -> (u64, u64, u64) {
    // Simplified Gregorian calendar calculation
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &months_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}
```

---

## src/main.rs

Lines: 844

```rust
//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Full Llama 3.1 + Niodoo physics steering with real tokenization.
//! Type a prompt → physics steers generation → decoded text output.

mod config;
mod dream;
mod field;
mod gpu;
mod llama;
mod logger;
mod memory;
mod niodoo;
mod ridge;
mod splat;
mod tui;
mod viz;
mod viz_metal;

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use config::Config;
use dream::{micro_dream, DreamEngine};
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

fn main() -> Result<()> {
    println!("=== SplatRAG v1 -- Hydrodynamic Swarm ===\n");

    // Load configuration (falls back to defaults if no config.toml)
    let cfg = Config::load(Path::new("config.toml")).unwrap_or_else(|e| {
        eprintln!("    [CONFIG] {}, using defaults", e);
        Config::default()
    });

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let clear_memory = args.iter().any(|a| a == "--clear-memory");
    let test_mode = args.iter().any(|a| a == "--test");
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
        .unwrap_or(cfg.generation.max_tokens);
    let viz_enabled = args.iter().any(|a| a == "--viz");
    let chat_mode = args.iter().any(|a| a == "--chat");

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
    // Phase 1: Load Llama 3.1 GGUF + Tokenizer
    // =========================================================
    println!("\n--- Phase 1: Loading Llama 3.1 + Tokenizer ---");

    // Find GGUF model
    let llama_path = find_file(
        "data/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "/Users/j/Desktop/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    )?;
    println!("    Model: {}", llama_path);

    let mut file = std::fs::File::open(&llama_path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;
    let mut llama = llama::ModelWeights::from_gguf(ct, &mut reader, &device)?;
    println!("    Llama 3.1 loaded");

    // Find tokenizer
    let tokenizer_path = find_file(
        "data/tokenizer.json",
        "/Users/j/Desktop/models/tokenizer.json",
    )?;
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    println!("    Tokenizer loaded ({})", tokenizer_path);

    // =========================================================
    // Phase 2: Build live Diderot field from model embeddings
    // =========================================================
    println!("\n--- Phase 2: Building Diderot Field ---");
    let field = ContinuousField::from_embeddings(llama.token_embeddings(), &device)?;
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
        return tui::run_chat(
            &mut llama,
            &tokenizer,
            &mut engine,
            &device,
            dim,
            max_tokens,
            &cfg,
        );
    }

    // Initialize telemetry logger
    let model_variant = cli_model.as_deref().unwrap_or("unsloth");
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
        model: "Llama-3.1-8B-Instruct-Q5_K_M".to_string(),
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
        let (logits, hidden) = llama.forward_with_hidden(&prompt_tensor, 0)?;
        (logits, Some(hidden))
    } else {
        let logits = llama.forward(&prompt_tensor, 0)?;
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
            steered: steered_slice,
            grad_mag,
            splat_mag,
            goal_mag,
        } = engine.steer(&steer_input, &goal_pos, step)?;
        last_steered_pos = Some(steered_slice.clone());

        // Micro-dream consolidation: adaptive frequency based on token entropy
        // Runs when entropy is high (uncertain generation) or on fixed schedule
        let steered_slice = if step > 10 {
            // Estimate entropy from raw logits (cheap: first 1000 logits only)
            let raw_probs_slice = candle_nn::ops::softmax(&raw_logits, 1)?;
            let raw_probs_flat: Vec<f32> = raw_probs_slice.squeeze(0)?.to_vec1()?;
            let sample_n = raw_probs_flat.len().min(1000);
            let entropy: f32 = raw_probs_flat[..sample_n]
                .iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum::<f32>()
                .max(0.0); // never negative/NaN

            // Temporarily disable micro-dreams for stability testing
            let should_dream = false;
            if should_dream {
                // Adaptive depth: higher entropy -> deeper projection
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
                    micro_dream(&engine, &steered_slice, &goal_pos, step, dream_steps, blend)?;
                if step <= 15 || step % 10 == 0 {
                    println!(
                        "    [MICRO-DREAM] step {} | correction: {:.2} | entropy: {:.2} | depth: {}{}",
                        step,
                        result.correction_norm,
                        entropy,
                        dream_steps,
                        if result.reflection_triggered {
                            " ** HYDRAULIC JUMP **"
                        } else {
                            ""
                        }
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

        // Reconstruct full logits for sampling
        let steered_logits = if is_hidden_steer {
            // Project steered hidden state through lm_head to get full vocab logits
            llama.project_to_logits(&steered_slice)?
        } else {
            // Logit-space steering: cat steered slice with remaining logits
            if raw_logits.dim(1)? > dim {
                let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
                Tensor::cat(&[&steered_slice, &rest], 1)?
            } else {
                steered_slice
            }
        };

        // Temperature sampling -- softmax over scaled logits, then sample
        let temperature: f64 = cfg.generation.temperature;
        let scaled_logits = (&steered_logits / temperature)?;
        let probs = candle_nn::ops::softmax(&scaled_logits, 1)?;
        let mut probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        // REPETITION PENALTY -- kills attractor trapping
        let repetition_penalty: f32 = 1.18;
        let recent: std::collections::HashSet<u32> =
            generated_tokens.iter().rev().take(32).cloned().collect();
        for (i, p) in probs_vec.iter_mut().enumerate() {
            if recent.contains(&(i as u32)) {
                *p = p.powf(1.0 / repetition_penalty);
            }
        }
        // Renormalize after penalty
        let prob_sum: f32 = probs_vec.iter().sum();
        if prob_sum > 0.0 {
            for p in probs_vec.iter_mut() {
                *p /= prob_sum;
            }
        }

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

        // Online splat creation (conditional)
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
            let (logits, hidden) = llama.forward_with_hidden(&next_input, index_pos)?;
            raw_logits = logits;
            raw_hidden = Some(hidden);
        } else {
            raw_logits = llama.forward(&next_input, index_pos)?;
            raw_hidden = None;
        }
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
                pos_1d,
                cfg.physics.splat_sigma,
                1.8, // positive scar (pleasure)
            ));
            println!(
                "    + Added PLEASURE splat (generation succeeded: {} tokens)",
                generated_tokens.len()
            );
        } else {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                -0.9, // negative scar (pain)
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

    // Save persistent splat memory to disk (before dream decay wipes them)
    engine.memory().save(splat_file)?;
    engine
        .memory()
        .save_metadata(splat_file, prompt, logger.session_id())?;

    // =========================================================
    // Memory Museum: Save to exhibit / Toss
    // =========================================================
    println!("\n--- Memory Museum ---");
    println!(
        "    Splats: {} | Source: \"{}\"",
        engine.memory().len(),
        prompt
    );

    // List existing exhibits
    let exhibits_dir = Path::new("exhibits");
    if exhibits_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(exhibits_dir) {
            let names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
                .filter_map(|e| {
                    e.path()
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect();
            if !names.is_empty() {
                println!("    Existing exhibits: {}", names.join(", "));
            }
        }
    }

    print!("    Save to exhibit? [name / n(ew) / t(oss)]: ");
    std::io::stdout().flush().ok();
    let museum_input = if test_mode {
        println!("(skipped -- test mode)");
        "t".to_string()
    } else {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input
    };
    let museum_input = museum_input.trim();

    if !museum_input.is_empty() && museum_input != "t" && museum_input != "toss" {
        let exhibit_name = if museum_input == "n" || museum_input == "new" {
            print!("    Exhibit name: ");
            std::io::stdout().flush().ok();
            let mut name = String::new();
            std::io::stdin().read_line(&mut name)?;
            name.trim().to_string()
        } else {
            museum_input.to_string()
        };

        if !exhibit_name.is_empty() {
            // Sanitize exhibit name for filesystem
            let safe_name: String = exhibit_name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();

            std::fs::create_dir_all(exhibits_dir)?;
            let exhibit_path = exhibits_dir.join(format!("{}.safetensors", safe_name));
            let exhibit_meta = exhibits_dir.join(format!("{}.meta.json", safe_name));

            std::fs::copy(splat_file, &exhibit_path)?;
            // Copy metadata sidecar if it exists
            let meta_src = splat_file.with_extension("meta.json");
            if meta_src.exists() {
                std::fs::copy(&meta_src, &exhibit_meta)?;
            }

            println!(
                "    Saved exhibit: {} ({} splats)",
                exhibit_path.display(),
                engine.memory().len()
            );
        } else {
            println!("    (empty name, skipping)");
        }
    } else {
        println!("    (tossed)");
    }

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
        let secs_per_day: u64 = 86400;
        let days = now / secs_per_day;
        let day_secs = now % secs_per_day;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        // Approximate date from Unix days
        let mut y = 1970i64;
        let mut remaining = days as i64;
        loop {
            let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                366
            } else {
                365
            };
            if remaining < days_in_year {
                break;
            }
            remaining -= days_in_year;
            y += 1;
        }
        let month_days = [
            31,
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                29
            } else {
                28
            },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        let mut m = 1;
        for md in &month_days {
            if remaining < *md as i64 {
                break;
            }
            remaining -= *md as i64;
            m += 1;
        }
        let d = remaining + 1;
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
    // Visualization export + Metal window
    // =========================================================
    if let Some(collector) = viz_collector {
        // Export JSON snapshot data
        let viz_path = logger.path().with_extension("viz.json");
        let _ = collector.export_json(&viz_path);

        // Launch Metal 3D window (does not return on macOS)
        let render_data = collector.into_render_data();
        viz_metal::launch(render_data);
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
```

---

## src/memory.rs

Lines: 619

```rust
#![allow(dead_code)]
//! SplatMemory — manages a collection of splats and computes aggregate forces.
//!
//! This is the "scar tissue" layer: accumulated experience that biases
//! the particle's trajectory through the field.
//! Pain lasts longer than pleasure (asymmetric decay).
//! Supports save/load to disk via safetensors for persistent memory.

use crate::splat::Splat;
use candle_core::{DType, Result, Tensor};
use std::path::Path;

pub struct SplatMemory {
    splats: Vec<Splat>,
    device: candle_core::Device,
}

impl SplatMemory {
    pub fn new(device: candle_core::Device) -> Self {
        Self {
            splats: Vec::new(),
            device,
        }
    }

    pub fn add_splat(&mut self, splat: Splat) {
        self.splats.push(splat);
    }

    /// Time-based exponential decay: V(t) = V0 * exp(-lambda * delta_t).
    /// Asymmetric: pain decays at 70% of the pleasure rate.
    /// Anchors (lambda=0 or is_anchor=true) never decay.
    /// `decay_rate` is the legacy per-step fallback for splats without lambda.
    pub fn decay_step(&mut self, decay_rate: f32) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for splat in &mut self.splats {
            // Anchors never decay
            if splat.is_anchor || splat.lambda == 0.0 {
                continue;
            }

            let dt = (now.saturating_sub(splat.created_at)) as f32;
            let effective_lambda = if splat.alpha < 0.0 {
                // Pain lasts longer: 70% decay rate
                splat.lambda * 0.7
            } else {
                splat.lambda
            };

            if dt > 0.0 {
                // Exponential decay: alpha *= exp(-lambda * dt)
                let decay_factor = (-effective_lambda * dt).exp();
                splat.alpha *= decay_factor;
            } else {
                // Fallback: per-step decay for freshly created splats
                if splat.alpha > 0.0 {
                    splat.alpha *= decay_rate;
                } else {
                    splat.alpha *= decay_rate * 0.7;
                }
            }
        }
    }

    /// Culling horizon: purge splats whose |alpha| has dropped below threshold.
    /// Keeps the memory file lean and prevents dead splats from wasting compute.
    /// Returns the number of splats culled.
    pub fn cull(&mut self, threshold: f32) -> usize {
        let before = self.splats.len();
        self.splats
            .retain(|s| s.is_anchor || s.alpha.abs() >= threshold);
        before - self.splats.len()
    }

    /// Core function: summed Gaussian pull/push from all nearby splats.
    ///
    /// For each splat: force = alpha * (mu - pos) * exp(-||mu - pos||^2 / sigma^2)
    /// Positive alpha pulls toward the splat (pleasure), negative pushes away (pain).
    pub fn query_force(&self, pos: &Tensor) -> Result<Tensor> {
        let dims = pos.dims().to_vec();
        let mut total_force = Tensor::zeros(&dims[..], DType::F32, &self.device)?;

        for splat in &self.splats {
            let diff = (&splat.mu - pos)?;
            let dist_sq: f32 = diff.sqr()?.sum_all()?.to_scalar()?;
            let sigma_sq = splat.sigma * splat.sigma;
            let kernel = (-dist_sq / sigma_sq).exp();
            let scale = (splat.alpha * kernel) as f64;
            let signed_force = diff.affine(scale, 0.0)?;
            total_force = (&total_force + &signed_force)?;
        }
        Ok(total_force)
    }

    /// Number of active splats.
    pub fn len(&self) -> usize {
        self.splats.len()
    }

    /// Read-only access to splat data (used by GPU backend for buffer upload).
    pub fn splats_ref(&self) -> &[Splat] {
        &self.splats
    }

    /// Check if any splat center is within min_dist of pos (L2).
    /// Samples at most 50 splats for performance when memory is large.
    pub fn has_nearby(&self, pos: &Tensor, min_dist: f32) -> Result<bool> {
        let min_dist_sq = min_dist * min_dist;
        let max_check = 50.min(self.splats.len());
        // Check last N splats (most recently added, most likely nearby)
        let start = self.splats.len().saturating_sub(max_check);
        for splat in &self.splats[start..] {
            let dist_sq: f32 = (&splat.mu - pos)?.sqr()?.sum_all()?.to_scalar()?;
            if dist_sq < min_dist_sq {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Prune dead splats below threshold.
    pub fn prune(&mut self, threshold: f32) {
        self.splats.retain(|s| s.alpha.abs() >= threshold);
    }

    /// Consolidate nearby splats with matching sign into single weighted splats.
    ///
    /// Greedy merge: for each splat, find all same-sign splats within `merge_dist`
    /// (L2 in embedding space). Replace the cluster with a single splat whose:
    /// - mu = weighted mean (by |alpha|)
    /// - sigma = max sigma in cluster (conservative width)
    /// - alpha = sum of alphas in cluster
    ///
    /// Returns the number of merges performed.
    pub fn consolidate(&mut self, merge_dist: f32) -> Result<usize> {
        if self.splats.len() < 2 {
            return Ok(0);
        }

        let merge_dist_sq = merge_dist * merge_dist;
        let mut merged = Vec::new();
        let mut consumed = vec![false; self.splats.len()];
        let mut merge_count = 0usize;

        for i in 0..self.splats.len() {
            if consumed[i] {
                continue;
            }

            let sign_i = self.splats[i].alpha >= 0.0;
            let mut cluster_mu = self.splats[i].mu.clone();
            let mut cluster_weight = self.splats[i].alpha.abs();
            let mut cluster_alpha = self.splats[i].alpha;
            let mut cluster_sigma = self.splats[i].sigma;
            let mut cluster_size = 1usize;

            // Find nearby same-sign splats
            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..self.splats.len() {
                if consumed[j] {
                    continue;
                }
                let sign_j = self.splats[j].alpha >= 0.0;
                if sign_i != sign_j {
                    continue;
                }
                let dist_sq: f32 = (&cluster_mu - &self.splats[j].mu)?
                    .sqr()?
                    .sum_all()?
                    .to_scalar()?;
                if dist_sq < merge_dist_sq {
                    // Weighted mean of mu (cluster_mu is current centroid)
                    let w_j = self.splats[j].alpha.abs();
                    let total_w = cluster_weight + w_j;
                    if total_w > 0.0 {
                        cluster_mu = (&cluster_mu
                            .affine((cluster_weight / total_w) as f64, 0.0)?
                            + &self.splats[j].mu.affine((w_j / total_w) as f64, 0.0)?)?;
                    }
                    cluster_weight = total_w;
                    cluster_alpha += self.splats[j].alpha;
                    cluster_sigma = cluster_sigma.max(self.splats[j].sigma);
                    cluster_size += 1;
                    consumed[j] = true;
                }
            }

            if cluster_size > 1 {
                merge_count += cluster_size - 1;
            }
            // Preserve the strongest splat's metadata for the merged result
            let is_anchor = self.splats[i].is_anchor;
            let scale = self.splats[i].scale;
            let lambda = if is_anchor {
                0.0
            } else {
                self.splats[i].lambda
            };
            merged.push(Splat {
                mu: cluster_mu,
                sigma: cluster_sigma,
                alpha: cluster_alpha,
                lambda,
                created_at: self.splats[i].created_at,
                scale,
                is_anchor,
            });
        }

        let old_count = self.splats.len();
        self.splats = merged;
        if merge_count > 0 {
            println!(
                "    [CONSOLIDATE] {} -> {} splats ({} merged)",
                old_count,
                self.splats.len(),
                merge_count
            );
        }
        Ok(merge_count)
    }

    /// Keep only the N strongest splats (by |alpha|), discarding the weakest.
    pub fn prune_to_limit(&mut self, max_count: usize) {
        if self.splats.len() <= max_count {
            return;
        }
        self.splats
            .sort_by(|a, b| b.alpha.abs().total_cmp(&a.alpha.abs()));
        self.splats.truncate(max_count);
        println!("    [PRUNE] Capped to {} strongest splats", max_count);
    }

    /// Save all splats to a safetensors file.
    /// Format: mu=(N,D), sigma=(N,), alpha=(N,), lambda=(N,), created_at=(N,), scale=(N,), is_anchor=(N,)
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if self.splats.is_empty() {
            println!("    No splats to save.");
            return Ok(());
        }

        let n = self.splats.len();

        // Stack mu tensors into one (N, D) tensor
        let mu_rows: Vec<Tensor> = self
            .splats
            .iter()
            .map(|s| s.mu.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?;
        let mu_stack = Tensor::cat(&mu_rows, 0)?;

        // Scalar fields
        let sigmas: Vec<f32> = self.splats.iter().map(|s| s.sigma).collect();
        let alphas: Vec<f32> = self.splats.iter().map(|s| s.alpha).collect();
        let lambdas: Vec<f32> = self.splats.iter().map(|s| s.lambda).collect();
        let created_ats: Vec<f32> = self.splats.iter().map(|s| s.created_at as f32).collect();
        let scales: Vec<f32> = self.splats.iter().map(|s| s.scale as u8 as f32).collect();
        let anchors: Vec<f32> = self
            .splats
            .iter()
            .map(|s| if s.is_anchor { 1.0 } else { 0.0 })
            .collect();

        let sigma_tensor = Tensor::from_vec(sigmas, n, &self.device)?;
        let alpha_tensor = Tensor::from_vec(alphas, n, &self.device)?;
        let lambda_tensor = Tensor::from_vec(lambdas, n, &self.device)?;
        let created_at_tensor = Tensor::from_vec(created_ats, n, &self.device)?;
        let scale_tensor = Tensor::from_vec(scales, n, &self.device)?;
        let anchor_tensor = Tensor::from_vec(anchors, n, &self.device)?;

        // Convert to raw bytes
        let mu_data: Vec<f32> = mu_stack.flatten_all()?.to_vec1()?;
        let sigma_data: Vec<f32> = sigma_tensor.to_vec1()?;
        let alpha_data: Vec<f32> = alpha_tensor.to_vec1()?;
        let lambda_data: Vec<f32> = lambda_tensor.to_vec1()?;
        let created_at_data: Vec<f32> = created_at_tensor.to_vec1()?;
        let scale_data: Vec<f32> = scale_tensor.to_vec1()?;
        let anchor_data: Vec<f32> = anchor_tensor.to_vec1()?;

        let to_bytes =
            |data: &[f32]| -> Vec<u8> { data.iter().flat_map(|f| f.to_le_bytes()).collect() };

        let mu_bytes = to_bytes(&mu_data);
        let sigma_bytes = to_bytes(&sigma_data);
        let alpha_bytes = to_bytes(&alpha_data);
        let lambda_bytes = to_bytes(&lambda_data);
        let created_at_bytes = to_bytes(&created_at_data);
        let scale_bytes = to_bytes(&scale_data);
        let anchor_bytes = to_bytes(&anchor_data);

        let mu_shape = mu_stack.dims().to_vec();
        let n_shape = vec![n];

        let mu_view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, mu_shape, &mu_bytes)?;
        let sigma_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &sigma_bytes,
        )?;
        let alpha_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &alpha_bytes,
        )?;
        let lambda_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &lambda_bytes,
        )?;
        let created_at_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &created_at_bytes,
        )?;
        let scale_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            n_shape.clone(),
            &scale_bytes,
        )?;
        let anchor_view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, n_shape, &anchor_bytes)?;

        let tensors: Vec<(String, safetensors::tensor::TensorView)> = vec![
            ("mu".to_string(), mu_view),
            ("sigma".to_string(), sigma_view),
            ("alpha".to_string(), alpha_view),
            ("lambda".to_string(), lambda_view),
            ("created_at".to_string(), created_at_view),
            ("scale".to_string(), scale_view),
            ("is_anchor".to_string(), anchor_view),
        ];

        safetensors::tensor::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            &None::<std::collections::HashMap<String, String>>,
            path,
        )?;

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!(
            "    Saved {} splats ({} anchors) to {}",
            n,
            anchor_count,
            path.display()
        );
        Ok(())
    }

    /// Load splats from a safetensors file. Appends to existing splats.
    /// Backward-compatible: loads v1 files (mu, sigma, alpha only) with defaults for new fields.
    pub fn load(&mut self, path: &Path) -> anyhow::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }

        let file_data = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_data)?;

        let mu_view = tensors.tensor("mu")?;
        let sigma_view = tensors.tensor("sigma")?;
        let alpha_view = tensors.tensor("alpha")?;

        let mu_shape = mu_view.shape().to_vec();
        let n = mu_shape[0];
        let d = mu_shape[1];

        // Parse raw bytes to f32
        let parse_f32 = |data: &[u8]| -> Vec<f32> {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        let mu_data = parse_f32(mu_view.data());
        let sigma_data = parse_f32(sigma_view.data());
        let alpha_data = parse_f32(alpha_view.data());

        // Optional v2 fields (backward-compatible)
        let lambda_data: Option<Vec<f32>> =
            tensors.tensor("lambda").ok().map(|v| parse_f32(v.data()));
        let created_at_data: Option<Vec<f32>> = tensors
            .tensor("created_at")
            .ok()
            .map(|v| parse_f32(v.data()));
        let scale_data: Option<Vec<f32>> =
            tensors.tensor("scale").ok().map(|v| parse_f32(v.data()));
        let anchor_data: Option<Vec<f32>> = tensors
            .tensor("is_anchor")
            .ok()
            .map(|v| parse_f32(v.data()));

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Reconstruct splats
        for i in 0..n {
            let mu_row = &mu_data[i * d..(i + 1) * d];
            let mu_tensor = Tensor::from_vec(mu_row.to_vec(), d, &self.device)?;

            let lambda = lambda_data.as_ref().map_or(0.02, |v| v[i]);
            let created_at = created_at_data.as_ref().map_or(now, |v| v[i] as u64);
            let scale = scale_data
                .as_ref()
                .map_or(crate::splat::SplatScale::Fine, |v| {
                    crate::splat::SplatScale::from_u8(v[i] as u8)
                });
            let is_anchor = anchor_data.as_ref().is_some_and(|v| v[i] > 0.5);

            self.splats.push(Splat {
                mu: mu_tensor,
                sigma: sigma_data[i],
                alpha: alpha_data[i],
                lambda,
                created_at,
                scale,
                is_anchor,
            });
        }

        let anchor_count = self.splats.iter().filter(|s| s.is_anchor).count();
        println!(
            "    Loaded {} splats ({} anchors) from {} (total: {})",
            n,
            anchor_count,
            path.display(),
            self.splats.len()
        );
        Ok(n)
    }

    /// Save metadata sidecar JSON alongside safetensors.
    /// Records source prompt, timestamp, splat count, and session info.
    pub fn save_metadata(
        &self,
        safetensors_path: &Path,
        prompt: &str,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let meta_path = safetensors_path.with_extension("meta.json");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let meta = serde_json::json!({
            "splat_count": self.splats.len(),
            "timestamp": now,
            "source_prompt": prompt,
            "session_id": session_id,
            "sigma_range": {
                "min": self.splats.iter().map(|s| s.sigma).fold(f32::INFINITY, f32::min),
                "max": self.splats.iter().map(|s| s.sigma).fold(f32::NEG_INFINITY, f32::max),
            },
            "alpha_range": {
                "min": self.splats.iter().map(|s| s.alpha).fold(f32::INFINITY, f32::min),
                "max": self.splats.iter().map(|s| s.alpha).fold(f32::NEG_INFINITY, f32::max),
            },
            "pleasure_count": self.splats.iter().filter(|s| s.alpha > 0.0).count(),
            "pain_count": self.splats.iter().filter(|s| s.alpha < 0.0).count(),
        });

        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        println!("    Saved splat metadata to {}", meta_path.display());
        Ok(())
    }

    /// Load and display metadata sidecar if it exists.
    pub fn load_metadata(safetensors_path: &Path) -> Option<serde_json::Value> {
        let meta_path = safetensors_path.with_extension("meta.json");
        if !meta_path.exists() {
            return None;
        }
        match std::fs::read_to_string(&meta_path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(val) => {
                    println!("    Loaded splat metadata from {}", meta_path.display());
                    Some(val)
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pleasure_splat_attracts() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(Splat::new(mu, 1.0, 5.0));

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let force_vec: Vec<f32> = force.to_vec1().unwrap();

        // force = 5.0 * ([0]-[1]) * kernel => negative x (pulls toward origin)
        assert!(
            force_vec[0] < 0.0,
            "pleasure should attract, got {}",
            force_vec[0]
        );
    }

    #[test]
    fn pain_splat_repels() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu = Tensor::zeros(&[4], DType::F32, &device).unwrap();
        memory.add_splat(Splat::new(mu, 1.0, -5.0));

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let force_vec: Vec<f32> = force.to_vec1().unwrap();

        assert!(
            force_vec[0] > 0.0,
            "pain should repel, got {}",
            force_vec[0]
        );
    }

    #[test]
    fn empty_memory_zero_force() {
        let device = candle_core::Device::Cpu;
        let memory = SplatMemory::new(device.clone());

        let pos = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = memory.query_force(&pos).unwrap();
        let mag: f32 = force
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(mag < 1e-10, "empty force should be 0, got {}", mag);
    }

    #[test]
    fn consolidation_merges_nearby_same_sign() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[0.1f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, 3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert!(merged > 0);
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn consolidation_preserves_distant() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[100.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, 3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert_eq!(merged, 0);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn consolidation_no_merge_opposite_signs() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        let mu1 = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu2 = Tensor::new(&[0.1f32, 0.0, 0.0, 0.0], &device).unwrap();
        memory.add_splat(Splat::new(mu1, 1.0, 2.0));
        memory.add_splat(Splat::new(mu2, 1.0, -3.0));

        let merged = memory.consolidate(1.0).unwrap();
        assert_eq!(merged, 0);
        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn prune_to_limit_keeps_strongest() {
        let device = candle_core::Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());

        for i in 0..10 {
            let mu = Tensor::new(&[i as f32, 0.0, 0.0, 0.0], &device).unwrap();
            memory.add_splat(Splat::new(mu, 1.0, (i + 1) as f32));
        }

        memory.prune_to_limit(5);
        assert_eq!(memory.len(), 5);
        for splat in memory.splats_ref() {
            assert!(
                splat.alpha >= 6.0,
                "should keep strongest, got alpha={}",
                splat.alpha
            );
        }
    }
}
```

---

## src/niodoo.rs

Lines: 202

```rust
//! Niodoo Physics Steering Engine
//!
//! The core steering function: apply physics forces to the LLM residual stream.
//! Three forces act on the token trajectory each step:
//!   1. Field gradient (ridge-running): pulls toward high-density regions of the
//!      continuous Diderot embedding field. Scaled by viscosity.
//!   2. Splat scar tissue: accumulated Gaussian pleasure/pain scars pull/push
//!      the trajectory based on past generation experience.
//!   3. Goal attractor: linear pull toward the prompt's semantic goal position.
//!
//! The combined force is clamped per-element (force cap) to prevent runaway,
//! then scaled by dt and added to the residual.

use crate::field::ContinuousField;
use crate::gpu::PhysicsBackend;
use crate::memory::SplatMemory;
use candle_core::{Result, Tensor};

/// Result of a single steering step, including force telemetry.
pub struct SteerResult {
    pub steered: Tensor,
    pub grad_mag: f32,
    pub splat_mag: f32,
    pub goal_mag: f32,
}

pub struct NiodooEngine {
    field: ContinuousField,
    memory: SplatMemory,
    backend: Box<dyn PhysicsBackend>,
    dt: f32,
    viscosity_scale: f32,
    force_cap: f32,
    gradient_topk: usize,
}

impl NiodooEngine {
    pub fn new(
        field: ContinuousField,
        memory: SplatMemory,
        backend: Box<dyn PhysicsBackend>,
        dt: f32,
        viscosity_scale: f32,
        force_cap: f32,
    ) -> Self {
        Self {
            field,
            memory,
            backend,
            dt,
            viscosity_scale,
            force_cap,
            gradient_topk: 0, // 0 = exact gradient (default)
        }
    }

    /// Set the Top-K gradient approximation parameter.
    /// 0 = exact gradient, >0 = use K nearest field points.
    pub fn set_gradient_topk(&mut self, k: usize) {
        self.gradient_topk = k;
    }

    /// Core steering: apply physics to LLM residual stream.
    ///
    /// `baseline_residual` must be shape `(1, D)` -- single-batch residual.
    /// Returns the steered residual with the same shape `(1, D)`.
    ///
    /// steered = baseline + dt * (grad_force * viscosity + splat_force + goal_force)
    pub fn steer(
        &self,
        baseline_residual: &Tensor,
        goal_pos: &Tensor,
        _step: usize,
    ) -> Result<SteerResult> {
        // Shape validation: require exactly (1, D)
        let dims = baseline_residual.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "steer: baseline_residual must be 2D (batch, dim), got {}D shape {:?}",
                dims.len(),
                dims
            )));
        }
        if dims[0] != 1 {
            return Err(candle_core::Error::Msg(format!(
                "steer: baseline_residual batch size must be 1, got {} (shape {:?}). \
                 Multi-batch steering is not supported in v1.",
                dims[0], dims
            )));
        }

        // Extract position vector: (1, D) -> (D,)
        let pos = baseline_residual.squeeze(0)?;

        // === SCALE NORMALIZATION — hidden state (norm ~140) -> unit norm (matches field) ===
        let pos_norm: f32 = pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-6);
        let pos_unit = pos.affine(1.0 / pos_norm as f64, 0.0)?;

        // 1. Field gradient: ridge-running force (via backend, with optional Top-K)
        //    Query in unit-norm space to match field embeddings
        let raw_grad = if self.gradient_topk > 0 {
            self.backend
                .field_gradient_topk(&self.field, &pos_unit, self.gradient_topk)?
        } else {
            self.backend.field_gradient(&self.field, &pos_unit)?
        };
        let grad_force = raw_grad.affine(self.viscosity_scale as f64, 0.0)?;

        // 2. Splat scar tissue force (via backend, also in unit-norm space)
        let splat_force = self.backend.splat_force(&self.memory, &pos_unit)?;

        // 3. Goal attractor (operates in original hidden-state space)
        let goal_force = (goal_pos - &pos)?;

        // Force telemetry: capture magnitudes for JSONL logging
        let splat_mag: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let grad_mag: f32 = grad_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let goal_mag: f32 = goal_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Sum and scale by dt
        let total_force = ((&grad_force + &splat_force)? + &goal_force)?;
        // Force cap: prevent any single dimension from dominating (Variant 3)
        let total_force = total_force.clamp(-self.force_cap, self.force_cap)?;
        let steering = total_force.affine(self.dt as f64, 0.0)?;

        // Restore batch dim: (D,) -> (1, D) and add to baseline
        let steering_2d = steering.unsqueeze(0)?;
        let steered = (baseline_residual + &steering_2d)?;

        // === MANIFOLD LOCK — CLIP FIRST, THEN RESCALE ===
        let baseline_norm: f32 = baseline_residual
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-6);

        // 1. Hard clip outlier dimensions first
        let steered = steered.clamp(-8.0, 8.0)?;

        // 2. Recompute norm after clipping
        let steered_norm: f32 = steered
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-6);

        // 3. Rescale to target norm (keeps direction, fixes magnitude)
        let target_norm = baseline_norm.clamp(130.0, 150.0);
        let steered = steered.affine((target_norm / steered_norm) as f64, 0.0)?;

        Ok(SteerResult {
            steered,
            grad_mag,
            splat_mag,
            goal_mag,
        })
    }

    /// Get a reference to the field for external access (viz, etc.).
    #[allow(dead_code)]
    pub fn field(&self) -> &ContinuousField {
        &self.field
    }

    /// Get a reference to the memory for external queries.
    pub fn memory(&self) -> &SplatMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory for splat insertion.
    pub fn memory_mut(&mut self) -> &mut SplatMemory {
        &mut self.memory
    }

    /// Get a reference to the field's embedding positions for visualization.
    pub fn field_positions(&self) -> &Tensor {
        &self.field.positions
    }

    /// Get the embedding dimension.
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.field.dim
    }

    /// Get the physics backend name for telemetry.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Get the field's kernel sigma for telemetry logging.
    pub fn field_kernel_sigma(&self) -> f32 {
        self.field.kernel_sigma
    }

    /// Get the number of field points for telemetry logging.
    pub fn field_n_points(&self) -> usize {
        self.field.n_points()
    }
}
```

---

## src/ridge.rs

Lines: 221

```rust
#![allow(dead_code)]
//! Ridge-Running Loop
//!
//! A query particle slides down the continuous Diderot field,
//! steered by splat scar tissue, until it settles on a stable ridge.
//! This proves the physics works before we touch the LLM.

use crate::field::ContinuousField;
use crate::gpu::PhysicsBackend;
use crate::memory::SplatMemory;
use candle_core::{DType, Result, Tensor};

/// A particle that rides the field gradient toward high-density ridges.
pub struct QueryParticle {
    pub pos: Tensor,
    pub vel: Tensor,
    pub mass: f32,
}

impl QueryParticle {
    pub fn new(start_pos: Tensor) -> Result<Self> {
        let dims = start_pos.dims().to_vec();
        let device = start_pos.device().clone();
        let vel = Tensor::zeros(&dims[..], DType::F32, &device)?;
        Ok(Self {
            pos: start_pos,
            vel,
            mass: 1.0,
        })
    }

    /// L2 norm of current velocity.
    pub fn speed(&self) -> Result<f32> {
        let v2: f32 = self.vel.sqr()?.sum_all()?.to_scalar()?;
        Ok(v2.sqrt())
    }

    /// L2 norm of current position.
    pub fn pos_norm(&self) -> Result<f32> {
        let p2: f32 = self.pos.sqr()?.sum_all()?.to_scalar()?;
        Ok(p2.sqrt())
    }
}

/// Drives a particle through the field until it settles on a ridge.
pub struct RidgeRunner<'a> {
    field: &'a ContinuousField,
    splat_memory: &'a SplatMemory,
    backend: &'a dyn PhysicsBackend,
    dt: f32,
    viscosity_scale: f32,
    damping: f32,
    goal_pos: Tensor,
}

impl<'a> RidgeRunner<'a> {
    pub fn new(
        field: &'a ContinuousField,
        splat_memory: &'a SplatMemory,
        backend: &'a dyn PhysicsBackend,
        goal_pos: Tensor,
    ) -> Self {
        Self {
            field,
            splat_memory,
            backend,
            dt: 0.01,
            viscosity_scale: 0.5,
            damping: 0.95,
            goal_pos,
        }
    }

    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_viscosity(mut self, v: f32) -> Self {
        self.viscosity_scale = v;
        self
    }

    pub fn with_damping(mut self, d: f32) -> Self {
        self.damping = d;
        self
    }

    /// The core ridge-running loop.
    ///
    /// The particle integrates forces from:
    /// 1. Field gradient (ridge-running force) -- via PhysicsBackend
    /// 2. Splat memory (scar tissue pull/push) -- via PhysicsBackend
    /// 3. Goal attractor (prompt embedding)
    ///
    /// Stops when velocity drops below threshold (settled on ridge)
    /// or max_steps reached.
    pub fn run(
        &self,
        mut particle: QueryParticle,
        max_steps: usize,
        settle_threshold: f32,
    ) -> Result<(QueryParticle, RunStats)> {
        let mut stats = RunStats::default();

        for step in 0..max_steps {
            // 1. Field gradient: the ridge-running force (via backend)
            let grad_force = self
                .backend
                .field_gradient(self.field, &particle.pos)?
                .affine(self.viscosity_scale as f64, 0.0)?;

            // 2. Splat scar tissue force (via backend)
            let splat_force = self.backend.splat_force(self.splat_memory, &particle.pos)?;

            // 3. Goal attractor: pull toward prompt embedding
            let goal_force = (&self.goal_pos - &particle.pos)?;

            // Sum all forces
            let total_force = ((&grad_force + &splat_force)? + &goal_force)?;

            // Euler integration: a = F/m, v += a*dt, x += v*dt
            let accel = total_force.affine(1.0 / particle.mass as f64, 0.0)?;
            let dv = accel.affine(self.dt as f64, 0.0)?;
            particle.vel = (&particle.vel + &dv)?;

            // Apply damping to prevent runaway
            particle.vel = particle.vel.affine(self.damping as f64, 0.0)?;

            let dx = particle.vel.affine(self.dt as f64, 0.0)?;
            particle.pos = (&particle.pos + &dx)?;

            // Track stats
            let speed = particle.speed()?;
            let density: f32 = self.field.probe(&particle.pos)?.to_scalar()?;

            stats.steps = step + 1;
            stats.final_speed = speed;
            stats.final_density = density;

            // Log every 20 steps
            if step % 20 == 0 {
                let pos_norm = particle.pos_norm()?;
                println!(
                    "    step {:>4} | speed: {:.6} | density: {:.6e} | pos_norm: {:.4}",
                    step, speed, density, pos_norm,
                );
            }

            // Settled on ridge?
            if speed < settle_threshold {
                stats.settled = true;
                println!(
                    "    -> Particle settled on ridge after {} steps (speed={:.6})",
                    step, speed
                );
                break;
            }
        }

        if !stats.settled {
            println!(
                "    -> Max steps reached ({}) (speed={:.6})",
                max_steps, stats.final_speed
            );
        }

        Ok((particle, stats))
    }

    /// Simplified ridge loop for testing splat memory forces.
    #[allow(dead_code)]
    pub fn run_with_memory(
        &self,
        mut particle: QueryParticle,
        max_steps: usize,
    ) -> Result<QueryParticle> {
        for step in 0..max_steps {
            let grad_force = self
                .backend
                .field_gradient(self.field, &particle.pos)?
                .affine(self.viscosity_scale as f64, 0.0)?;
            let splat_force = self.backend.splat_force(self.splat_memory, &particle.pos)?;
            let goal_force = (&self.goal_pos - &particle.pos)?;

            let total_force = ((&grad_force + &splat_force)? + &goal_force)?;

            // Euler integration: a = F/m, v += a*dt, x += v*dt
            let accel = total_force.affine(1.0 / particle.mass as f64, 0.0)?;
            let dv = accel.affine(self.dt as f64, 0.0)?;
            particle.vel = (&particle.vel + &dv)?;
            let dx = particle.vel.affine(self.dt as f64, 0.0)?;
            particle.pos = (&particle.pos + &dx)?;

            let vel_norm = particle.speed()?;
            if vel_norm < 0.001 {
                println!("    Particle settled on ridge after {} steps", step);
                break;
            }

            if step % 30 == 0 {
                println!(
                    "    step {:>4} | speed: {:.6} | pos_norm: {:.4}",
                    step,
                    vel_norm,
                    particle.pos_norm()?
                );
            }
        }
        Ok(particle)
    }
}

/// Statistics from a ridge-running session.
#[derive(Debug, Default)]
pub struct RunStats {
    pub steps: usize,
    pub settled: bool,
    pub final_speed: f32,
    pub final_density: f32,
}
```

---

## src/splat.rs

Lines: 121

```rust
#![allow(dead_code)]
//! Gaussian Splat -- individual memory unit.
//!
//! Each splat has:
//! - mu (position in embedding space)
//! - sigma (isotropic covariance -- scalar for v1)
//! - alpha (signed opacity: positive = pleasure, negative = pain/trauma)
//! - lambda (decay rate: 0 = anchor, higher = faster evaporation)
//! - created_at (epoch secs for time-based decay)
//! - scale (hierarchical: 0=fine, 1=medium, 2=coarse)
//! - is_anchor (true = core fact, never decays)

use candle_core::Tensor;

/// Hierarchical splat scale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SplatScale {
    /// Fine-grain: precise scars from small steering deltas
    Fine = 0,
    /// Medium: moderate steering events
    Medium = 1,
    /// Coarse: broad memories from large trajectory warps
    Coarse = 2,
}

impl SplatScale {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Fine,
            1 => Self::Medium,
            _ => Self::Coarse,
        }
    }

    /// Suggested sigma multiplier for this scale.
    pub fn sigma_multiplier(self) -> f32 {
        match self {
            Self::Fine => 1.0,
            Self::Medium => 2.0,
            Self::Coarse => 4.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Splat {
    /// Position in embedding space (D,)
    pub mu: Tensor,
    /// Isotropic covariance (scalar for v1)
    pub sigma: f32,
    /// Signed opacity: + = pleasure, - = pain/trauma
    pub alpha: f32,
    /// Decay rate: controls evaporation speed.
    /// 0.0 = anchor (never decays), default ~0.02 for normal splats.
    /// Pain splats use 70% of this rate (asymmetric decay).
    pub lambda: f32,
    /// Creation timestamp (seconds since Unix epoch).
    /// Used for time-based decay: V(t) = V0 * exp(-lambda * dt).
    pub created_at: u64,
    /// Hierarchical scale (fine/medium/coarse).
    pub scale: SplatScale,
    /// Anchor splat: core fact, lambda forced to 0.
    pub is_anchor: bool,
}

impl Splat {
    /// Create a standard (non-anchor) splat with default lambda.
    pub fn new(mu: Tensor, sigma: f32, alpha: f32) -> Self {
        Self {
            mu,
            sigma,
            alpha,
            lambda: 0.02,
            created_at: now_secs(),
            scale: SplatScale::Fine,
            is_anchor: false,
        }
    }

    /// Create a splat with explicit scale based on steering delta magnitude.
    pub fn with_scale(mu: Tensor, sigma: f32, alpha: f32, delta_norm: f32) -> Self {
        let (scale, sigma_mult) = if delta_norm > 30.0 {
            (SplatScale::Coarse, SplatScale::Coarse.sigma_multiplier())
        } else if delta_norm > 20.0 {
            (SplatScale::Medium, SplatScale::Medium.sigma_multiplier())
        } else {
            (SplatScale::Fine, SplatScale::Fine.sigma_multiplier())
        };
        Self {
            mu,
            sigma: sigma * sigma_mult,
            alpha,
            lambda: 0.02,
            created_at: now_secs(),
            scale,
            is_anchor: false,
        }
    }

    /// Create an anchor splat (lambda=0, never decays).
    pub fn anchor(mu: Tensor, sigma: f32, alpha: f32) -> Self {
        Self {
            mu,
            sigma,
            alpha,
            lambda: 0.0,
            created_at: now_secs(),
            scale: SplatScale::Coarse,
            is_anchor: true,
        }
    }
}

/// Current time in seconds since Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
```

---

## src/tui.rs

Lines: 235

```rust
//! Chat-style TUI for one-shot prompt demonstration.
//!
//! Usage: `cargo run -- --chat`
//! Shows a styled prompt, takes user input, runs physics-steered generation
//! with live token streaming, then exits.

use crate::llama::ModelWeights;
use anyhow::Result;
use candle_core::Tensor;
use std::io::{self, Write};
use tokenizers::Tokenizer;

use crate::config::Config;
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
                || (entropy > cfg.micro_dream.entropy_threshold
                    && step % cfg.micro_dream.adaptive_interval == 0);
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
                let too_close = engine
                    .memory()
                    .has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
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
```

---

## src/viz.rs

Lines: 342

```rust
//! SplatLens -- Visualization Data Collector
//!
//! Captures per-step snapshots from the generation pipeline,
//! projects 4096D embedding vectors to 3D via random projection,
//! and exports session data as JSON for the Metal renderer or replay.
//!
//! This module is purely READ-ONLY: it never mutates the engine,
//! field, memory, or any generation state.

use candle_core::Tensor;
use serde::Serialize;
use std::path::Path;

// ---------------------------------------------------------------
// Data types
// ---------------------------------------------------------------

/// A single token neighbor with its model probability.
#[derive(Serialize, Clone)]
pub struct TokenNeighbor {
    pub token_id: u32,
    pub token_text: String,
    /// Softmax probability from the model (used for sizing/labeling, not projection).
    pub probability: f32,
    #[serde(skip_serializing_if = "is_zero_position")]
    pub position_3d: [f32; 3],
}

/// Returns true if a 3D position is all-zero (degenerate projection).
fn is_zero_position(pos: &[f32; 3]) -> bool {
    pos[0] == 0.0 && pos[1] == 0.0 && pos[2] == 0.0
}

/// Per-step visualization snapshot.
#[derive(Serialize, Clone)]
pub struct VizSnapshot {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub position_3d: [f32; 3],
    pub steering_delta: f32,
    pub neighbors: Vec<TokenNeighbor>,
}

/// Splat scar projected to 3D.
#[derive(Serialize, Clone)]
pub struct VizSplat {
    pub position_3d: [f32; 3],
    pub alpha: f32,
    pub sigma: f32,
}

/// Full visualization session -- JSON-serializable.
#[derive(Serialize)]
pub struct VizSession {
    pub prompt: String,
    pub embedding_dim: usize,
    pub snapshots: Vec<VizSnapshot>,
    pub field_points_3d: Vec<[f32; 3]>,
    pub splat_scars: Vec<VizSplat>,
    pub goal_position_3d: [f32; 3],
}

/// Lightweight render data passed to the Metal window.
pub struct VizRenderData {
    pub field_points_3d: Vec<[f32; 3]>,
    pub trajectory_3d: Vec<[f32; 3]>,
    pub trajectory_deltas: Vec<f32>,
    pub trajectory_tokens: Vec<String>,
    pub splat_positions_3d: Vec<[f32; 3]>,
    pub splat_alphas: Vec<f32>,
    pub goal_position_3d: [f32; 3],
    pub prompt: String,
    /// Per-step neighbor data: Vec of (step_index, neighbors)
    pub step_neighbors: Vec<Vec<StepNeighbor>>,
    /// Ridge ghost trail (predicted path from ridge runner)
    pub ridge_ghost: Vec<[f32; 3]>,
}

/// Neighbor data for a single step, ready for rendering.
pub struct StepNeighbor {
    pub token_text: String,
    pub probability: f32,
    pub position_3d: [f32; 3],
}

// ---------------------------------------------------------------
// Collector
// ---------------------------------------------------------------

/// Collects visualization data during generation.
/// Created once after the field and goal are available.
pub struct VizCollector {
    /// Random projection matrix, flat layout (D * 3)
    projection: Vec<f32>,
    dim: usize,
    /// Flat copy of field positions for neighbor projection
    field_positions_flat: Vec<f32>,
    snapshots: Vec<VizSnapshot>,
    field_points_3d: Vec<[f32; 3]>,
    goal_3d: [f32; 3],
    prompt: String,
    /// Ridge ghost trail points (projected to 3D)
    ridge_ghost: Vec<[f32; 3]>,
}

impl VizCollector {
    /// Create a new collector.
    ///
    /// Builds a deterministic random projection matrix (seed=42) and
    /// projects all field embedding positions to 3D. Subsamples if
    /// the field has more than 5000 points to keep rendering fast.
    pub fn new(
        field_positions: &Tensor, // (N, D)
        goal_pos: &Tensor,        // (D,)
        prompt: &str,
        dim: usize,
    ) -> anyhow::Result<Self> {
        // Deterministic random projection matrix (D x 3)
        // Simple LCG for reproducibility, no external RNG crate needed.
        let mut rng_state: u64 = 42;
        let scale = 1.0 / (dim as f32).sqrt();
        let mut projection = vec![0.0f32; dim * 3];
        for val in projection.iter_mut() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (rng_state >> 33) as f32 / (1u64 << 31) as f32;
            *val = (u + v - 1.0) * scale;
        }

        // Keep flat field positions for neighbor projection
        let field_positions_flat: Vec<f32> = field_positions.flatten_all()?.to_vec1()?;

        // Validate that field_positions_flat has the expected layout
        let n = field_positions.dim(0)?;
        let expected_len = n * dim;
        if field_positions_flat.len() != expected_len {
            return Err(anyhow::anyhow!(
                "[VIZ] field_positions_flat length mismatch: got {} expected {} (n={}, dim={}). \
                 Embeddings may not match the full vocabulary layout.",
                field_positions_flat.len(),
                expected_len,
                n,
                dim
            ));
        }

        // Project field positions (subsample large fields)
        let stride = if n > 5000 { n / 5000 } else { 1 };
        let mut field_points_3d = Vec::with_capacity(n / stride);
        for i in (0..n).step_by(stride) {
            let row = &field_positions_flat[i * dim..(i + 1) * dim];
            field_points_3d.push(project_vec(row, &projection, dim));
        }

        // Project goal
        let goal_flat: Vec<f32> = goal_pos.to_vec1()?;
        let goal_3d = project_vec(&goal_flat, &projection, dim);

        println!(
            "    [VIZ] Collector ready: {} field points projected to 3D (vocab={})",
            field_points_3d.len(),
            n
        );

        Ok(Self {
            projection,
            dim,
            field_positions_flat,
            snapshots: Vec::new(),
            field_points_3d,
            goal_3d,
            prompt: prompt.to_string(),
            ridge_ghost: Vec::new(),
        })
    }

    /// Capture a snapshot at the current generation step.
    /// `neighbors` is a list of (token_id, token_text, probability)
    /// where probability is the softmax model probability.
    pub fn snapshot(
        &mut self,
        step: usize,
        token_id: u32,
        token_text: &str,
        steered_pos: &Tensor, // (1, D) or (D,)
        steering_delta: f32,
        neighbors: Vec<(u32, String, f32)>,
    ) -> anyhow::Result<()> {
        let pos_flat: Vec<f32> = steered_pos.flatten_all()?.to_vec1()?;
        let pos_3d = project_vec(&pos_flat, &self.projection, self.dim);

        // Project neighbor positions to 3D using field embedding positions
        let neighbor_data: Vec<TokenNeighbor> = neighbors
            .into_iter()
            .filter_map(|(tid, text, prob)| {
                let idx = tid as usize;
                if idx * self.dim + self.dim <= self.field_positions_flat.len() {
                    let row = &self.field_positions_flat[idx * self.dim..(idx + 1) * self.dim];
                    let pos = project_vec(row, &self.projection, self.dim);
                    Some(TokenNeighbor {
                        token_id: tid,
                        token_text: text,
                        probability: prob,
                        position_3d: pos,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.snapshots.push(VizSnapshot {
            step,
            token_id,
            token_text: token_text.to_string(),
            position_3d: pos_3d,
            steering_delta,
            neighbors: neighbor_data,
        });

        Ok(())
    }

    /// Set the ridge ghost trail (projected from 4096D to 3D).
    #[allow(dead_code)]
    pub fn set_ridge_ghost(&mut self, positions: &[Vec<f32>]) {
        self.ridge_ghost = positions
            .iter()
            .map(|p| project_vec(p, &self.projection, self.dim))
            .collect();
    }

    /// Export all collected data to a JSON file.
    /// Detects degenerate all-zero field_points_3d and omits them with a warning.
    pub fn export_json(&self, path: &Path) -> anyhow::Result<()> {
        // Detect degenerate (all-zero) field_points_3d
        let field_points_degenerate = !self.field_points_3d.is_empty()
            && self
                .field_points_3d
                .iter()
                .all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0);

        if field_points_degenerate {
            eprintln!(
                "    [VIZ] WARNING: field_points_3d is all-zero (degenerate projection). \
                 Omitting from export. Check embedding data and projection matrix."
            );
        }

        let session = VizSession {
            prompt: self.prompt.clone(),
            embedding_dim: self.dim,
            snapshots: self.snapshots.clone(),
            field_points_3d: if field_points_degenerate {
                Vec::new()
            } else {
                self.field_points_3d.clone()
            },
            splat_scars: Vec::new(),
            goal_position_3d: self.goal_3d,
        };

        let json = serde_json::to_string_pretty(&session)?;
        std::fs::write(path, json)?;
        println!(
            "    [VIZ] Exported {} snapshots to {}",
            self.snapshots.len(),
            path.display()
        );
        Ok(())
    }

    /// Convert into render data for the Metal window.
    pub fn into_render_data(self) -> VizRenderData {
        let trajectory_3d: Vec<[f32; 3]> = self.snapshots.iter().map(|s| s.position_3d).collect();
        let trajectory_deltas: Vec<f32> = self.snapshots.iter().map(|s| s.steering_delta).collect();
        let trajectory_tokens: Vec<String> = self
            .snapshots
            .iter()
            .map(|s| s.token_text.clone())
            .collect();

        // Collect per-step neighbor data for rendering
        let step_neighbors: Vec<Vec<StepNeighbor>> = self
            .snapshots
            .iter()
            .map(|s| {
                s.neighbors
                    .iter()
                    .map(|n| StepNeighbor {
                        token_text: n.token_text.clone(),
                        probability: n.probability,
                        position_3d: n.position_3d,
                    })
                    .collect()
            })
            .collect();

        VizRenderData {
            field_points_3d: self.field_points_3d,
            trajectory_3d,
            trajectory_deltas,
            trajectory_tokens,
            splat_positions_3d: Vec::new(),
            splat_alphas: Vec::new(),
            goal_position_3d: self.goal_3d,
            prompt: self.prompt,
            step_neighbors,
            ridge_ghost: self.ridge_ghost,
        }
    }

    /// Number of snapshots collected so far.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }
}

// ---------------------------------------------------------------
// Projection utilities
// ---------------------------------------------------------------

/// Project a D-dimensional vector to 3D via the random projection matrix.
fn project_vec(vec: &[f32], projection: &[f32], dim: usize) -> [f32; 3] {
    let mut result = [0.0f32; 3];
    let len = dim.min(vec.len());
    for j in 0..3 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += vec[i] * projection[i * 3 + j];
        }
        result[j] = sum;
    }
    result
}
```

---

## src/viz_metal.rs

Lines: 697

```rust
//! SplatLens -- 3D Viewer Generator
//!
//! "Architectural Fluidity" palette -- deep space / ocean depths aesthetic.
//! Canvas 2D with software 3D projection, no WebGL needed.
//!
//! Trail: ice-blue cyan, thickness/opacity = confidence (inverse delta).
//! Jumps: white/gold marker dots at high-delta steps.
//! Attractors: soft gold/amber, subtle.  Splats: teal (pleasure) / rust (pain).
//! Token orbs: small, at real 3D positions with force-directed label avoidance.

use crate::viz::VizRenderData;
use std::path::Path;

/// Generate an HTML 3D viewer and open it in the default browser.
pub fn launch(data: VizRenderData) {
    let html_path = Path::new("logs").join("splatlens_viewer.html");

    let trajectory_js = points_to_js(&data.trajectory_3d);
    let deltas_js = floats_to_js(&data.trajectory_deltas);
    let field_js = points_to_js(&data.field_points_3d);
    let goal_js = format!(
        "[{},{},{}]",
        data.goal_position_3d[0], data.goal_position_3d[1], data.goal_position_3d[2]
    );
    let splats_js = points_to_js(&data.splat_positions_3d);
    let splat_alphas_js = floats_to_js(&data.splat_alphas);
    let prompt_escaped = data
        .prompt
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ");

    let tokens_js = {
        let strs: Vec<String> = data
            .trajectory_tokens
            .iter()
            .map(|t| {
                let escaped = t
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', " ")
                    .replace('\r', "");
                format!("\"{}\"", escaped)
            })
            .collect();
        format!("[{}]", strs.join(","))
    };

    let neighbors_js = {
        let step_strs: Vec<String> = data
            .step_neighbors
            .iter()
            .map(|step_n| {
                let n_strs: Vec<String> = step_n
                    .iter()
                    .map(|n| {
                        let text_escaped = n
                            .token_text
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', " ")
                            .replace('\r', "");
                        format!(
                            "{{t:\"{}\",s:{:.3},p:[{:.4},{:.4},{:.4}]}}",
                            text_escaped,
                            n.probability,
                            n.position_3d[0],
                            n.position_3d[1],
                            n.position_3d[2]
                        )
                    })
                    .collect();
                format!("[{}]", n_strs.join(","))
            })
            .collect();
        format!("[{}]", step_strs.join(","))
    };

    let ridge_js = points_to_js(&data.ridge_ghost);

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplatLens -- Token Field Visualizer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0F111A;overflow:hidden;font-family:'Inter','SF Pro',system-ui,sans-serif}}
canvas{{display:block;width:100vw;height:100vh}}
#hud{{
  position:fixed;top:16px;left:16px;color:#7B8CA8;font-size:12px;
  background:rgba(15,17,26,0.88);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(80,100,140,0.15);max-width:420px;
  backdrop-filter:blur(14px);pointer-events:auto;z-index:10;
}}
#hud h2{{color:#A0B4D0;font-size:14px;margin-bottom:4px;letter-spacing:1.5px;font-weight:600}}
#hud p{{color:#5A6A82;font-size:11px;line-height:1.5}}
.legend{{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:4px;font-size:10px;color:#5A6A82}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
#gen-text{{
  position:fixed;top:16px;right:16px;color:#A0B4D0;font-size:13px;
  background:rgba(15,17,26,0.88);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(80,100,140,0.15);max-width:380px;max-height:55vh;
  overflow-y:auto;backdrop-filter:blur(14px);z-index:10;
  line-height:1.7;word-wrap:break-word;font-family:'SF Mono',Monaco,Consolas,monospace;
}}
#gen-text .token-current{{color:#D4A44C;font-weight:bold}}
#controls{{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;
  background:rgba(15,17,26,0.9);padding:10px 18px;border-radius:12px;
  border:1px solid rgba(80,100,140,0.12);backdrop-filter:blur(14px);z-index:10;
}}
#controls button{{
  background:rgba(60,80,120,0.15);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:11px;padding:5px 10px;border-radius:6px;cursor:pointer;
  font-family:inherit;transition:all 0.15s;
}}
#controls button:hover{{background:rgba(60,80,120,0.3)}}
#controls button.active{{background:rgba(64,224,208,0.15);color:#40E0D0;border-color:rgba(64,224,208,0.3)}}
#controls button.off{{opacity:0.3;border-color:rgba(80,100,140,0.08)}}
.slider-group{{display:flex;align-items:center;gap:6px}}
.slider-group label{{color:#4A5A72;font-size:10px;white-space:nowrap}}
.ctrl-slider{{
  width:80px;height:3px;-webkit-appearance:none;appearance:none;
  background:rgba(60,80,120,0.25);border-radius:2px;outline:none;cursor:pointer;
}}
.ctrl-slider::-webkit-slider-thumb{{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:#40E0D0;border:2px solid #0F111A;cursor:pointer;
}}
#step-label{{color:#4A5A72;font-size:11px;min-width:55px;text-align:center}}
#speed-label{{color:#4A5A72;font-size:10px;min-width:30px;text-align:center}}
#settings-btn{{
  position:fixed;bottom:70px;right:20px;width:36px;height:36px;border-radius:50%;
  background:rgba(15,17,26,0.85);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:18px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;z-index:11;backdrop-filter:blur(14px);transition:all 0.15s;
}}
#settings-btn:hover{{background:rgba(60,80,120,0.3);color:#40E0D0}}
#settings-panel{{
  position:fixed;bottom:70px;right:64px;background:rgba(15,17,26,0.94);
  border:1px solid rgba(80,100,140,0.15);border-radius:12px;padding:16px 20px;
  backdrop-filter:blur(14px);z-index:11;display:none;min-width:220px;
}}
#settings-panel.open{{display:block}}
#settings-panel h3{{color:#A0B4D0;font-size:12px;margin:0 0 10px;letter-spacing:1px;font-weight:600}}
.setting-row{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.setting-row label{{color:#5A6A82;font-size:10px;min-width:60px}}
.setting-row .ctrl-slider{{width:90px}}
.setting-row span{{color:#4A5A72;font-size:10px;min-width:28px;text-align:right}}
.mode-btn{{
  background:rgba(60,80,120,0.15);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:10px;padding:4px 10px;border-radius:5px;cursor:pointer;
  font-family:inherit;transition:all 0.15s;
}}
.mode-btn:hover{{background:rgba(60,80,120,0.3)}}
.mode-btn.active{{background:rgba(64,224,208,0.15);color:#40E0D0;border-color:rgba(64,224,208,0.3)}}
#prompt-bar{{
  position:fixed;top:16px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:8px;z-index:12;
  background:rgba(15,17,26,0.92);padding:10px 16px;border-radius:14px;
  border:1px solid rgba(80,100,140,0.18);backdrop-filter:blur(16px);
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
}}
#prompt-bar label{{color:#5A6A82;font-size:11px;white-space:nowrap;letter-spacing:0.5px}}
#prompt-input{{
  background:rgba(30,35,50,0.6);border:1px solid rgba(80,100,140,0.15);
  color:#C8D8F0;font-size:14px;padding:8px 14px;border-radius:8px;
  outline:none;width:420px;font-family:'SF Mono',Monaco,Consolas,monospace;
  transition:border-color 0.2s;
}}
#prompt-input:focus{{border-color:rgba(64,224,208,0.4)}}
#generate-btn{{
  background:linear-gradient(135deg,rgba(64,224,208,0.2),rgba(64,224,208,0.08));
  border:1px solid rgba(64,224,208,0.35);color:#40E0D0;font-size:12px;
  padding:8px 18px;border-radius:8px;cursor:pointer;font-family:inherit;
  font-weight:600;letter-spacing:0.5px;transition:all 0.2s;
}}
#generate-btn:hover{{background:linear-gradient(135deg,rgba(64,224,208,0.35),rgba(64,224,208,0.15));box-shadow:0 0 16px rgba(64,224,208,0.2)}}
</style>
</head>
<body>
<div id="prompt-bar">
  <label>PROMPT</label>
  <input type="text" id="prompt-input" value="{prompt}" spellcheck="false">
  <button id="generate-btn">Generate</button>
</div>
<div id="hud" style="top:72px">
  <h2>SPLATLENS</h2>
  <p>"{prompt}"</p>
  <p>{traj_len} steps | {field_len} field pts</p>
  <div class="legend">
    <span><span class="dot" style="background:#40E0D0"></span>Trail</span>
    <span><span class="dot" style="background:#F0E6C8"></span>Jump</span>
    <span><span class="dot" style="background:#D4A44C"></span>Goal</span>
    <span><span class="dot" style="background:#1E2840"></span>Field</span>
    <span><span class="dot" style="background:#2DD4A8"></span>Pleasure</span>
    <span><span class="dot" style="background:#8B3A3A"></span>Pain</span>
    <span><span class="dot" style="background:#C49650"></span>Attractors</span>
  </div>
  <p style="margin-top:6px;color:#2A3448">Drag=orbit | Scroll=zoom | Shift+drag=pan</p>
</div>
<div id="gen-text"></div>
<div id="controls">
  <button id="btn-play" class="active">Play</button>
  <button id="btn-pause">Pause</button>
  <div class="slider-group">
    <label>Step</label>
    <input type="range" id="scrubber" class="ctrl-slider" style="width:140px" min="0" max="{traj_max}" value="{traj_max}">
    <span id="step-label">{traj_len}/{traj_len}</span>
  </div>
  <div class="slider-group">
    <label>Speed</label>
    <input type="range" id="speed" class="ctrl-slider" min="30" max="800" value="200">
    <span id="speed-label">200ms</span>
  </div>
  <span style="color:#1A1E2A;margin:0 2px">|</span>
  <button id="btn-field" class="active">Field</button>
  <button id="btn-trail" class="active">Trail</button>
  <button id="btn-splats" class="active">Splats</button>
  <button id="btn-tokens" class="active">Tokens</button>
</div>
<button id="settings-btn">&#9881;</button>
<div id="settings-panel">
  <h3>SETTINGS</h3>
  <div class="setting-row">
    <label>Theme</label>
    <button id="btn-dark" class="mode-btn active">Dark</button>
    <button id="btn-light" class="mode-btn">Light</button>
  </div>
  <div class="setting-row">
    <label>Rotate</label>
    <button id="btn-rotate-on" class="mode-btn active">On</button>
    <button id="btn-rotate-off" class="mode-btn">Off</button>
  </div>
  <div class="setting-row">
    <label>UI Scale</label>
    <input type="range" id="ui-scale" class="ctrl-slider" min="50" max="150" value="100">
    <span id="ui-scale-val">100%</span>
  </div>
  <div class="setting-row">
    <label>Text Size</label>
    <input type="range" id="text-scale" class="ctrl-slider" min="50" max="200" value="100">
    <span id="text-scale-val">100%</span>
  </div>
  <div class="setting-row">
    <label>Particles</label>
    <input type="range" id="particle-density" class="ctrl-slider" min="0" max="300" value="80">
    <span id="particle-val">80</span>
  </div>
</div>
<canvas id="c"></canvas>
<script>
const T={trajectory_js};
const D={deltas_js};
const F={field_js};
const G={goal_js};
const SP={splats_js};
const SA={splat_alphas_js};
const NB={neighbors_js};
const RG={ridge_js};
const TK={tokens_js};

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const dpr=devicePixelRatio;
const genText=document.getElementById('gen-text');

function resize(){{canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px'}}
resize();window.addEventListener('resize',resize);

// Scene centering
let cx=0,cy=0,cz=0;
for(const p of T){{cx+=p[0];cy+=p[1];cz+=p[2]}}
const n=T.length||1;cx/=n;cy/=n;cz/=n;
let maxDist=1;
for(const p of T){{const d=Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz);if(d>maxDist)maxDist=d}}
const tC=T.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const fC=F.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const gC=[G[0]-cx,G[1]-cy,G[2]-cz];
const spC=SP.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const rgC=RG.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const nbC=NB.map(stepN=>stepN.map(nb=>({{t:nb.t,s:nb.s,p:[nb.p[0]-cx,nb.p[1]-cy,nb.p[2]-cz]}})));
const maxD=Math.max(...D,1);

// Detect jump threshold (top 15% of deltas = jumps)
const sortedD=[...D].sort((a,b)=>b-a);
const jumpThreshold=sortedD[Math.floor(sortedD.length*0.15)]||maxD*0.7;

// Camera
let camTheta=0.5,camPhi=0.4,camDist=maxDist*2.5;
let panX=0,panY=0;
let dragging=false,panning=false,lastX=0,lastY=0;
canvas.addEventListener('mousedown',e=>{{dragging=true;panning=e.shiftKey;lastX=e.clientX;lastY=e.clientY}});
window.addEventListener('mouseup',()=>{{dragging=false;panning=false}});
window.addEventListener('mousemove',e=>{{
  if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
  if(panning){{panX-=dx*camDist*0.001;panY+=dy*camDist*0.001}}
  else{{camTheta+=dx*0.005;camPhi=Math.max(-1.4,Math.min(1.4,camPhi+dy*0.005))}}
}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();camDist=Math.max(0.5,camDist*(1+e.deltaY*0.001))}},{{passive:false}});

// State
let visibleSteps=T.length,playing=false,replaySpeed=200;
let showField=true,showTrail=true,showSplats=true,showTokens=true;
let isDark=true,uiScale=1,textScale=1,particleCount=80,autoRotate=true;
const darkBg='#0F111A',lightBg='#E0E4EA';
const darkHalo='#0F111A',lightHalo='#E0E4EA';

// Ambient particles
let particles=[];
function initParticles(count){{
  particles=[];
  for(let i=0;i<count;i++){{
    particles.push({{x:(Math.random()-0.5)*maxDist*4,y:(Math.random()-0.5)*maxDist*4,z:(Math.random()-0.5)*maxDist*4,
      vx:(Math.random()-0.5)*0.003,vy:(Math.random()-0.5)*0.003,vz:(Math.random()-0.5)*0.003,
      r:1+Math.random()*3,hue:170+Math.random()*50}});
  }}
}}
initParticles(particleCount);

const scrubber=document.getElementById('scrubber');
const stepLabel=document.getElementById('step-label');
const speedSlider=document.getElementById('speed');
const speedLabel=document.getElementById('speed-label');
const btnPlay=document.getElementById('btn-play');
const btnPause=document.getElementById('btn-pause');
scrubber.addEventListener('input',()=>{{visibleSteps=parseInt(scrubber.value)+1;playing=false;updateBtns();updateGenText()}});
speedSlider.addEventListener('input',()=>{{replaySpeed=parseInt(speedSlider.value);speedLabel.textContent=replaySpeed+'ms'}});
btnPlay.addEventListener('click',()=>{{playing=true;visibleSteps=1;updateBtns();updateGenText()}});
btnPause.addEventListener('click',()=>{{playing=false;visibleSteps=T.length;scrubber.value=T.length-1;updateBtns();updateGenText()}});
// Prompt bar -- Generate triggers Play replay
const promptInput=document.getElementById('prompt-input');
const generateBtn=document.getElementById('generate-btn');
function triggerGenerate(){{
  // Reset and replay the pre-baked generation
  visibleSteps=1;playing=true;genText.innerHTML='';updateBtns();updateGenText();
  // Flash the button
  generateBtn.style.boxShadow='0 0 20px rgba(64,224,208,0.5)';
  setTimeout(()=>generateBtn.style.boxShadow='',400);
}}
generateBtn.addEventListener('click',triggerGenerate);
promptInput.addEventListener('keydown',e=>{{if(e.key==='Enter')triggerGenerate()}});
function toggleLayer(id,get,set){{const b=document.getElementById(id);b.addEventListener('click',()=>{{set(!get());b.classList.toggle('active');b.classList.toggle('off')}})}}
toggleLayer('btn-field',()=>showField,v=>showField=v);
toggleLayer('btn-trail',()=>showTrail,v=>showTrail=v);
toggleLayer('btn-splats',()=>showSplats,v=>showSplats=v);
toggleLayer('btn-tokens',()=>showTokens,v=>showTokens=v);
function updateBtns(){{
  btnPlay.classList.toggle('active',playing);btnPause.classList.toggle('active',!playing);
  stepLabel.textContent=visibleSteps+'/'+T.length;scrubber.value=Math.max(0,visibleSteps-1);
}}
function updateGenText(){{
  let html='';
  for(let i=0;i<Math.min(visibleSteps,TK.length);i++){{
    const tok=TK[i].replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if(i===visibleSteps-1)html+=`<span class="token-current">${{tok}}</span>`;
    else html+=tok;
  }}
  genText.innerHTML=html||'...';
  genText.scrollTop=genText.scrollHeight;
}}
updateGenText();

// Settings panel
const settingsBtn=document.getElementById('settings-btn');
const settingsPanel=document.getElementById('settings-panel');
settingsBtn.addEventListener('click',()=>settingsPanel.classList.toggle('open'));
document.getElementById('btn-dark').addEventListener('click',()=>{{isDark=true;document.getElementById('btn-dark').classList.add('active');document.getElementById('btn-light').classList.remove('active');
  document.body.style.background=darkBg;document.getElementById('hud').style.background='rgba(15,17,26,0.88)';document.getElementById('gen-text').style.background='rgba(15,17,26,0.88)';
  document.getElementById('hud').style.color='#7B8CA8';document.getElementById('gen-text').style.color='#A0B4D0';
  settingsPanel.style.background='rgba(15,17,26,0.94)';
}});
document.getElementById('btn-light').addEventListener('click',()=>{{isDark=false;document.getElementById('btn-light').classList.add('active');document.getElementById('btn-dark').classList.remove('active');
  document.body.style.background=lightBg;document.getElementById('hud').style.background='rgba(240,242,248,0.92)';document.getElementById('gen-text').style.background='rgba(240,242,248,0.92)';
  document.getElementById('hud').style.color='#3A4A5A';document.getElementById('gen-text').style.color='#2A3A4A';
  settingsPanel.style.background='rgba(240,242,248,0.94)';
}});
document.getElementById('ui-scale').addEventListener('input',e=>{{uiScale=parseInt(e.target.value)/100;document.getElementById('ui-scale-val').textContent=e.target.value+'%';
  document.getElementById('hud').style.transform=`scale(${{uiScale}})`;document.getElementById('hud').style.transformOrigin='top left';
  document.getElementById('controls').style.transform=`translateX(-50%) scale(${{uiScale}})`;
  document.getElementById('gen-text').style.transform=`scale(${{uiScale}})`;document.getElementById('gen-text').style.transformOrigin='top right';
}});
document.getElementById('text-scale').addEventListener('input',e=>{{textScale=parseInt(e.target.value)/100;document.getElementById('text-scale-val').textContent=e.target.value+'%'}});
document.getElementById('btn-rotate-on').addEventListener('click',()=>{{autoRotate=true;document.getElementById('btn-rotate-on').classList.add('active');document.getElementById('btn-rotate-off').classList.remove('active')}});
document.getElementById('btn-rotate-off').addEventListener('click',()=>{{autoRotate=false;document.getElementById('btn-rotate-off').classList.add('active');document.getElementById('btn-rotate-on').classList.remove('active')}});
document.getElementById('particle-density').addEventListener('input',e=>{{particleCount=parseInt(e.target.value);document.getElementById('particle-val').textContent=e.target.value;initParticles(particleCount)}});

// Projection
function project(x,y,z){{
  const ex=camDist*Math.cos(camPhi)*Math.cos(camTheta)+panX;
  const ey=camDist*Math.sin(camPhi)+panY;
  const ez=camDist*Math.cos(camPhi)*Math.sin(camTheta);
  const fx=panX-ex,fy=panY-ey,fz=-ez;
  const fl=Math.hypot(fx,fy,fz);const fdx=fx/fl,fdy=fy/fl,fdz=fz/fl;
  const rx=fdy*0-fdz*1,ry=fdz*0-fdx*0,rz=fdx*1-fdy*0;
  const rl=Math.hypot(rx,ry,rz)||1;const rdx=rx/rl,rdy=ry/rl,rdz=rz/rl;
  const udx=rdy*fdz-rdz*fdy,udy=rdz*fdx-rdx*fdz,udz=rdx*fdy-rdy*fdx;
  const dx=x-ex,dy=y-ey,dz=z-ez;
  const vx=rdx*dx+rdy*dy+rdz*dz;const vy=udx*dx+udy*dy+udz*dz;const vz=fdx*dx+fdy*dy+fdz*dz;
  if(vz<0.01)return null;
  const fov=1.2,w=canvas.width,h=canvas.height,asp=w/h;
  return {{x:w/2+(vx/vz)*w/(fov*asp),y:h/2-(vy/vz)*w/fov,z:vz}};
}}

function drawCircle(px,py,r,color,alpha){{
  ctx.beginPath();ctx.arc(px,py,r,0,Math.PI*2);ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.fill();
}}
function drawPill(x,y,w,h,r){{
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath();
}}
// Text with dark halo outline for readability
function haloText(text,x,y,fillColor,alpha){{
  ctx.globalAlpha=1;ctx.lineWidth=3*dpr;ctx.strokeStyle=isDark?'#0F111A':'#E0E4EA';ctx.lineJoin='round';
  ctx.strokeText(text,x,y);ctx.fillStyle=fillColor;ctx.fillText(text,x,y);
}}

// Z-depth fog: dims alpha for far objects, enhances near objects
function fog(z,alpha){{const near=camDist*0.3,far=camDist*4;const t=Math.max(0,Math.min(1,(z-near)/(far-near)));return alpha*(1-t*0.85);}}

// ---- RENDER ----
let lastTick=0;
function frame(ts){{
  requestAnimationFrame(frame);
  // Subtle ambient camera breathing
  if(!dragging&&autoRotate)camTheta+=0.0003;
  if(playing&&ts-lastTick>replaySpeed){{
    lastTick=ts;visibleSteps=Math.min(visibleSteps+1,T.length);
    if(visibleSteps>=T.length)playing=false;
    updateBtns();updateGenText();
  }}
  const w=canvas.width,h=canvas.height;
  const bgColor=isDark?darkBg:lightBg;
  const haloColor=isDark?darkHalo:lightHalo;
  ctx.globalAlpha=1;ctx.fillStyle=bgColor;ctx.fillRect(0,0,w,h);
  const steps=visibleSteps;
  const ts_f=textScale; // text scale factor

  // ==== Ambient floating particles ====
  for(const pt of particles){{
    pt.x+=pt.vx;pt.y+=pt.vy;pt.z+=pt.vz;
    // Wrap around
    const bound=maxDist*3;
    if(pt.x>bound)pt.x=-bound;if(pt.x<-bound)pt.x=bound;
    if(pt.y>bound)pt.y=-bound;if(pt.y<-bound)pt.y=bound;
    if(pt.z>bound)pt.z=-bound;if(pt.z<-bound)pt.z=bound;
    const pp=project(pt.x,pt.y,pt.z);
    if(pp&&pp.z>0){{
      const pa=fog(pp.z,isDark?0.95:0.85);
      const col=isDark?`hsl(${{pt.hue}},55%,55%)`:`hsl(${{pt.hue}},30%,60%)`;
      ctx.globalAlpha=pa;ctx.fillStyle=col;
      const pr=Math.max(5,pt.r*10*dpr/pp.z);
      ctx.beginPath();ctx.arc(pp.x,pp.y,pr,0,Math.PI*2);ctx.fill();
    }}
  }}

  // ==== Field cloud (drifting dust motes) ====
  if(showField){{
    const drift=ts*0.0001;
    for(let i=0;i<fC.length;i+=3){{
      // Slow sinusoidal drift per particle
      const sx=fC[i][0]+Math.sin(drift+i*0.7)*0.02;
      const sy=fC[i][1]+Math.cos(drift+i*1.1)*0.02;
      const sz=fC[i][2]+Math.sin(drift+i*0.3)*0.015;
      const p=project(sx,sy,sz);
      if(p&&p.z>0){{
        const fa=fog(p.z,0.1);
        ctx.globalAlpha=fa;ctx.fillStyle='#1E2840';
        ctx.beginPath();ctx.arc(p.x,p.y,Math.max(0.5,1.2*dpr/p.z),0,Math.PI*2);ctx.fill();
      }}
    }}
  }}

  // ==== Ridge ghost trail ====
  if(rgC.length>1){{
    ctx.setLineDash([5*dpr,4*dpr]);ctx.lineWidth=1*dpr;ctx.strokeStyle='#2A3A52';ctx.globalAlpha=0.25;
    ctx.beginPath();let started=false;
    for(let i=0;i<rgC.length;i++){{const p=project(rgC[i][0],rgC[i][1],rgC[i][2]);
      if(p){{if(!started){{ctx.moveTo(p.x,p.y);started=true}}else ctx.lineTo(p.x,p.y)}}
    }}ctx.stroke();ctx.setLineDash([]);
  }}

  // ==== Trail: ice-blue, energy-based width/opacity ====
  if(showTrail&&steps>1){{
    ctx.lineCap='round';ctx.lineJoin='round';
    for(let i=1;i<steps;i++){{
      const p0=project(tC[i-1][0],tC[i-1][1],tC[i-1][2]);
      const p1=project(tC[i][0],tC[i][1],tC[i][2]);
      if(!p0||!p1)continue;
      const dn=(D[i]||0)/maxD;
      // Confidence = inverse of delta. Low delta = confident = bright/thick
      const confidence=1-dn;
      const lineW=(0.8+confidence*3)*dpr;
      const alpha=fog(p1.z,0.25+confidence*0.5);
      ctx.globalAlpha=alpha;
      ctx.strokeStyle='#40E0D0';
      ctx.lineWidth=lineW;
      ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y);ctx.stroke();
    }}

    // Trail dots -- small, subtle, fog-faded
    for(let i=0;i<steps;i++){{const p=project(tC[i][0],tC[i][1],tC[i][2]);
      if(p&&p.z>0){{
        const dn=(D[i]||0)/maxD;
        const confidence=1-dn;
        const r=Math.max(1.5,(1.5+confidence*2)*dpr/p.z);
        drawCircle(p.x,p.y,r,'#40E0D0',fog(p.z,0.3+confidence*0.5));
      }}
    }}

    // ==== Jump markers (high-delta steps) ====
    for(let i=0;i<steps;i++){{
      if((D[i]||0)>=jumpThreshold){{
        const p=project(tC[i][0],tC[i][1],tC[i][2]);
        if(p&&p.z>0){{
          const r=Math.max(4,8*dpr/p.z);
          // Soft glow
          drawCircle(p.x,p.y,r*2.5,'#F0E6C8',0.1);
          drawCircle(p.x,p.y,r*1.5,'#F0E6C8',0.2);
          // Solid marker
          drawCircle(p.x,p.y,r,'#F0E6C8',0.85);
        }}
      }}
    }}

    // Decoded token labels on trail (every 5 steps)
    for(let i=0;i<steps;i++){{
      if(i===steps-1 || (i%5===0 && TK[i])){{
        const p=project(tC[i][0],tC[i][1],tC[i][2]);
        if(p&&p.z>0){{
          const tok=(TK[i]||'').trim();
          if(tok.length>0){{
            const fs=Math.round(13*dpr*ts_f);
            ctx.font=`600 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
            ctx.textBaseline='bottom';
            // Current = bright white, past = muted dark gray (not transparent)
            const col=i===steps-1?'#FFFFFF':'#3A4A5A';
            haloText(tok,p.x+8*dpr,p.y-3*dpr,col,1);
          }}
        }}
      }}
    }}

    // Current step highlight
    if(steps>0&&steps<=tC.length){{const p=project(tC[steps-1][0],tC[steps-1][1],tC[steps-1][2]);
      if(p){{
        drawCircle(p.x,p.y,Math.max(5,14*dpr/p.z),'#40E0D0',0.15);
        drawCircle(p.x,p.y,Math.max(3,8*dpr/p.z),'#40E0D0',0.7);
      }}
    }}
  }}

  // ==== Goal attractor (soft gold, subtle) ====
  const gp=project(gC[0],gC[1],gC[2]);
  if(gp){{
    drawCircle(gp.x,gp.y,Math.max(6,14*dpr/gp.z),'#D4A44C',0.12);
    drawCircle(gp.x,gp.y,Math.max(4,10*dpr/gp.z),'#D4A44C',0.6);
    // Label
    const fs=Math.round(10*dpr);
    ctx.font=`600 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
    ctx.globalAlpha=0.5;ctx.fillStyle='#D4A44C';ctx.textBaseline='middle';
    ctx.fillText('GOAL',gp.x+12*dpr,gp.y);
  }}

  // ==== Start / End markers ====
  if(showTrail&&steps>0){{
    const sp=project(tC[0][0],tC[0][1],tC[0][2]);
    if(sp)drawCircle(sp.x,sp.y,Math.max(3,8*dpr/sp.z),'#40E0D0',0.8);
  }}
  if(showTrail&&steps>1){{
    const idx=Math.min(steps-1,tC.length-1);
    const ep=project(tC[idx][0],tC[idx][1],tC[idx][2]);
    if(ep)drawCircle(ep.x,ep.y,Math.max(3,8*dpr/ep.z),'#D4A44C',0.8);
  }}

  // ==== Splat scars (muted teal / rust) ====
  if(showSplats){{
    for(let i=0;i<spC.length;i++){{const p=project(spC[i][0],spC[i][1],spC[i][2]);
      if(p&&p.z>0){{
        const isPleasure=(SA[i]||1)>0;
        const col=isPleasure?'#2DD4A8':'#8B3A3A';
        const r=Math.max(2,6*dpr/p.z);
        if(isPleasure)drawCircle(p.x,p.y,r*2,col,0.05);
        drawCircle(p.x,p.y,r,col,0.45);
      }}
    }}
  }}

  // ==== Ephemeral token attraction orbs (screen-space fan, fixed pixel radius) ====
  if(showTokens){{
    const fadeWindow=5;
    for(let si=Math.max(0,steps-fadeWindow);si<steps;si++){{
      if(!nbC[si]||nbC[si].length===0)continue;
      const age=steps-1-si;
      const fade=Math.max(0,1-age/fadeWindow);
      const stepP=project(tC[si][0],tC[si][1],tC[si][2]);
      if(!stepP)continue;
      const neighbors=nbC[si].filter(nb=>nb.s>=0.001);
      const count=neighbors.length;
      if(!count)continue;
      const fanR=110*dpr; // fixed pixel radius -- zoom independent
      const startAngle=-Math.PI*0.8;
      const sweep=Math.PI*1.6;
      for(let j=0;j<count;j++){{
        const nb=neighbors[j];const prob=nb.s;
        // Fan position in screen space around trajectory point
        const angle=startAngle+(j/(Math.max(count-1,1)))*sweep;
        const ox=stepP.x+Math.cos(angle)*fanR*(0.7+prob*0.3);
        const oy=stepP.y+Math.sin(angle)*fanR*(0.7+prob*0.3);
        // Orb
        const orbR=Math.max(4,(5+prob*10)*dpr);
        const color=`hsl(35,65%,${{45+prob*15}}%)`;
        drawCircle(ox,oy,orbR*1.5,color,0.08*fade);
        drawCircle(ox,oy,orbR,color,0.6*fade);
        // Connection line (current step only)
        if(age===0){{
          ctx.globalAlpha=0.2;ctx.strokeStyle=color;ctx.lineWidth=1*dpr;
          ctx.setLineDash([3*dpr,3*dpr]);
          ctx.beginPath();ctx.moveTo(stepP.x,stepP.y);ctx.lineTo(ox,oy);ctx.stroke();
          ctx.setLineDash([]);
        }}
        // Label with pill background + halo text
        if(fade>0.3){{
          const label=nb.t.trim();
          if(label.length>0&&label.length<25){{
            const fs=Math.round(14*dpr*ts_f);
            ctx.font=`700 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
            ctx.textBaseline='middle';
            const probText=`${{(prob*100).toFixed(1)}}%`;
            const fullText=label+'  '+probText;
            const tw=ctx.measureText(fullText).width;
            const lx=ox+orbR+5*dpr;
            // Background pill
            ctx.globalAlpha=0.85*fade;ctx.fillStyle='rgba(15,17,26,0.92)';
            drawPill(lx-4*dpr,oy-fs/2-3*dpr,tw+10*dpr,fs+6*dpr,4*dpr);ctx.fill();
            // Token name -- bright
            haloText(label,lx,oy,'#E0D0A0',fade);
            // Probability
            const labelW=ctx.measureText(label+'  ').width;
            haloText(probText,lx+labelW,oy,'#8A7A5A',0.7*fade);
          }}
        }}
      }}
    }}
  }}

  ctx.globalAlpha=1;
}}
requestAnimationFrame(frame);
</script>
</body>
</html>"##,
        prompt = prompt_escaped,
        traj_len = data.trajectory_3d.len(),
        traj_max = data.trajectory_3d.len().saturating_sub(1),
        field_len = data.field_points_3d.len(),
        trajectory_js = trajectory_js,
        deltas_js = deltas_js,
        field_js = field_js,
        goal_js = goal_js,
        splats_js = splats_js,
        splat_alphas_js = splat_alphas_js,
        neighbors_js = neighbors_js,
        ridge_js = ridge_js,
        tokens_js = tokens_js,
    );

    if let Err(e) = std::fs::write(&html_path, &html) {
        eprintln!("    [VIZ] Failed to write viewer: {}", e);
        return;
    }

    println!("    [VIZ] 3D viewer: {}", html_path.display());
    let _ = std::process::Command::new("open").arg(&html_path).spawn();
}

// ---------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------

fn points_to_js(points: &[[f32; 3]]) -> String {
    let entries: Vec<String> = points
        .iter()
        .map(|p| format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]))
        .collect();
    format!("[{}]", entries.join(","))
}

fn floats_to_js(vals: &[f32]) -> String {
    let entries: Vec<String> = vals.iter().map(|v| format!("{:.4}", v)).collect();
    format!("[{}]", entries.join(","))
}
```

---

