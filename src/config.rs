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
    /// Per-step blend factor pulling steered state back toward baseline (0.0 = off, 0.15 = gentle).
    pub manifold_pullback: f32,
    pub bundle_min_dist: f32,
    pub splat_lambda_default: f32,
    pub pain_decay_factor: f32,
    pub dream_correction_threshold: f32,
    /// Scale applied to splat force before sum (0.05–0.15 = gentle for Gemma).
    pub splat_force_scale: f32,
    /// Soft max on ||F_s|| after scale (0 = disabled).
    pub splat_force_max: f32,
    /// Scale applied to goal attractor (prefill residual).
    pub goal_force_scale: f32,
    /// Soft max on ||F_a|| after scale (0 = disabled).
    pub goal_force_max: f32,
    /// Step index to begin late F_a attenuation (0 = off). Typical B4d: 48.
    pub goal_late_start: usize,
    /// Tokens to ramp from full F_a → goal_late_end.
    pub goal_late_span: usize,
    /// F_a multiplier at end of late attenuation (0–1). e.g. 0.35.
    pub goal_late_end: f32,
    /// Min steps between online splat deposits (anti-spam).
    pub online_splat_interval: usize,
    /// Field wake mode: "off" | "wake" | "blend" | "dist_weighted"
    /// See research_logs/*field-wake* ablation table.
    pub field_wake_mode: String,
    /// k nearest embeddings for wake pull.
    pub field_wake_k: usize,
    /// Strength of nearest-emb pull (before soft cap).
    pub field_wake_scale: f32,
    /// Soft max on ||wake force|| (0 = off).
    pub field_wake_max: f32,
    /// In blend mode: weight of pure ∇ρ vs wake (0=all wake, 1=all grad when alive).
    pub field_grad_blend: f32,
    /// Distance scale for dist_weighted: strength ∝ 1/(1+(d/τ)²).
    pub field_wake_dist_tau: f32,
    /// Surface logit bias: z += α · normalize(E û_g). 0 = off.
    /// û_g = unit field force direction from residual steer (same D as emb).
    pub field_logit_alpha: f32,
    /// Force ramp: first N tokens scale total force from `force_ramp_start` → 1.0.
    /// Original Niodoo spirit: gentler early, respect prefill J-space. 0 = off.
    pub force_ramp_tokens: usize,
    /// Multiplier at step 0 when ramping (e.g. 0.15).
    pub force_ramp_start: f32,
    /// If true, only deposit splats on high-signal steps (δ > thresh OR pain OR strong pleasure).
    /// If false, any non-Skip quality deposit (current default path).
    pub targeted_splat_only: bool,
    /// After prefill, run one micro-dream against goal (respect initial hidden / J-space).
    pub prefill_micro_dream: bool,
    /// On Pain, deposit a stronger ocean "recovery" packet (variant E).
    pub pain_recovery_ocean: bool,
}

/// Generation parameters.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub default_prompt: String,
    pub eos_token_ids: Vec<u32>,
    pub rep_penalty: f32,
    pub min_success_tokens: usize,
    pub pleasure_alpha: f32,
    pub pain_alpha: f32,
}

/// Splat memory management.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub max_splats: usize,
    pub consolidation_dist: f32,
    /// End-of-run / session wall-clock evaporation fallback (see `decay_step`).
    pub decay_rate: f32,
    pub prune_threshold: f32,
    /// Per-token scar strength multiply during generation (`decay_per_token`).
    /// `1.0` = off. Typical B4b: `0.97`–`0.99`. Controls mid-run F_s climb.
    pub online_decay_rate: f32,
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
            dt: 0.035,
            // Slightly more field influence now that sigma isn't dead
            viscosity_scale: 0.25,
            force_cap: 5.0,
            // Residual-space friendly; was 35 and treated every splat as global
            splat_sigma: 12.0,
            splat_alpha: 1.2,
            min_splat_dist: 25.0,
            // Was 12 — steering delta is routinely 100+, so that deposited every step
            splat_delta_threshold: 90.0,
            gradient_topk: 1024,
            steer_hidden: true,
            manifold_pullback: 0.20,
            bundle_min_dist: 0.05,
            splat_lambda_default: 0.02,
            pain_decay_factor: 0.7,
            dream_correction_threshold: 6.0,
            // Wake memory a bit; 0.08 left F_s≈0 mid-run
            splat_force_scale: 0.25,
            splat_force_max: 60.0,
            // Stop prefill goal from monopolizing (~450 uncapped)
            goal_force_scale: 0.15,
            goal_force_max: 60.0,
            goal_late_start: 0,
            goal_late_span: 30,
            goal_late_end: 0.4,
            online_splat_interval: 6,
            // Phase 1 default: nearest-emb wake (k=1)
            field_wake_mode: "wake".into(),
            field_wake_k: 1,
            field_wake_scale: 0.20,
            field_wake_max: 40.0,
            field_grad_blend: 0.15,
            field_wake_dist_tau: 50.0,
            // Gentle surface tip; residual physics remains primary
            field_logit_alpha: 0.15,
            force_ramp_tokens: 12,
            force_ramp_start: 0.20,
            targeted_splat_only: true,
            prefill_micro_dream: false,
            pain_recovery_ocean: false,
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
            rep_penalty: 1.25,
            min_success_tokens: 15,
            pleasure_alpha: 1.2,
            pain_alpha: -0.6,
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
            online_decay_rate: 1.0, // off unless config sets < 1
        }
    }
}

impl Default for MicroDreamConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 3.0,
            fixed_interval: 25,
            adaptive_interval: 8,
            blend_normal: 0.10,
            blend_high_entropy: 0.15,
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
        if p.bundle_min_dist <= 0.0 {
            return Err("physics.bundle_min_dist must be > 0".into());
        }
        if p.splat_lambda_default < 0.0 {
            return Err("physics.splat_lambda_default must be >= 0".into());
        }
        if p.pain_decay_factor <= 0.0 || p.pain_decay_factor > 1.0 {
            return Err("physics.pain_decay_factor must be in (0,1]".into());
        }
        if p.dream_correction_threshold < 0.0 {
            return Err("physics.dream_correction_threshold must be >= 0".into());
        }
        if p.goal_late_end < 0.0 || p.goal_late_end > 1.0 {
            return Err("physics.goal_late_end must be in [0, 1]".into());
        }
        if p.goal_late_start > 0 && p.goal_late_span == 0 {
            return Err("physics.goal_late_span must be > 0 when goal_late_start is set".into());
        }

        let g = &self.generation;
        if g.max_tokens == 0 {
            return Err("generation.max_tokens must be > 0".into());
        }
        if g.temperature <= 0.0 {
            return Err(
                "generation.temperature must be > 0 (zero causes division-by-zero in sampling)"
                    .into(),
            );
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
        if m.online_decay_rate <= 0.0 || m.online_decay_rate > 1.0 {
            return Err("memory.online_decay_rate must be in (0, 1]".into());
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
        assert!((cfg.physics.viscosity_scale - 0.15).abs() < 1e-6);
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
