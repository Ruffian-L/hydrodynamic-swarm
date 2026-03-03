
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
            dt: 0.08,
            viscosity_scale: 0.35,
            force_cap: 35.0,
            splat_sigma: 35.0,
            splat_alpha: 2.0,
            min_splat_dist: 100.0,
            splat_delta_threshold: 12.0,
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
        assert!((cfg.physics.viscosity_scale - 0.35).abs() < 1e-6);
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
