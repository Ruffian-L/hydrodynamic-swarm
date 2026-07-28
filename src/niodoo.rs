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
use crate::memory::{PrimeGovernor, SplatMemory};
use crate::ocean::SharedOcean;
use candle_core::{Result, Tensor};

/// Result of a single steering step, including force telemetry.
pub struct SteerResult {
    pub steered: Tensor,
    pub grad_mag: f32,
    pub splat_mag: f32,
    pub goal_mag: f32,
    pub ocean_mag: f32,
    /// Unit direction of the field force F_g (D,) for surface logit bias.
    /// Zero vector if field force was dead.
    pub field_dir: Tensor,
}

/// How F_g is built from the Diderot emb cloud (Phase 1 field wake).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldWakeMode {
    /// Pure ∇ρ only (baseline; dead from residual space).
    Off,
    /// Replace dead/weak ∇ρ with nearest-emb pull.
    Wake,
    /// Always: (1-β)·wake + β·∇ρ (hybrid / local GMM component).
    Blend,
    /// Wake strength falls with L2 distance to nearest emb.
    DistWeighted,
}

impl FieldWakeMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "baseline" => Self::Off,
            "blend" | "hybrid" => Self::Blend,
            "dist" | "dist_weighted" | "distance" => Self::DistWeighted,
            _ => Self::Wake,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Wake => "wake",
            Self::Blend => "blend",
            Self::DistWeighted => "dist_weighted",
        }
    }
}

/// Knobs for nearest-embedding manifold wake.
#[derive(Debug, Clone)]
pub struct FieldWakeConfig {
    pub mode: FieldWakeMode,
    pub k: usize,
    pub scale: f32,
    pub max_mag: f32,
    /// Blend weight of pure ∇ρ when mode=Blend.
    pub grad_blend: f32,
    /// τ for dist-weighted strength ∝ 1/(1 + d/τ).
    pub dist_tau: f32,
}

impl Default for FieldWakeConfig {
    fn default() -> Self {
        Self {
            mode: FieldWakeMode::Wake,
            k: 1,
            scale: 0.20,
            max_mag: 40.0,
            grad_blend: 0.15,
            dist_tau: 50.0,
        }
    }
}

pub struct NiodooEngine {
    field: ContinuousField,
    memory: SplatMemory,
    prime_governor: PrimeGovernor,
    backend: Box<dyn PhysicsBackend>,
    dt: f32,
    viscosity_scale: f32,
    force_cap: f32,
    gradient_topk: usize,
    /// Multiplies splat force before it joins the sum (Gemma needs a gentle nudge).
    splat_force_scale: f32,
    /// Soft ceiling on ||F_s|| after scale (0 = off).
    splat_force_max: f32,
    /// Multiplies goal attractor (prefill residual pull).
    goal_force_scale: f32,
    /// Soft ceiling on ||F_a|| after scale (0 = off).
    goal_force_max: f32,
    /// After this step, linearly attenuate F_a toward `goal_late_end` (0 = off).
    /// Keeps early J-space pull; eases sticky prefill attractor late (B4d).
    goal_late_start: usize,
    /// Tokens over which to ramp attenuation (after start).
    goal_late_span: usize,
    /// Multiplier on F_a at full late attenuation (e.g. 0.35).
    goal_late_end: f32,
    /// Nearest-emb / hybrid field wake (Phase 1).
    field_wake: FieldWakeConfig,
    /// Force ramp: steps + start multiplier (original Niodoo gentler early).
    force_ramp_tokens: usize,
    force_ramp_start: f32,
    // Endocrine modulation (Monolith BloomEvents from src/endocrine.rs — Shep restore)
    /// Cool noise after a monolith "Eureka" (optional; default unused by steer).
    pub noise_sigma: f32,
    /// Transient viscosity hint after monolith.
    pub viscosity: f32,
    /// Tag gravity mult after monolith (reserved for future endocrine wiring).
    #[allow(dead_code)]
    pub tag_gravity_mult: f32,
    /// Impulse mass * scale from last apply_monolith.
    pub eureka_impulse: f32,
    /// Native-model mean embedding of bloom text (D,) — real attractor during eureka.
    eureka_target: Option<Tensor>,
    /// Shared multi-mind ocean (Lane C). Optional force source.
    ocean: Option<SharedOcean>,
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
            prime_governor: PrimeGovernor::new(),
            backend,
            dt,
            viscosity_scale,
            force_cap,
            gradient_topk: 0, // 0 = exact gradient (default)
            // Gentle defaults: high-dim residual steering (Gemma 5376-d) dies under F_s sledgehammers.
            splat_force_scale: 0.08,
            splat_force_max: 80.0,
            // Prefill goal is a huge vector in 5376-d; uncapped it monopolizes the sum (~450).
            goal_force_scale: 0.15,
            goal_force_max: 60.0,
            goal_late_start: 0, // off
            goal_late_span: 30,
            goal_late_end: 0.4,
            field_wake: FieldWakeConfig::default(),
            force_ramp_tokens: 12,
            force_ramp_start: 0.20,
            noise_sigma: 0.3,
            viscosity: viscosity_scale,
            tag_gravity_mult: 1.0,
            eureka_impulse: 0.0,
            eureka_target: None,
            ocean: None,
        }
    }

    /// Apply a Monolith without a full-D target (impulse + viscosity only).
    #[allow(dead_code)]
    pub fn apply_monolith(&mut self, monolith: &crate::endocrine::Monolith) {
        self.apply_monolith_native(monolith, None);
    }

    /// Same as `apply_monolith`, plus a native (D,) target from the live model.
    pub fn apply_monolith_native(
        &mut self,
        monolith: &crate::endocrine::Monolith,
        native_target: Option<Tensor>,
    ) {
        // Cap impulse so one bloom cannot sledgehammer residual (same policy as F_s/F_a caps).
        self.eureka_impulse = (monolith.mass * 0.001).clamp(0.5, 8.0);
        self.viscosity = (self.viscosity * 0.3).max(0.05);
        self.noise_sigma = 0.05;
        if let Some(t) = native_target {
            self.eureka_target = Some(t);
        }
        let has_native = self.eureka_target.is_some();
        println!(
            "[NIODOO] Monolith applied (mass={:.0} impulse={:.2} pos4={:?} native={}). Eureka window open.",
            monolith.mass, self.eureka_impulse, monolith.pos, has_native
        );
    }

    /// Decay eureka window each token (called from generation loop).
    pub fn tick_endocrine(&mut self) {
        if self.eureka_impulse > 1e-4 {
            self.eureka_impulse *= 0.92;
            if self.eureka_impulse < 0.05 {
                self.eureka_impulse = 0.0;
                self.eureka_target = None;
                // restore viscosity toward base scale
                self.viscosity = self.viscosity_scale;
                self.noise_sigma = 0.3;
            }
        }
    }

    #[allow(dead_code)] // public API for endocrine / TUI callers
    pub fn eureka_impulse(&self) -> f32 {
        self.eureka_impulse
    }

    /// Original Niodoo ramp: weaker forces for first N tokens (respect prefill).
    pub fn set_force_ramp(&mut self, tokens: usize, start: f32) {
        self.force_ramp_tokens = tokens;
        self.force_ramp_start = start.clamp(0.0, 1.0);
    }

    /// Damp / soft-cap scar-tissue force (anti-runaway for multi-splat memory).
    pub fn set_splat_force_limits(&mut self, scale: f32, max_mag: f32) {
        self.splat_force_scale = scale.max(0.0);
        self.splat_force_max = max_mag.max(0.0);
    }

    /// Damp / soft-cap goal attractor (anti-monopoly for frozen prefill residual).
    pub fn set_goal_force_limits(&mut self, scale: f32, max_mag: f32) {
        self.goal_force_scale = scale.max(0.0);
        self.goal_force_max = max_mag.max(0.0);
    }

    /// Late-run F_a attenuation: after `start` steps, scale goal force down to `end_mult`
    /// over `span` tokens. `start == 0` disables. Early ramp/J-space unchanged.
    pub fn set_goal_late_attenuate(&mut self, start: usize, span: usize, end_mult: f32) {
        self.goal_late_start = start;
        self.goal_late_span = span.max(1);
        self.goal_late_end = end_mult.clamp(0.0, 1.0);
    }

    /// Configure nearest-embedding field wake (Phase 1).
    pub fn set_field_wake(&mut self, cfg: FieldWakeConfig) {
        self.field_wake = cfg;
    }

    #[allow(dead_code)] // public API for field-wake inspection
    pub fn field_wake(&self) -> &FieldWakeConfig {
        &self.field_wake
    }

    /// Nearest-emb pull as a **direction** toward k-NN emb rows, with strength
    /// independent of residual||emb|| distance (avoids ||μ−h||≈450 sledgehammer).
    ///
    /// F = strength · û ,  û = unit vector along weighted Σ w_i (μ_i − x)
    /// strength = scale · falloff(d_min) , then soft-capped by max_mag.
    ///
    /// DistWeighted: falloff = 1/(1 + d_min/τ)  (must NOT cancel via /wsum — that
    /// made k=1 dist_weighted identical to wake before the 2026-07-11 fix).
    fn nearest_emb_wake(&self, pos: &Tensor) -> Result<Tensor> {
        let k = self.field_wake.k.max(1);
        let nearest = match self.field.nearest_tokens(pos, k) {
            Ok(n) if !n.is_empty() => n,
            _ => {
                return Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device());
            }
        };

        let mut pull = Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device())?;
        let mut wsum = 0.0f32;
        let mut d_min = f32::MAX;

        for (tid, cos_sim) in &nearest {
            let emb = self.field.positions.get(*tid as usize)?;
            let delta = (&emb - pos)?;
            let d: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            d_min = d_min.min(d);

            // Direction weights only (cos for multi-k; equal for pure dist mode)
            let w = match self.field_wake.mode {
                FieldWakeMode::DistWeighted => 1.0,
                _ => cos_sim.max(0.0) + 0.05,
            };
            pull = (&pull + delta.affine(w as f64, 0.0)?)?;
            wsum += w;
        }

        if wsum < 1e-8 {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device());
        }

        // Unit direction toward weighted emb mean
        let pull = pull.affine(1.0 / wsum as f64, 0.0)?;
        let pull_mag: f32 = pull.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        if pull_mag < 1e-8 {
            return Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device());
        }
        let unit = pull.affine(1.0 / pull_mag as f64, 0.0)?;

        // Strength: base scale × optional distance falloff (applied AFTER unitize)
        let falloff = match self.field_wake.mode {
            FieldWakeMode::DistWeighted => {
                let tau = self.field_wake.dist_tau.max(1e-3);
                1.0 / (1.0 + d_min / tau)
            }
            // Mild residual-distance softening even in plain wake so ||μ−h|| doesn't
            // force us into max_mag every step (optional: falloff=1.0 for pure old wake)
            FieldWakeMode::Wake | FieldWakeMode::Blend | FieldWakeMode::Off => 1.0,
        };
        let mut strength = self.field_wake.scale * 200.0 * falloff; // ~unit·40 when falloff=1, scale=0.2
        // Map scale 0.2 → typical strength ~40 before cap (historical max_mag range)
        // strength = scale * 200 * falloff → 0.2*200=40 at falloff=1
        if self.field_wake.max_mag > 0.0 {
            strength = strength.min(self.field_wake.max_mag * falloff.max(0.05));
            // For Wake mode falloff=1 → still capped by max_mag
            // For DistWeighted at d=450,τ=80: falloff≈0.151 → strength≤30*0.151≈4.5
        }

        Ok(unit.affine(strength as f64, 0.0)?)
    }

    /// Attach shared multi-mind ocean (Lane C).
    pub fn set_ocean(&mut self, ocean: SharedOcean) {
        self.ocean = Some(ocean);
    }

    /// Mutable ocean access for deposits during generation.
    pub fn ocean_mut(&mut self) -> Option<&mut SharedOcean> {
        self.ocean.as_mut()
    }

    pub fn ocean(&self) -> Option<&SharedOcean> {
        self.ocean.as_ref()
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
    /// steered = baseline + dt * (grad*visc + splat + goal + ocean)
    pub fn steer(
        &mut self,
        baseline_residual: &Tensor,
        goal_pos: &Tensor,
        step: usize,
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

        // 1. Field force F_g — pure ∇ρ and/or nearest-emb wake (Phase 1)
        //
        // Geometry fact (field_audit): residual ||h||~450 sits outside emb shell
        // ||μ||~1, so pure ∇ρ underflows. Wake = pull toward k-NN emb rows
        // (local GMM component / manifold snap). See research log ablation table.
        let raw_grad = if self.gradient_topk > 0 {
            self.backend
                .field_gradient_topk(&self.field, &pos, self.gradient_topk)?
        } else {
            self.backend.field_gradient(&self.field, &pos)?
        };
        // Endocrine: during Eureka, use cooled viscosity for field grad (truth window).
        let visc_scale = if self.eureka_impulse > 1e-4 {
            self.viscosity
        } else {
            self.viscosity_scale
        };
        let pure_grad = raw_grad.affine(visc_scale as f64, 0.0)?;
        let pure_mag: f32 = pure_grad.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        let grad_force = match self.field_wake.mode {
            FieldWakeMode::Off => pure_grad,
            FieldWakeMode::Wake => {
                // Prefer pure ∇ρ when on-manifold; else nearest-emb wake
                if pure_mag > 1e-3 {
                    pure_grad
                } else {
                    self.nearest_emb_wake(&pos)?
                }
            }
            FieldWakeMode::Blend => {
                let wake = self.nearest_emb_wake(&pos)?;
                let beta = self.field_wake.grad_blend.clamp(0.0, 1.0) as f64;
                // When pure_grad is dead, beta contributes nothing — wake carries
                (&wake.affine(1.0 - beta, 0.0)? + &pure_grad.affine(beta, 0.0)?)?
            }
            FieldWakeMode::DistWeighted => {
                // Always wake with distance falloff (ignore dead pure grad)
                let wake = self.nearest_emb_wake(&pos)?;
                if pure_mag > 1e-3 {
                    let beta = self.field_wake.grad_blend.clamp(0.0, 1.0) as f64;
                    (&wake.affine(1.0 - beta, 0.0)? + &pure_grad.affine(beta, 0.0)?)?
                } else {
                    wake
                }
            }
        };

        // 2. Splat scar tissue force (via backend) — then damp + soft-cap
        let mut splat_force = self.backend.splat_force(&self.memory, &pos)?;
        if self.splat_force_scale != 1.0 {
            splat_force = splat_force.affine(self.splat_force_scale as f64, 0.0)?;
        }
        if self.splat_force_max > 0.0 {
            let mag: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if mag > self.splat_force_max && mag > 1e-8 {
                let s = (self.splat_force_max / mag) as f64;
                splat_force = splat_force.affine(s, 0.0)?;
            }
        }

        // 3. Goal attractor governed by PrimeGovernor + Embed phases — then damp + soft-cap
        let progress = (step as f32 / 200.0).min(1.0);
        let gov_factor = self.prime_governor.govern(1.0, progress);
        let mut goal_force = (goal_pos - &pos)?.affine(gov_factor as f64, 0.0)?;
        if self.goal_force_scale != 1.0 {
            goal_force = goal_force.affine(self.goal_force_scale as f64, 0.0)?;
        }
        if self.goal_force_max > 0.0 {
            let mag: f32 = goal_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if mag > self.goal_force_max && mag > 1e-8 {
                goal_force = goal_force.affine((self.goal_force_max / mag) as f64, 0.0)?;
            }
        }
        // Late F_a attenuation (B4d): after ramp, ease sticky prefill attractor.
        // Multiplier 1.0 until goal_late_start, then linear → goal_late_end over span.
        if self.goal_late_start > 0 && step >= self.goal_late_start {
            let t = ((step - self.goal_late_start) as f32 / self.goal_late_span as f32).min(1.0);
            let mult = 1.0 + (self.goal_late_end - 1.0) * t;
            if (mult - 1.0).abs() > 1e-6 {
                goal_force = goal_force.affine(mult as f64, 0.0)?;
            }
        }

        // 4. Shared ocean force (Lane C multi-mind crystallization)
        let ocean_force = if let Some(ocean) = self.ocean.as_mut() {
            ocean.query_force(&pos)?
        } else {
            Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device())?
        };

        // Force telemetry: magnitudes AFTER damp/cap (what actually enters the sum)
        let splat_mag: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let grad_mag: f32 = grad_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let goal_mag: f32 = goal_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let ocean_mag: f32 = ocean_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Unit field direction for surface logit bias: z += α (E · û_g)
        let field_dir = if grad_mag > 1e-8 {
            grad_force.affine(1.0 / grad_mag as f64, 0.0)?
        } else {
            Tensor::zeros(pos.dims(), candle_core::DType::F32, pos.device())?
        };

        // Sum and scale by dt
        let mut total_force = (((&grad_force + &splat_force)? + &goal_force)? + &ocean_force)?;

        // Endocrine Eureka: soft force boost + optional pull toward **native** bloom embed.
        if self.eureka_impulse > 1e-4 {
            let boost = (1.0 + 0.12 * self.eureka_impulse.min(5.0)) as f64;
            total_force = total_force.affine(boost, 0.0)?;
            if let Some(ref target) = self.eureka_target {
                // Pull residual toward native mean tok-emb of bloom text (same D as pos).
                if target.dims() == pos.dims() {
                    let pull = (target - &pos)?.affine(
                        (0.04 * self.eureka_impulse.min(5.0)) as f64,
                        0.0,
                    )?;
                    total_force = (&total_force + &pull)?;
                }
            }
        }

        // Ramp: gentler early (J-space / prefill respect). Linear start→1 over N tokens.
        if self.force_ramp_tokens > 0 {
            let t = (step as f32 / self.force_ramp_tokens as f32).min(1.0);
            let ramp = self.force_ramp_start + (1.0 - self.force_ramp_start) * t;
            total_force = total_force.affine(ramp as f64, 0.0)?;
        }

        // Force cap: prevent any single dimension from dominating (Variant 3)
        let total_force = total_force.clamp(-self.force_cap, self.force_cap)?;
        let steering = total_force.affine(self.dt as f64, 0.0)?;

        // Restore batch dim: (D,) -> (1, D) and add to baseline
        let steering_2d = steering.unsqueeze(0)?;
        let steered = (baseline_residual + &steering_2d)?;

        // === RENORMALIZATION: stay on the model manifold ===
        // Without this, cumulative steering drifts the hidden state norm,
        // causing lm_head to produce garbage after ~40-80 tokens.
        let baseline_norm: f32 = baseline_residual
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let steered_norm: f32 = steered.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let steered = if steered_norm > 0.0 && baseline_norm > 0.0 {
            steered.affine((baseline_norm / steered_norm) as f64, 0.0)?
        } else {
            steered
        };

        Ok(SteerResult {
            steered,
            grad_mag,
            splat_mag,
            goal_mag,
            ocean_mag,
            field_dir,
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
