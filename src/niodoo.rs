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

use crate::control_tags::ControlTag;
use crate::field::ContinuousField;
use crate::gpu::PhysicsBackend;
use crate::memory::{MemoryForceMode, MemoryPickConfig, PrimeGovernor, SplatMemory};
use crate::ocean::SharedOcean;
use crate::remember_geometry::RememberOffsetProbe;
use crate::remember_store::RememberStore;
use candle_core::{Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

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
    /// True when this step used ranked Top-K scar force (not soft-sum).
    pub memory_ranked: bool,
    /// ‖h‖ before steering.
    pub baseline_norm: f32,
    /// ‖h‖ after steering, before manifold renormalization.
    pub steered_norm: f32,
    /// Renorm factor actually applied (baseline_norm / steered_norm). 1.0 when
    /// steering is off — the residual is passed through untouched.
    pub pullback: f32,
    /// ‖dt · F_total‖ — how far this step moved the residual.
    pub delta_h_norm: f32,
    /// Ramp multiplier in effect this step (1.0 once past force_ramp_tokens).
    pub ramp: f32,
    /// Scar potential at this step's residual (Σ α exp(−d²/σ²)).
    pub scar_pot: f32,
    /// True when `scar_pot >= memory_warm_pot` and the warm gate is on, so F_s
    /// skipped the early ramp.
    pub memory_warm: bool,
    /// Endocrine Eureka force boost (1.0 when no impulse).
    pub eureka_boost: f32,
    /// Fraction of dimensions the force cap clipped. Nonzero means force_cap is
    /// binding — the knob is doing work, not just sitting there.
    pub clip_frac: f32,
    /// Live hand blend (SPIKE 6.5). Identity is 1.0.
    pub physics_blend: f32,
    /// QSMA β used this step (hand override or scheduled).
    pub qsma_beta: f32,
    /// Kinetic noise σ from the last hand.
    pub kinetic_noise: f32,
}

impl SteerResult {
    /// Same force telemetry, different residual — for callers that adjust the
    /// steered state (manifold pullback, bundle stress) after `steer` returns
    /// but before the logit surface reads it back as context.
    pub fn with_residual(&self, steered: Tensor) -> Self {
        Self {
            steered,
            field_dir: self.field_dir.clone(),
            ..*self
        }
    }

    /// Fixture base: no forces, no drift. Tests override only what they assert on
    /// via struct update syntax, so adding telemetry fields stays a one-line change.
    #[cfg(test)]
    pub fn zeroed(steered: Tensor, field_dir: Tensor) -> Self {
        Self {
            steered,
            field_dir,
            grad_mag: 0.0,
            splat_mag: 0.0,
            goal_mag: 0.0,
            ocean_mag: 0.0,
            memory_ranked: false,
            baseline_norm: 0.0,
            steered_norm: 0.0,
            pullback: 1.0,
            delta_h_norm: 0.0,
            ramp: 1.0,
            scar_pot: 0.0,
            memory_warm: false,
            eureka_boost: 1.0,
            clip_frac: 0.0,
            physics_blend: 1.0,
            qsma_beta: 0.0,
            kinetic_noise: 0.0,
        }
    }
}

/// Original niodoo `apply_request_effects` knobs. Hands fire these; cooldown is
/// not the invention and is not implemented here.
#[derive(Debug, Clone)]
pub struct Hands {
    pub physics_blend: f32,
    pub dynamic_repulsion: f32,
    pub adrenaline: f32,
    pub beta: f64,
    pub beta_from_hand: bool,
    pub kinetic_noise: f32,
    pub focus_lock_remaining: usize,
    pub focus_lock_max: usize,
    pub heartbeat_blend: f32,
    pub heartbeat_repulsion: f32,
    pub heartbeat_goal_scale: f32,
    pub last_request: Option<&'static str>,
}

impl Default for Hands {
    fn default() -> Self {
        Self {
            physics_blend: 1.0,
            dynamic_repulsion: 0.0,
            adrenaline: 0.0,
            beta: 0.0,
            beta_from_hand: false,
            kinetic_noise: 0.0,
            focus_lock_remaining: 0,
            focus_lock_max: 20,
            heartbeat_blend: 1.0,
            heartbeat_repulsion: 0.0,
            heartbeat_goal_scale: 0.15,
            last_request: None,
        }
    }
}

impl Hands {
    pub fn idle() -> Self {
        Self::default()
    }
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
    /// Scar-potential gate: F_s skips early ramp when pot ≥ this. 0 = off.
    memory_warm_pot: f32,
    /// Blend steered hidden toward a topic-matched bridge μ. 0 = off.
    topic_mix: f32,
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
    /// Ranked scar picker (soft-sum remains default / ablation path).
    memory_pick: MemoryPickConfig,
    /// Current prompt FNV fingerprint (bridge scar match term).
    prompt_fp: u32,
    /// Last-step top-k entropy (nats) for selective pick gate.
    pick_entropy: f32,
    /// Last-step confidence margin (p1−p2 or p_chosen proxy).
    pick_margin: f32,
    /// True once generation has reported a real quality sample.
    pick_context_ready: bool,
    /// Path B hands (blend / repulsion / β / σ). Default is idle identity.
    pub hands: Hands,
    /// Detect tags and log receipts, but do not mutate β / σ / blend.
    /// REFUSE arm of the Path B IFEval/TruthfulQA bench (`HYDRO_TAGS_DETECT_ONLY`).
    pub tags_detect_only: bool,
    /// Per-token flux F(s,a) for QSMA.
    qsma_flux: HashMap<u32, f64>,
    /// Durable remember-store (same JSONL as niodoo partner/remember).
    remember: RememberStore,
    /// ORG-H5 dormant why-geometry sidecar. Not a RememberLine column.
    remember_probe: RememberOffsetProbe,
    /// Last steer Δh (for tag-ablation / collapse proof).
    last_delta_h_norm: f32,
    last_qsma_beta: f32,
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
            memory_warm_pot: 0.0,
            topic_mix: 0.0,
            noise_sigma: 0.3,
            viscosity: viscosity_scale,
            tag_gravity_mult: 1.0,
            eureka_impulse: 0.0,
            eureka_target: None,
            ocean: None,
            memory_pick: MemoryPickConfig::default(),
            prompt_fp: 0,
            // Pre-sample: treat as unsettled so early tokens can hard-pick bridges.
            pick_entropy: 99.0,
            pick_margin: 0.0,
            pick_context_ready: false,
            hands: Hands {
                heartbeat_goal_scale: 0.15,
                ..Hands::idle()
            },
            tags_detect_only: std::env::var("HYDRO_TAGS_DETECT_ONLY")
                .ok()
                .is_some_and(|v| {
                    matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "on" | "yes"
                    )
                }),
            qsma_flux: HashMap::new(),
            remember: RememberStore::default(),
            remember_probe: RememberOffsetProbe::default(),
            last_delta_h_norm: 0.0,
            last_qsma_beta: 0.0,
        }
    }

    /// Configure ranked memory picker (feature-gated; mode=Soft keeps legacy path).
    pub fn set_memory_pick(&mut self, pick: MemoryPickConfig) {
        self.memory_pick = pick;
    }

    pub fn memory_pick(&self) -> &MemoryPickConfig {
        &self.memory_pick
    }

    /// Prompt fingerprint for bridge scar match scoring.
    pub fn set_prompt_fp(&mut self, fp: u32) {
        self.prompt_fp = fp;
    }

    pub fn prompt_fp(&self) -> u32 {
        self.prompt_fp
    }

    /// Feed posterior entropy + confidence margin for the selective pick gate.
    /// Call once per token after quality scoring (previous step gates next steer).
    pub fn set_pick_context(&mut self, entropy: f32, margin: f32) {
        self.pick_entropy = entropy;
        self.pick_margin = margin.clamp(0.0, 1.0);
        self.pick_context_ready = true;
    }

    /// Residual unsettled? → allow ranked hard pick. Settled → soft-sum fallback.
    fn memory_pick_unsettled(&self, pos: &Tensor, goal_pos: &Tensor) -> Result<bool> {
        if self.memory_pick.mode != MemoryForceMode::Ranked {
            return Ok(false);
        }
        if !self.memory_pick.selective {
            return Ok(true);
        }
        if self.pick_entropy >= self.memory_pick.entropy_min {
            return Ok(true);
        }
        if self.pick_margin <= self.memory_pick.margin_max {
            return Ok(true);
        }
        if self.memory_pick.residual_l2_min > 0.0 {
            let dist: f32 = (goal_pos - pos)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt();
            if dist >= self.memory_pick.residual_l2_min {
                return Ok(true);
            }
        }
        // No quality sample yet: hard-pick (bridges matter at step 0).
        if !self.pick_context_ready {
            return Ok(true);
        }
        Ok(false)
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
        crate::hud::hud_quiet_println!(
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

    /// When pot ≥ `threshold`, F_s is not early-ramped. 0 disables the gate.
    pub fn set_memory_warm_pot(&mut self, threshold: f32) {
        self.memory_warm_pot = threshold.max(0.0);
    }

    /// Blend steered residual toward a topic-matched bridge. 0 disables.
    pub fn set_topic_mix(&mut self, mix: f32) {
        self.topic_mix = mix.clamp(0.0, 1.0);
    }

    /// Live force cap — the σ knob in the √-law readout.
    pub fn force_cap(&self) -> f32 {
        self.force_cap
    }

    /// Live per-force ceilings, for saturation marking in the readout.
    pub fn field_wake_max(&self) -> f32 {
        self.field_wake.max_mag
    }

    pub fn splat_force_max(&self) -> f32 {
        self.splat_force_max
    }

    pub fn goal_force_max(&self) -> f32 {
        self.goal_force_max
    }

    /// Live goal-attractor scale — the θ knob in the √-law readout.
    pub fn goal_force_scale(&self) -> f32 {
        self.goal_force_scale
    }

    /// Residual-surface parameters exposed to the operator REPL.
    pub fn live_params(&self) -> Vec<(&'static str, f32, f32, f32)> {
        vec![
            ("residual.cap", self.force_cap, 0.0, 20.0),
            ("residual.dt", self.dt, 0.001, 0.2),
            ("residual.field", self.field_wake.scale, 0.0, 2.0),
            ("residual.splat", self.splat_force_scale, 0.0, 2.0),
            ("residual.goal", self.goal_force_scale, 0.0, 2.0),
            // Per-force ceilings. These are what actually bind on a small model:
            // a saturated F_g is a constant shove, not steering, and no amount of
            // scale tuning shows it — only the ceiling does.
            ("residual.field_max", self.field_wake.max_mag, 0.0, 100.0),
            ("residual.splat_max", self.splat_force_max, 0.0, 100.0),
            ("residual.goal_max", self.goal_force_max, 0.0, 100.0),
            ("force_ramp_tokens", self.force_ramp_tokens as f32, 0.0, 200.0),
            ("force_ramp_start", self.force_ramp_start, 0.0, 1.0),
            (
                "hands.repulsion",
                self.hands.dynamic_repulsion,
                -5.0,
                5.0,
            ),
            ("hands.beta", self.hands.beta as f32, 0.0, 5.0),
            ("hands.blend", self.hands.physics_blend, 0.0, 10.0),
        ]
    }

    /// Set one live residual / ramp / hands parameter, clamped to `live_params`.
    ///
    /// Formula-native σ/θ/β are not accepted here. `residual.cap` is the Hydro
    /// force ceiling, not Algo σ.
    pub fn set_live_param(&mut self, name: &str, value: f32) -> bool {
        match name {
            "residual.cap" => self.force_cap = value.clamp(0.0, 20.0),
            "residual.dt" => self.dt = value.clamp(0.001, 0.2),
            "residual.field" => self.field_wake.scale = value.clamp(0.0, 2.0),
            "residual.splat" => self.splat_force_scale = value.clamp(0.0, 2.0),
            "residual.goal" => {
                self.goal_force_scale = value.clamp(0.0, 2.0);
                self.hands.heartbeat_goal_scale = self.goal_force_scale;
            }
            "residual.field_max" => self.field_wake.max_mag = value.clamp(0.0, 100.0),
            "residual.splat_max" => self.splat_force_max = value.clamp(0.0, 100.0),
            "residual.goal_max" => self.goal_force_max = value.clamp(0.0, 100.0),
            "force_ramp_tokens" => self.force_ramp_tokens = value.clamp(0.0, 200.0) as usize,
            "force_ramp_start" => self.force_ramp_start = value.clamp(0.0, 1.0),
            "hands.repulsion" => self.hands.dynamic_repulsion = value.clamp(-5.0, 5.0),
            "hands.beta" => {
                self.hands.beta = value.clamp(0.0, 5.0) as f64;
                self.hands.beta_from_hand = true;
            }
            "hands.blend" => self.hands.physics_blend = value.clamp(0.0, 10.0),
            _ => return false,
        }
        true
    }

    pub fn render_live_sliders(&self) -> String {
        let mut out = String::from("  residual physics\n");
        for (name, value, min, max) in self.live_params() {
            let span = (max - min).max(1e-8);
            let filled = (((value - min) / span).clamp(0.0, 1.0) * 24.0).round() as usize;
            let bar: String = (0..24)
                .map(|i| if i < filled { '#' } else { '.' })
                .collect();
            out.push_str(&format!(
                "    {:<16} [{}] {:>8.4}   ({} … {})\n",
                name, bar, value, min, max
            ));
        }
        out
    }

    /// Whether the residual surface can move the state. A zero cap is an exact
    /// runtime bypass used by force-off profiles and the live REPL.
    pub fn residual_enabled(&self) -> bool {
        self.force_cap > 1e-8
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
        self.hands.heartbeat_goal_scale = self.goal_force_scale;
    }

    pub fn open_remember_store(&mut self, path: &Path) {
        match RememberStore::open(path) {
            Ok(store) => {
                crate::hud::hud_quiet_println!(
                    "[NIODOO] remember-store {} ({} keys)",
                    store.path.display(),
                    store.lines.len()
                );
                self.remember = store;
            }
            Err(e) => {
                crate::hud::hud_quiet_println!(
                    "[NIODOO] remember-store open failed ({e}); starting empty at {}",
                    path.display()
                );
                self.remember.path = path.to_path_buf();
                self.remember.lines.clear();
            }
        }
        self.remember_probe
            .set_sidecar_from_remember_path(&self.remember.path);
    }

    /// Decode-step residual / pre-unembed hidden for the H5 sidecar ring.
    pub fn push_remember_hidden(
        &mut self,
        step: usize,
        s_res: Option<&[f32]>,
        s_logit: Option<&[f32]>,
    ) {
        self.remember_probe.push(step, s_res, s_logit);
    }

    pub fn note_remember_pieces(&mut self, pieces: &str) {
        self.remember_probe.note_pieces(pieces);
    }

    /// KV drop is not a why-mint.
    pub fn on_kv_drop(&mut self) {
        self.remember_probe.on_kv_drop();
    }

    pub fn remember_probe_mint_count(&self) -> usize {
        self.remember_probe.mint_count()
    }

    pub fn remember_probe_sidecar(&self) -> &Path {
        self.remember_probe.sidecar_path()
    }

    pub fn remember_len(&self) -> usize {
        self.remember.lines.len()
    }

    pub fn remember_get(&self, key: &str) -> Option<String> {
        self.remember.get(key).map(|l| l.value.clone())
    }

    /// Idle identity (not original RESET). Used between ablation arms.
    pub fn restore_idle_hands(&mut self) {
        let hb_goal = self.hands.heartbeat_goal_scale;
        self.goal_force_scale = hb_goal;
        self.hands = Hands {
            heartbeat_goal_scale: hb_goal,
            ..Hands::idle()
        };
    }

    pub fn qsma_beta(&self, step: usize) -> f64 {
        if self.hands.beta_from_hand {
            self.hands.beta
        } else {
            crate::qsma::beta(step as u32)
        }
    }

    pub fn apply_qsma_logits(
        &mut self,
        logits: &mut [f32],
        generated: &[u32],
        step: usize,
    ) -> crate::qsma::QsmaAction {
        let beta = self.qsma_beta(step);
        self.last_qsma_beta = beta as f32;
        crate::qsma::bias_top_k(
            logits,
            64,
            &self.qsma_flux,
            generated,
            beta,
            self.hands.kinetic_noise as f64,
        )
    }

    /// Path B telemetry for ablation / collapse (not T/rep).
    pub fn hands_report(&self) -> serde_json::Value {
        serde_json::json!({
            "physics_blend": self.hands.physics_blend,
            "dynamic_repulsion": self.hands.dynamic_repulsion,
            "qsma_beta": self.last_qsma_beta,
            "hand_beta": self.hands.beta,
            "beta_from_hand": self.hands.beta_from_hand,
            "kinetic_noise": self.hands.kinetic_noise,
            "adrenaline": self.hands.adrenaline,
            "last_request": self.hands.last_request,
            "delta_h_norm": self.last_delta_h_norm,
            "force_cap": self.force_cap,
            "residual_enabled": self.residual_enabled(),
            "remember_keys": self.remember.lines.len(),
        })
    }

    pub fn observe_token(&mut self, token: u32, energy: f64) {
        let slot = self.qsma_flux.entry(token).or_insert(0.0);
        *slot = crate::qsma::update_flux(*slot, energy);
    }

    pub fn tick_hands(&mut self) {
        if self.hands.focus_lock_remaining > 0 {
            self.hands.focus_lock_remaining -= 1;
        }
    }

    fn env_truth(name: &str) -> bool {
        std::env::var(name).ok().is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    }

    /// Restore heartbeat Path B knobs so one item cannot bleed SPIKE/FOCUS into the next.
    pub fn reset_path_b_hands(&mut self) {
        let hb = self.hands.heartbeat_blend;
        let hr = self.hands.heartbeat_repulsion;
        let hg = self.hands.heartbeat_goal_scale;
        self.hands = Hands {
            physics_blend: hb,
            dynamic_repulsion: hr,
            heartbeat_blend: hb,
            heartbeat_repulsion: hr,
            heartbeat_goal_scale: hg,
            ..Hands::idle()
        };
        self.goal_force_scale = hg;
    }

    /// Original `apply_request_effects`. No accept/refuse gate. LOCK is not a hand.
    /// Emitted physics hands always write β / σ / blend.
    pub fn apply_request_effects(&mut self, req: ControlTag) {
        const FOCUS_GRAVITY_SCALE: f32 = 1.35;
        match req {
            ControlTag::Spike => {
                self.hands.focus_lock_remaining = 0;
                self.hands.adrenaline = 5.0;
                self.hands.physics_blend = 6.5;
                self.hands.dynamic_repulsion = -3.0;
                self.hands.beta = 1.5;
                self.hands.beta_from_hand = true;
                self.hands.kinetic_noise = 1.5;
                self.goal_force_scale = self.hands.heartbeat_goal_scale;
                self.hands.last_request = Some("spike");
            }
            ControlTag::Focus => {
                self.hands.focus_lock_remaining = self.hands.focus_lock_max;
                self.hands.physics_blend = 0.5;
                self.hands.dynamic_repulsion = self.hands.heartbeat_repulsion;
                self.hands.adrenaline = 0.0;
                self.hands.beta = 0.5;
                self.hands.beta_from_hand = true;
                self.hands.kinetic_noise = 0.05;
                let base = self.goal_force_scale.max(self.hands.heartbeat_goal_scale);
                self.goal_force_scale = base * FOCUS_GRAVITY_SCALE;
                self.hands.last_request = Some("focus");
            }
            ControlTag::Explore => {
                self.hands.focus_lock_remaining = 0;
                self.hands.physics_blend = 2.0;
                self.hands.dynamic_repulsion = -2.0;
                self.hands.adrenaline = 3.0;
                self.hands.beta = 2.0;
                self.hands.beta_from_hand = true;
                self.hands.kinetic_noise = 0.8;
                self.goal_force_scale = self.hands.heartbeat_goal_scale;
                self.hands.last_request = Some("explore");
            }
            ControlTag::Reset => {
                self.hands.focus_lock_remaining = 0;
                self.hands.adrenaline = 0.0;
                self.hands.physics_blend = 1.5;
                self.hands.dynamic_repulsion = -0.5;
                self.hands.beta = 0.0;
                self.hands.beta_from_hand = true;
                self.hands.kinetic_noise = 0.0;
                self.goal_force_scale = self.hands.heartbeat_goal_scale;
                self.hands.last_request = Some("reset");
            }
            ControlTag::Remember => {
                self.hands.focus_lock_remaining = 0;
                self.hands.physics_blend = 1.0;
                self.hands.dynamic_repulsion = 0.0;
                self.hands.adrenaline = 1.0;
                self.hands.last_request = Some("remember");
            }
            ControlTag::Lock => {}
        }
        // Hands fire in physics. The mouth only shows the tag she emitted.
        // Do not print CONTROL_RECEIPT — that is operator telemetry, not chat.
    }

    /// Emit → apply. Remember/Lock persist to the seat store. Lock does not FOCUS.
    /// Returns true when generation should stop (LOCK).
    /// There is no tag refusal. `HYDRO_LOCK_STOP_OFF=1` keeps LOCK from killing IFEval.
    pub fn fire_tag(&mut self, hit: &crate::control_tags::TagHit) -> bool {
        if let Some(payload) = hit.payload.as_deref() {
            if matches!(hit.tag, ControlTag::Remember | ControlTag::Lock) && !payload.is_empty() {
                match self.remember.upsert(payload) {
                    Ok(line) => {
                        crate::hud::hud_quiet_println!(
                            "[REMEMBER] saved {}={}",
                            line.key,
                            line.value
                        );
                        match self
                            .remember_probe
                            .dump_on_remember_close(&line.key, &line.value)
                        {
                            Ok(0) => {}
                            Ok(n) => crate::hud::hud_quiet_println!(
                                "[REMEMBER_PROBE] event=remember_offset_probe key={} rows={} inject=0",
                                line.key,
                                n
                            ),
                            Err(e) => crate::hud::hud_quiet_println!(
                                "[REMEMBER_PROBE] dump failed: {e}"
                            ),
                        }
                    }
                    Err(e) => crate::hud::hud_quiet_println!("[REMEMBER] save failed: {e}"),
                }
            }
        }
        if hit.tag.is_physics_hand() {
            self.apply_request_effects(hit.tag);
        }
        hit.tag.stops_turn() && !Self::env_truth("HYDRO_LOCK_STOP_OFF")
    }

    /// Decode-loop Path B: newly appeared tags in `pieces` fire hands.
    /// Returns (lock_stop, tags applied this call).
    pub fn apply_emitted_tags(
        &mut self,
        pieces: &str,
        tags_seen: &mut Vec<crate::control_tags::TagHit>,
    ) -> (bool, Vec<ControlTag>) {
        let found = crate::control_tags::scan_hits(pieces);
        let mut stop = false;
        let mut applied = Vec::new();

        // A block tag arrives incrementally as `<remember>` (payload None), then
        // upgrades in the same slot once `</remember>` closes. Persist that upgrade
        // exactly once without depositing a second residual hand splat.
        for idx in 0..tags_seen.len().min(found.len()) {
            if tags_seen[idx] != found[idx] {
                let hit = found[idx].clone();
                if hit.tag == tags_seen[idx].tag
                    && tags_seen[idx].payload.is_none()
                    && hit.payload.is_some()
                {
                    if self.fire_tag(&hit) {
                        stop = true;
                    }
                    tags_seen[idx] = hit;
                }
            }
        }

        while tags_seen.len() < found.len() {
            let hit = found[tags_seen.len()].clone();
            if self.fire_tag(&hit) {
                stop = true;
            }
            applied.push(hit.tag);
            tags_seen.push(hit);
        }
        (stop, applied)
    }

    /// Decode-loop Path B with residual write: emit → hand → splat at the live
    /// residual. Later tokens query that scar. Not JSONL, not env inject.
    /// `residual_pos` is the decode hidden that just produced the tagged token.
    pub fn apply_emitted_control(
        &mut self,
        pieces: &str,
        tags_seen: &mut Vec<crate::control_tags::TagHit>,
        residual_pos: Option<&Tensor>,
    ) -> Result<(bool, Vec<ControlTag>)> {
        let (stop, applied) = self.apply_emitted_tags(pieces, tags_seen);
        if applied.is_empty() {
            return Ok((stop, applied));
        }
        if let Some(pos) = residual_pos {
            if self.residual_enabled() {
                let pos = if pos.dims().len() > 1 {
                    pos.flatten_all()?
                } else {
                    pos.clone()
                };
                for tag in &applied {
                    if !tag.is_physics_hand() {
                        continue;
                    }
                    let alpha = match tag {
                        ControlTag::Spike | ControlTag::Explore => 0.9,
                        ControlTag::Focus | ControlTag::Remember => 0.75,
                        _ => 0.5,
                    };
                    let splat = crate::splat::Splat::new(pos.clone(), 12.0, alpha);
                    self.memory.add_splat(splat);
                }
            }
        }
        Ok((stop, applied))
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

    /// Live residual width from the Diderot field (GGUF embedding_length).
    #[inline]
    pub fn residual_dim(&self) -> usize {
        self.field.dim
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
        // Shape validation: require exactly (1, D) with D = live GGUF residual width
        let expected_d = self.residual_dim();
        crate::dim_assert::assert_last_dim(
            baseline_residual,
            expected_d,
            "steer.baseline_residual",
        )?;
        crate::dim_assert::assert_last_dim(goal_pos, expected_d, "steer.goal_pos")?;
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
        crate::dim_assert::assert_last_dim(&pos, expected_d, "steer.pos")?;

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

        let mut grad_force = match self.field_wake.mode {
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
        // Scale + soft L2 cap on field force (was uncapped pure ∇ρ → F_g~10–37 shredding text).
        // System stays ON; field_wake_scale / field_wake_max are the dials.
        if self.field_wake.scale != 1.0 {
            grad_force = grad_force.affine(self.field_wake.scale as f64, 0.0)?;
        }
        if self.field_wake.max_mag > 0.0 {
            let mag: f32 = grad_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if mag > self.field_wake.max_mag && mag > 1e-8 {
                grad_force = grad_force.affine((self.field_wake.max_mag / mag) as f64, 0.0)?;
            }
        }

        // 2. Learned-will force — soft-sum (legacy) or ranked Top-K picker
        //
        // Soft path: PhysicsBackend → SplatMemory::query_force_soft (all scars).
        // Ranked path: only when mode=Ranked and residual unsettled (selective gate);
        // settled residual falls back to soft-sum for ablation-safe continuity.
        let memory_ranked = self.memory_pick_unsettled(&pos, goal_pos)?;
        let mut splat_force = if memory_ranked {
            self.memory
                .query_force_with_pick(&pos, &self.memory_pick, self.prompt_fp, true)?
        } else {
            self.backend.splat_force(&self.memory, &pos)?
        };
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

        // Related-prompt coupling: pull toward a prefill-bridge whose stored fp
        // matches this turn's topic/prompt fingerprint even when L2 is COLD.
        let topic_hit = self.prompt_fp != 0 && self.memory.has_matching_bridge(self.prompt_fp);
        if topic_hit {
            if let Some(mu) = self.memory.matched_bridge_mu(self.prompt_fp)? {
                let mu = if mu.dims().len() == 2 {
                    mu.squeeze(0)?
                } else {
                    mu
                };
                if mu.dims() == pos.dims() {
                    let pull =
                        (&mu - &pos)?.affine(self.splat_force_scale.max(0.03) as f64, 0.0)?;
                    splat_force = (&splat_force + &pull)?;
                }
            }
        }

        // `residual.splat_max` is the ceiling on the final learned-will force,
        // including topic-matched bridge coupling. The historical cap above only
        // covered the memory query; the later bridge pull could bypass it by >5×.
        if self.splat_force_max > 0.0 {
            let mag: f32 = splat_force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if mag > self.splat_force_max && mag > 1e-8 {
                splat_force = splat_force.affine((self.splat_force_max / mag) as f64, 0.0)?;
            }
        }

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

        // Sum and scale by dt — every force vector must share live residual width
        crate::dim_assert::assert_last_dim(&grad_force, expected_d, "steer.grad_force")?;
        crate::dim_assert::assert_last_dim(&splat_force, expected_d, "steer.splat_force")?;
        crate::dim_assert::assert_last_dim(&goal_force, expected_d, "steer.goal_force")?;
        crate::dim_assert::assert_last_dim(&ocean_force, expected_d, "steer.ocean_force")?;

        let scar_pot = if self.memory.len() > 0 {
            self.memory.query_potential(&pos).unwrap_or(0.0)
        } else {
            0.0
        };
        let memory_warm =
            (self.memory_warm_pot > 1e-8 && scar_pot >= self.memory_warm_pot) || topic_hit;

        // Ramp: gentler early (J-space / prefill respect). Linear start→1 over N tokens.
        // Matching basin: F_s skips the early ramp so particular scars can move first tokens.
        let mut ramp = 1.0f32;
        let mut grad_f = grad_force;
        let mut splat_f = splat_force;
        let mut goal_f = goal_force;
        let mut ocean_f = ocean_force;
        if self.force_ramp_tokens > 0 {
            let t = (step as f32 / self.force_ramp_tokens as f32).min(1.0);
            ramp = self.force_ramp_start + (1.0 - self.force_ramp_start) * t;
            if (ramp - 1.0).abs() > 1e-8 {
                grad_f = grad_f.affine(ramp as f64, 0.0)?;
                goal_f = goal_f.affine(ramp as f64, 0.0)?;
                ocean_f = ocean_f.affine(ramp as f64, 0.0)?;
                if !memory_warm {
                    splat_f = splat_f.affine(ramp as f64, 0.0)?;
                }
            }
        }
        let mut total_force = (((&grad_f + &splat_f)? + &goal_f)? + &ocean_f)?;
        crate::dim_assert::assert_last_dim(&total_force, expected_d, "steer.total_force")?;

        // Original dynamic_repulsion is a black-hole coefficient. Mapped here as a
        // radial residual term so later tokens move in physics state, not T/rep.
        if self.hands.dynamic_repulsion.abs() > 1e-8 {
            let pos_mag: f32 = pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if pos_mag > 1e-8 {
                let radial = pos.affine((self.hands.dynamic_repulsion / pos_mag) as f64, 0.0)?;
                total_force = (&total_force + &radial)?;
            }
        }

        // Endocrine Eureka: soft force boost + optional pull toward **native** bloom embed.
        let mut eureka_boost = 1.0f32;
        if self.eureka_impulse > 1e-4 {
            let boost = (1.0 + 0.12 * self.eureka_impulse.min(5.0)) as f64;
            eureka_boost = boost as f32;
            total_force = total_force.affine(boost, 0.0)?;
            if let Some(ref target) = self.eureka_target {
                // Pull residual toward native mean tok-emb of bloom text (same D as pos).
                if target.dims() == pos.dims() {
                    let pull = (target - &pos)?
                        .affine((0.04 * self.eureka_impulse.min(5.0)) as f64, 0.0)?;
                    total_force = (&total_force + &pull)?;
                }
            }
        }

        // Cap telemetry: how much of the force vector the cap is about to clip.
        // Measured pre-clamp — post-clamp the evidence is gone.
        let clip_frac: f32 = total_force
            .abs()?
            .gt(self.force_cap)?
            .to_dtype(candle_core::DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;

        // Force cap: prevent any single dimension from dominating (Variant 3)
        let total_force = total_force.clamp(-self.force_cap, self.force_cap)?;
        // Path B hands: original physics_blend (SPIKE 6.5) is a mix strength, not a
        // pre-cap scale. Live Gemma cap is 0.2; scaling before clamp made SPIKE a no-op.
        let mut steering = total_force.affine(self.dt as f64, 0.0)?;
        if (self.hands.physics_blend - 1.0).abs() > 1e-8 {
            steering = steering.affine(self.hands.physics_blend as f64, 0.0)?;
        }
        let delta_h_norm: f32 = steering.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        self.last_delta_h_norm = delta_h_norm;

        // Restore batch dim: (D,) -> (1, D) and add to baseline
        let steering_2d = steering.unsqueeze(0)?;
        crate::dim_assert::assert_last_dim(&steering_2d, expected_d, "steer.steering_2d")?;
        let steered = (baseline_residual + &steering_2d)?;
        crate::dim_assert::assert_last_dim(&steered, expected_d, "steer.residual_add")?;

        // === RENORMALIZATION: stay on the model manifold ===
        // Without this, cumulative steering drifts the hidden state norm,
        // causing lm_head to produce garbage after ~40-80 tokens.
        let baseline_norm: f32 = baseline_residual
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let steered_norm: f32 = steered.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        let mut pullback = 1.0f32;
        let steered = if !self.residual_enabled() {
            // Preserve exact model output for the force-off control. Renormalizing
            // baseline + 0 introduces a small roundoff delta and invalidates A/B logs.
            baseline_residual.clone()
        } else if steered_norm > 0.0 && baseline_norm > 0.0 {
            pullback = baseline_norm / steered_norm;
            steered.affine(pullback as f64, 0.0)?
        } else {
            steered
        };

        // Topic mix: lm_head reads a residual on the minted scar, not only a
        // capped F_s nudge. Isolation leaves topic_mix=0 so this is a no-op.
        let steered = if self.residual_enabled() && topic_hit && self.topic_mix > 1e-6 && step < 6 {
            if let Some(mu) = self.memory.matched_bridge_mu(self.prompt_fp)? {
                let mu = if mu.dims().len() == 1 {
                    mu.unsqueeze(0)?
                } else {
                    mu
                };
                if mu.dims() == steered.dims() && baseline_norm > 1e-8 {
                    let mu_n: f32 = mu.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    if mu_n > 1e-8 {
                        let mu_on_shell = mu.affine((baseline_norm / mu_n) as f64, 0.0)?;
                        let l = self.topic_mix as f64;
                        let mixed =
                            (&steered.affine(1.0 - l, 0.0)? + &mu_on_shell.affine(l, 0.0)?)?;
                        let mn: f32 = mixed.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                        if mn > 1e-8 {
                            mixed.affine((baseline_norm / mn) as f64, 0.0)?
                        } else {
                            steered
                        }
                    } else {
                        steered
                    }
                } else {
                    steered
                }
            } else {
                steered
            }
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
            memory_ranked,
            baseline_norm,
            steered_norm,
            pullback,
            delta_h_norm,
            ramp,
            scar_pot,
            memory_warm,
            eureka_boost,
            clip_frac,
            physics_blend: self.hands.physics_blend,
            qsma_beta: self.qsma_beta(step) as f32,
            kinetic_noise: self.hands.kinetic_noise,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::CpuBackend;
    use candle_core::{Device, Tensor};

    #[test]
    #[test]
    fn set_live_param_hands_and_ramp_are_not_formula_beta() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        assert!(engine.set_live_param("hands.beta", 1.5));
        assert!((engine.hands.beta - 1.5).abs() < 1e-12);
        assert!(engine.hands.beta_from_hand);
        assert!(!engine.set_live_param("beta", 100.0), "Algo β is not hands.beta");
        assert!(engine.set_live_param("force_ramp_tokens", 24.0));
        assert_eq!(engine.force_ramp_tokens, 24);
        assert!(engine.set_live_param("hands.blend", 6.5));
        assert!((engine.hands.physics_blend - 6.5).abs() < 1e-6);
    }

    #[test]
    fn zero_force_cap_is_an_exact_residual_bypass() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 0.0);
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
        let result = engine.steer(&baseline, &goal, 0).unwrap();
        let before: Vec<f32> = baseline.flatten_all().unwrap().to_vec1().unwrap();
        let after: Vec<f32> = result.steered.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(after, before);
        assert!(!engine.residual_enabled());
    }

    #[test]
    fn spike_always_applies_even_if_detect_only_flag_is_set() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        engine.tags_detect_only = true;
        engine.apply_request_effects(ControlTag::Spike);
        assert_eq!(engine.hands.last_request, Some("spike"));
        assert!((engine.hands.physics_blend - 6.5).abs() < 1e-6);
        assert!((engine.hands.beta - 1.5).abs() < 1e-12);
        assert!((engine.hands.kinetic_noise - 1.5).abs() < 1e-6);
    }

    #[test]
    fn spike_hand_sets_original_blend_beta_sigma() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        engine.apply_request_effects(ControlTag::Spike);
        let report = engine.hands_report();
        assert_eq!(report["physics_blend"], 6.5);
        assert_eq!(report["last_request"], "spike");
        assert_eq!(report["residual_enabled"], true);
        assert!((engine.hands.physics_blend - 6.5).abs() < 1e-6);
        assert!((engine.hands.dynamic_repulsion + 3.0).abs() < 1e-6);
        assert!((engine.hands.beta - 1.5).abs() < 1e-12);
        assert!((engine.hands.kinetic_noise - 1.5).abs() < 1e-6);
        assert!(engine.hands.beta_from_hand);
    }

    #[test]
    fn spike_hand_changes_later_residual_delta() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut idle =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut spiked =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        spiked.apply_request_effects(ControlTag::Spike);
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
        let a = idle.steer(&baseline, &goal, 12).unwrap();
        let b = spiked.steer(&baseline, &goal, 12).unwrap();
        assert!(
            (b.delta_h_norm - a.delta_h_norm).abs() > 1e-6
                || (b.physics_blend - a.physics_blend).abs() > 1.0,
            "SPIKE must move physics state (delta_h or blend); idle Δh={} spike Δh={} blend {}",
            a.delta_h_norm,
            b.delta_h_norm,
            b.physics_blend
        );
        assert!((b.physics_blend - 6.5).abs() < 1e-6);
        assert!(b.qsma_beta > 0.0);
    }

    #[test]
    fn lock_is_not_a_focus_hand() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let hit = crate::control_tags::TagHit {
            tag: ControlTag::Lock,
            payload: Some("k=v".into()),
        };
        assert!(engine.fire_tag(&hit));
        assert_ne!(engine.hands.last_request, Some("focus"));
        assert!((engine.hands.physics_blend - 1.0).abs() < 1e-6);
    }

    #[test]
    fn remember_survives_engine_drop_on_same_seat() {
        let device = Device::Cpu;
        let dir = std::env::temp_dir().join(format!(
            "hydro_seat_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("seat_remember.jsonl");
        {
            let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
            let memory = SplatMemory::new(device.clone());
            let mut engine =
                NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
            engine.open_remember_store(&path);
            let hit = crate::control_tags::TagHit {
                tag: ControlTag::Remember,
                payload: Some("tuesday-boy=13/27".into()),
            };
            assert!(!engine.fire_tag(&hit), "remember must not stop decode");
            assert_eq!(engine.remember_get("tuesday-boy").as_deref(), Some("13/27"));
            assert_eq!(engine.hands.last_request, Some("remember"));
        }
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut revived =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        revived.open_remember_store(&path);
        assert_eq!(
            revived.remember_get("tuesday-boy").as_deref(),
            Some("13/27")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streamed_remember_payload_upgrades_the_existing_cursor_slot() {
        let device = Device::Cpu;
        let dir = std::env::temp_dir().join(format!(
            "hydro_stream_remember_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("seat_remember.jsonl");
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        engine.open_remember_store(&path);
        let mut cursor = Vec::new();

        let (stop_open, applied_open) = engine.apply_emitted_tags("<remember>", &mut cursor);
        assert!(!stop_open);
        assert_eq!(applied_open, vec![ControlTag::Remember]);
        assert_eq!(cursor.len(), 1);
        assert_eq!(engine.remember_get("protocol"), None);

        let (stop_closed, applied_closed) = engine.apply_emitted_tags(
            "<remember>protocol=runtime_behavior</remember>",
            &mut cursor,
        );
        assert!(!stop_closed);
        assert!(
            applied_closed.is_empty(),
            "payload upgrade must not deposit a second hand splat"
        );
        assert_eq!(
            engine.remember_get("protocol").as_deref(),
            Some("runtime_behavior")
        );
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remember_geometry_sidecar_on_close_not_on_spike_or_kv_drop() {
        let device = Device::Cpu;
        let dir = std::env::temp_dir().join(format!(
            "hydro_h5_engine_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("seat_remember.jsonl");
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        engine.open_remember_store(&path);
        let scars0 = engine.memory().len();
        for i in 0..8 {
            let v = [(i as f32) + 0.1, 0.2, 0.3, 0.4];
            let l = [9.0, i as f32, 0.0, 1.0];
            engine.push_remember_hidden(i, Some(&v), Some(&l));
        }
        engine.note_remember_pieces("wait <spike>");
        let spike = crate::control_tags::TagHit {
            tag: ControlTag::Spike,
            payload: None,
        };
        assert!(!engine.fire_tag(&spike));
        assert_eq!(engine.remember_probe_mint_count(), 0);
        assert!(!engine.remember_probe_sidecar().exists());

        engine.note_remember_pieces("<remember>lumina=why-vector</remember>");
        let hit = crate::control_tags::TagHit {
            tag: ControlTag::Remember,
            payload: Some("lumina=why-vector".into()),
        };
        assert!(!engine.fire_tag(&hit));
        assert_eq!(engine.remember_get("lumina").as_deref(), Some("why-vector"));
        assert_eq!(engine.remember_probe_mint_count(), 1);
        assert_eq!(engine.memory().len(), scars0, "sidecar must not write a splat");
        let sidecar = engine.remember_probe_sidecar().to_path_buf();
        let body = std::fs::read_to_string(&sidecar).unwrap();
        assert!(body.contains("remember_offset_probe"));
        assert!(body.contains("\"inject\":false"));
        assert!(!body.contains("\"geometry\""));
        let remember_body = std::fs::read_to_string(&path).unwrap();
        let line: serde_json::Value = serde_json::from_str(remember_body.lines().next().unwrap()).unwrap();
        assert!(line.get("geometry").is_none(), "RememberLine stays payload-only");
        assert_eq!(line["key"], "lumina");

        engine.on_kv_drop();
        assert_eq!(engine.remember_probe_mint_count(), 1);
        let bytes = std::fs::metadata(&sidecar).unwrap().len();
        engine.on_kv_drop();
        assert_eq!(std::fs::metadata(&sidecar).unwrap().len(), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_spike_mid_decode_perturbs_later_residual_and_qsma() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut live =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();

        let pre = live.steer(&baseline, &goal, 4).unwrap();
        assert!((pre.physics_blend - 1.0).abs() < 1e-6);
        let scheduled_beta = live.qsma_beta(4);
        live.hands.kinetic_noise = 0.0;
        let mut logits_pre = vec![0.2f32, 1.1, 0.8, 0.4];
        live.apply_qsma_logits(&mut logits_pre, &[], 4);

        // Model emits into the decode buffer; same helper as generate_turn_ex.
        let mut tags_seen = Vec::new();
        let (stop, applied) = live.apply_emitted_tags("count 1 2 3\n<spike>\n4", &mut tags_seen);
        assert!(!stop);
        assert_eq!(applied, vec![ControlTag::Spike]);
        assert_eq!(tags_seen.len(), 1);
        assert!((live.hands.physics_blend - 6.5).abs() < 1e-6);
        assert!((live.qsma_beta(4) - 1.5).abs() < 1e-12);
        assert!((live.qsma_beta(4) - scheduled_beta).abs() > 1e-6);

        let post = live.steer(&baseline, &goal, 4).unwrap();
        assert!(
            (post.delta_h_norm - pre.delta_h_norm).abs() > 1e-6,
            "later residual must move after emit; pre Δh={} post Δh={}",
            pre.delta_h_norm,
            post.delta_h_norm
        );
        assert!((post.physics_blend - 6.5).abs() < 1e-6);

        live.hands.kinetic_noise = 0.0;
        let mut logits_post = vec![0.2f32, 1.1, 0.8, 0.4];
        live.apply_qsma_logits(&mut logits_post, &[], 4);
        assert!(
            logits_pre
                .iter()
                .zip(logits_post.iter())
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "later QSMA logits must perturb; pre={logits_pre:?} post={logits_post:?}"
        );
    }

    #[test]
    fn emit_control_writes_residual_later_query_reads() {
        use crate::gpu::CpuBackend;

        let device = candle_core::Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut live =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let pos = Tensor::new(&[0.5f32, 0.0, 0.0, 0.0], &device).unwrap();
        let far = Tensor::new(&[800.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let pre = live.memory().query_potential(&pos).unwrap_or(0.0);
        assert!(pre.abs() < 1e-6, "empty store must be ~0, got {pre}");

        let mut tags_seen = Vec::new();
        let (stop, applied) = live
            .apply_emitted_control("count 1\n<spike>\n2", &mut tags_seen, Some(&pos))
            .unwrap();
        assert!(!stop);
        assert_eq!(applied, vec![ControlTag::Spike]);
        assert!(live.memory().len() >= 1, "emit must write a residual scar");

        let at = live.memory().query_potential(&pos).unwrap_or(0.0);
        let cold = live.memory().query_potential(&far).unwrap_or(0.0);
        assert!(
            at > 1e-3,
            "later query at emit site must read the scar, pot={at}"
        );
        assert!(
            cold.abs() < at.abs() * 0.1,
            "far query must stay cold: far={cold} site={at}"
        );
        let later = live.memory().query_potential(&pos).unwrap_or(0.0);
        assert!(
            (later - at).abs() < 1e-4,
            "second query must still read the same scar"
        );

        let field2 = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory2 = SplatMemory::new(device.clone());
        let mut quiet = NiodooEngine::new(
            field2,
            memory2,
            Box::new(CpuBackend::new()),
            0.035,
            0.25,
            5.0,
        );
        let mut seen = Vec::new();
        let (stop2, applied2) = quiet
            .apply_emitted_control("hello with no tag", &mut seen, Some(&pos))
            .unwrap();
        assert!(!stop2);
        assert!(applied2.is_empty());
        let none = quiet.memory().query_potential(&pos).unwrap_or(0.0);
        assert!(
            none.abs() < 1e-6,
            "no-tag control must not write a scar, pot={none}"
        );
    }

    #[test]
    fn spike_still_moves_under_live_gemma_cap() {
        // config.toml force_cap = 0.2; pre-cap blend used to clip SPIKE into a no-op.
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 0.2);
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
        let idle = engine.steer(&baseline, &goal, 12).unwrap();
        engine.apply_request_effects(ControlTag::Spike);
        let spiked = engine.steer(&baseline, &goal, 12).unwrap();
        assert!((spiked.physics_blend - 6.5).abs() < 1e-6);
        assert!(
            (spiked.delta_h_norm - idle.delta_h_norm).abs() > 1e-6,
            "SPIKE must move Δh under cap=0.2; idle={} spike={}",
            idle.delta_h_norm,
            spiked.delta_h_norm
        );
        assert!(
            spiked.delta_h_norm > idle.delta_h_norm * 2.0,
            "post-cap blend 6.5 should enlarge the step; idle={} spike={}",
            idle.delta_h_norm,
            spiked.delta_h_norm
        );
    }

    #[test]
    fn warm_basin_splat_skips_early_ramp() {
        use crate::splat::Splat;

        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.0, 5.0);
        engine.set_field_wake(FieldWakeConfig {
            mode: FieldWakeMode::Off,
            k: 1,
            scale: 0.0,
            max_mag: 0.0,
            grad_blend: 0.0,
            dist_tau: 50.0,
        });
        engine.set_goal_force_limits(0.0, 0.0);
        engine.set_splat_force_limits(1.0, 0.0);
        engine.set_force_ramp(48, 0.03);

        // Offset ring so F_s ≠ 0 at the query site (on-center gradient is ~0).
        let scar_mu = Tensor::new(&[1.95f32, -2.5, 3.75, -4.0], &device).unwrap();
        engine.memory_mut().add_splat(Splat::new(scar_mu, 2.0, 1.0));
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();

        engine.set_memory_warm_pot(0.0);
        let ramped = engine.steer(&baseline, &goal, 0).unwrap();
        assert!(!ramped.memory_warm);
        assert!(ramped.scar_pot > 0.3, "on-center pot={}", ramped.scar_pot);

        engine.set_memory_warm_pot(0.3);
        let warm = engine.steer(&baseline, &goal, 0).unwrap();
        assert!(
            warm.memory_warm,
            "pot={} should trip warm gate",
            warm.scar_pot
        );
        assert!(
            warm.delta_h_norm > ramped.delta_h_norm * 2.0,
            "warm F_s must skip the 0.03 early ramp; ramped Δh={} warm Δh={}",
            ramped.delta_h_norm,
            warm.delta_h_norm
        );

        let far = Tensor::new(&[[10_000.0f32, 10_000.0, 10_000.0, 10_000.0]], &device).unwrap();
        let cold = engine.steer(&far, &goal, 0).unwrap();
        assert!(
            !cold.memory_warm,
            "far residual must stay cold, pot={}",
            cold.scar_pot
        );
    }

    #[test]
    fn topic_matched_far_bridge_skips_ramp() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.0, 5.0);
        engine.set_field_wake(FieldWakeConfig {
            mode: FieldWakeMode::Off,
            k: 1,
            scale: 0.0,
            max_mag: 0.0,
            grad_blend: 0.0,
            dist_tau: 50.0,
        });
        engine.set_goal_force_limits(0.0, 0.0);
        engine.set_splat_force_limits(1.0, 0.25);
        engine.set_force_ramp(48, 0.03);
        engine.set_memory_warm_pot(0.3);

        let scar_at = Tensor::new(&[1.25f32, -2.5, 3.75, -4.0], &device).unwrap();
        engine
            .memory_mut()
            .deposit_prefill_bridge(&scar_at, 90.0, 0.75, 0.005, 90.0, 0.35, 0xabcdu32)
            .unwrap();
        assert!(engine.memory().has_matching_bridge(0xabcd));
        assert!(!engine.memory().has_matching_bridge(0x1111));

        let far = Tensor::new(&[[200.0f32, 200.0, 200.0, 200.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
        engine.set_prompt_fp(0x1111);
        let miss = engine.steer(&far, &goal, 0).unwrap();
        assert!(
            !miss.memory_warm,
            "unrelated fp must stay cold at far L2, pot={}",
            miss.scar_pot
        );

        engine.set_prompt_fp(0xabcd);
        let hit = engine.steer(&far, &goal, 0).unwrap();
        assert!(
            hit.memory_warm,
            "topic-matched bridge must skip ramp even when pot is low, pot={}",
            hit.scar_pot
        );
        assert!(
            hit.delta_h_norm > miss.delta_h_norm * 2.0,
            "topic pull + un-ramped F_s must move farther; miss Δh={} hit Δh={}",
            miss.delta_h_norm,
            hit.delta_h_norm
        );
        assert!(
            hit.splat_mag <= 0.2501,
            "topic pull must respect final residual.splat_max, got {}",
            hit.splat_mag
        );
    }

    #[test]
    fn topic_mix_moves_steered_toward_matched_bridge() {
        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.0, 5.0);
        engine.set_field_wake(FieldWakeConfig {
            mode: FieldWakeMode::Off,
            k: 1,
            scale: 0.0,
            max_mag: 0.0,
            grad_blend: 0.0,
            dist_tau: 50.0,
        });
        engine.set_goal_force_limits(0.0, 0.0);
        engine.set_splat_force_limits(1.0, 0.0);
        engine.set_force_ramp(0, 1.0);
        engine.set_memory_warm_pot(0.3);

        let scar_at = Tensor::new(&[1.25f32, -2.5, 3.75, -4.0], &device).unwrap();
        engine
            .memory_mut()
            .deposit_prefill_bridge(&scar_at, 90.0, 0.75, 0.005, 90.0, 0.35, 0xabcdu32)
            .unwrap();
        let mu = engine
            .memory()
            .matched_bridge_mu(0xabcd)
            .unwrap()
            .expect("bridge mu");
        let mu_v: Vec<f32> = mu.flatten_all().unwrap().to_vec1().unwrap();

        let far = Tensor::new(&[[200.0f32, 200.0, 200.0, 200.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();

        engine.set_prompt_fp(0x1111);
        engine.set_topic_mix(0.0);
        let miss = engine.steer(&far, &goal, 0).unwrap();
        let miss_v: Vec<f32> = miss.steered.flatten_all().unwrap().to_vec1().unwrap();

        engine.set_prompt_fp(0xabcd);
        engine.set_topic_mix(0.35);
        let hit = engine.steer(&far, &goal, 0).unwrap();
        let hit_v: Vec<f32> = hit.steered.flatten_all().unwrap().to_vec1().unwrap();

        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na < 1e-8 || nb < 1e-8 {
                0.0
            } else {
                dot / (na * nb)
            }
        };
        let c_miss = cosine(&miss_v, &mu_v);
        let c_hit = cosine(&hit_v, &mu_v);
        assert!(
            c_hit > c_miss + 0.15,
            "topic_mix must rotate steered toward the matched scar; miss cos={c_miss} hit cos={c_hit}"
        );
        assert!(hit.memory_warm);
    }
}
