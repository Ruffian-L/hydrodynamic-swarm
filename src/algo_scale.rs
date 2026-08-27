//! Model-size → physics scaling, with the three historical transforms kept
//! distinct and receipted before they touch Hydro's residual seat.
//!
//! Port of `scripts/scale_physics_for_model.py` so the live HUD's "predicted"
//! column and the config generator can never drift apart. Constants and clamps
//! are copied verbatim from that script; see `docs/MODEL_SIZE_PHYSICS_SCALING.md`
//! for provenance.
//!
//! `algo_process` and `swarm_knobs` remain the exact legacy notebook contract.
//! New code should use `transform_prediction` and explicitly choose a
//! `SizeRule`; the current rule is √ to 8B and log-softened above 8B.

use serde::Serialize;

/// Golden reference point: 3B standard is where σ=0.15, θ=2.0 were validated.
const GOLDEN_PARAMS: f32 = 3.0;

/// Traversable stability zone — hard bounds on the Algo process params.
const SIGMA_BOUNDS: (f32, f32) = (0.04, 0.20);
const THETA_BOUNDS: (f32, f32) = (0.5, 3.0);
const BETA_BOUNDS: (f32, f32) = (40.0, 150.0);
const REPULSION_BOUNDS: (f32, f32) = (0.3, 3.0);

/// Model training objective. Determines fragility, which often matters more
/// than size: a thinking model's CoT scaffolding shatters at forces a base
/// model shrugs off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Standard,
    Instruct,
    Chat,
    Thinking,
    Coding,
}

impl ModelType {
    /// Force multiplier from the runs (`scale_physics_for_model.py:20`).
    pub fn multiplier(self) -> f32 {
        match self {
            ModelType::Standard => 1.0,
            ModelType::Instruct => 0.9,
            ModelType::Chat => 1.1,
            // CoT / house-of-cards — whisper force
            ModelType::Thinking => 0.4,
            // syntax wall — minimal jiggle
            ModelType::Coding => 0.27,
        }
    }

    /// July 2026 8B-anchored run-card archetype multipliers.
    pub fn july_multiplier(self) -> f32 {
        match self {
            ModelType::Standard => 1.0,
            ModelType::Instruct => 1.0,
            ModelType::Chat => 1.04,
            ModelType::Thinking => 0.88,
            ModelType::Coding => 0.82,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ModelType::Standard => "standard",
            ModelType::Instruct => "instruct",
            ModelType::Chat => "chat",
            ModelType::Thinking => "thinking",
            ModelType::Coding => "coding",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "standard" | "base" => Some(ModelType::Standard),
            "instruct" | "it" => Some(ModelType::Instruct),
            "chat" => Some(ModelType::Chat),
            "thinking" | "think" | "reasoning" => Some(ModelType::Thinking),
            "coding" | "code" | "coder" => Some(ModelType::Coding),
            _ => None,
        }
    }
}

/// The three source transforms plus the matched-panel auto-scale-off control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SizeRule {
    Legacy,
    EightBSqrt,
    Piecewise,
    Off,
}

impl SizeRule {
    pub const ALL: [Self; 4] = [Self::Legacy, Self::EightBSqrt, Self::Piecewise, Self::Off];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::EightBSqrt => "8b-sqrt",
            Self::Piecewise => "piecewise",
            Self::Off => "off",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "legacy" | "3b" | "3b-sqrt" => Some(Self::Legacy),
            "8b" | "8b-sqrt" | "sqrt-8b" => Some(Self::EightBSqrt),
            "piecewise" | "log-piecewise" | "log-soft" | "current" => Some(Self::Piecewise),
            "off" | "none" | "manual" => Some(Self::Off),
            _ => None,
        }
    }
}

/// Native output of one formula. This is not yet a Hydro seat configuration.
#[derive(Debug, Clone, Serialize)]
pub struct TransformPrediction {
    pub transform_id: SizeRule,
    pub params_b: f32,
    pub archetype: String,
    pub size_scale: f32,
    pub archetype_multiplier: f32,
    pub force_intensity: f32,
    pub sigma: f32,
    pub theta: f32,
    pub beta: f32,
    pub loop_repulsion: f32,
    pub predicted_temperature: f32,
    pub temperature_coupled: bool,
}

/// The complete coefficient vocabulary that actually configures Hydro.
/// Formula-native σ/θ/β are intentionally absent: those are recorded above.
#[derive(Debug, Clone, Serialize)]
pub struct SeatProfile {
    pub residual_cap: f32,
    pub residual_field: f32,
    pub residual_field_max: f32,
    pub residual_splat: f32,
    pub residual_splat_max: f32,
    pub residual_goal: f32,
    pub residual_goal_max: f32,
    pub force_ramp_tokens: usize,
    pub force_ramp_start: f32,
    pub temperature: f32,
    pub logit_field_alpha: f32,
    pub logit_splat_scale: f32,
    pub governor_brake: f32,
    pub governor_viscosity_gain: f32,
}

impl SeatProfile {
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        Self {
            residual_cap: cfg.physics.force_cap,
            residual_field: cfg.physics.field_wake_scale,
            residual_field_max: cfg.physics.field_wake_max,
            residual_splat: cfg.physics.splat_force_scale,
            residual_splat_max: cfg.physics.splat_force_max,
            residual_goal: cfg.physics.goal_force_scale,
            residual_goal_max: cfg.physics.goal_force_max,
            force_ramp_tokens: cfg.physics.force_ramp_tokens,
            force_ramp_start: cfg.physics.force_ramp_start,
            temperature: cfg.generation.temperature as f32,
            logit_field_alpha: cfg.logit_physics.field_alpha,
            logit_splat_scale: cfg.logit_physics.splat_scale,
            governor_brake: cfg.logit_physics.governor_brake,
            governor_viscosity_gain: cfg.logit_physics.governor_viscosity_gain,
        }
    }
}

/// One resolved startup input or override, retained in the immutable receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedValue {
    pub name: String,
    pub value: String,
    pub source: String,
    pub applied: bool,
}

/// Immutable file state at process start. An absent file is recorded rather than omitted so a
/// matched panel can prove that every arm began from the same empty store.
#[derive(Debug, Clone, Serialize)]
pub struct FileSnapshot {
    pub label: String,
    pub path: String,
    pub exists: bool,
    pub bytes: u64,
    pub sha256: String,
}

/// Decode and input conditions that must remain frozen across a matched panel.
#[derive(Debug, Clone, Serialize)]
pub struct RunContext {
    pub config: FileSnapshot,
    pub prompt_input: FileSnapshot,
    pub memory_inputs: Vec<FileSnapshot>,
    pub memory_clear_requested: bool,
    pub sample_seed: String,
    pub max_tokens: usize,
    pub official_pack_layout: bool,
    pub chat_template: String,
    pub control_tags_enabled: bool,
    pub tda_monitor_enabled: bool,
    pub tda_window_tokens: usize,
    pub tda_stride_tokens: usize,
    pub tda_cooldown_tokens: usize,
}

/// The transformer intervention is not a residual-seat coefficient, so receipt it separately.
#[derive(Debug, Clone, Serialize)]
pub struct HookProfile {
    pub enabled: bool,
    pub site: String,
    pub start_frac: f32,
    pub end_frac: f32,
    pub norm_fraction: f32,
    pub model_layers: usize,
    pub resolved_start_layer: usize,
    pub resolved_end_layer: usize,
    pub applications_per_decode: usize,
}

/// Equation → base profile → final Hydro seat, serialized before request 1.
#[derive(Debug, Clone, Serialize)]
pub struct ScalerReceipt {
    pub schema: String,
    pub receipt_id: String,
    pub created_unix_ms: u128,
    pub model_path: String,
    pub model_sha256: String,
    pub binary_path: String,
    pub binary_sha256: String,
    pub params_b: f32,
    pub archetype: String,
    pub transform_id: SizeRule,
    pub apply_to_residual_seat: bool,
    pub manual_gain: f32,
    pub effective_residual_gain: f32,
    pub selected_prediction: TransformPrediction,
    pub cross_check: Vec<TransformPrediction>,
    pub adapter_id: String,
    pub adapter_notes: Vec<String>,
    pub resolved_inputs_and_overrides: Vec<ResolvedValue>,
    pub run_context: RunContext,
    pub hook_profile: HookProfile,
    pub base_profile: SeatProfile,
    pub final_applied_coefficients: SeatProfile,
}

/// Current worktree size component: √ below/equal 8B, log-soft above 8B.
pub fn piecewise_size_scale(params_b: f32) -> f32 {
    let ratio = params_b.max(0.5) / 8.0;
    if ratio <= 1.0 {
        ratio.sqrt().max(0.35)
    } else {
        1.0 + 0.35 * ratio.ln()
    }
}

/// Compute one transform without adapting it to Hydro's residual vocabulary.
pub fn transform_prediction(
    params_b: f32,
    model_type: ModelType,
    rule: SizeRule,
    base_temperature: f32,
) -> TransformPrediction {
    let p = params_b.max(0.0);
    let base_t = base_temperature.clamp(0.25, 1.2);
    let (scale, type_mult, sigma, theta, beta, repulsion, predicted_t, coupled) = match rule {
        SizeRule::Legacy => {
            let scale = (p / 3.0).sqrt();
            let force = scale * model_type.multiplier();
            (
                scale,
                model_type.multiplier(),
                (0.15 * force).clamp(0.04, 0.20),
                (2.0 * force).clamp(0.5, 3.0),
                (100.0 * scale).clamp(40.0, 150.0),
                (2.0 * force).clamp(0.3, 3.0),
                if p <= 3.0 { 0.85 } else { 0.80 },
                true,
            )
        }
        SizeRule::EightBSqrt => {
            let scale = (p.max(0.5) / 8.0).sqrt();
            let type_mult = model_type.july_multiplier();
            let force = scale * type_mult;
            let beta = (100.0 * scale).clamp(70.0, 220.0);
            (
                scale,
                type_mult,
                (0.15 * force).clamp(0.04, 0.42),
                (0.55 * force).clamp(0.45, 1.8),
                beta,
                (0.60 * force).clamp(0.35, 1.9),
                (base_t * (100.0 / beta)).clamp(0.25, 1.2),
                true,
            )
        }
        SizeRule::Piecewise => {
            let scale = piecewise_size_scale(p);
            let type_mult = model_type.multiplier();
            let force = scale * type_mult;
            (
                scale,
                type_mult,
                (0.15 * force).clamp(0.04, 0.50),
                (0.55 * force).clamp(0.12, 2.20),
                (100.0 * scale).clamp(70.0, 220.0),
                (0.60 * force).clamp(0.15, 2.20),
                base_t,
                false,
            )
        }
        SizeRule::Off => (1.0, 1.0, 0.15, 0.55, 100.0, 0.60, base_t, false),
    };
    TransformPrediction {
        transform_id: rule,
        params_b: p,
        archetype: model_type.as_str().into(),
        size_scale: scale,
        archetype_multiplier: type_mult,
        force_intensity: scale * type_mult,
        sigma,
        theta,
        beta,
        loop_repulsion: repulsion,
        predicted_temperature: predicted_t,
        temperature_coupled: coupled,
    }
}

/// Adapt the selected formula onto the *existing* Hydro residual profile.
///
/// Only residual force scales and their ceilings move. Ramp, sampling,
/// logit-surface forces, and governor values remain frozen so a matched panel
/// changes one downstream force family. Historical temperature predictions are
/// still present in the receipt, but are not applied by this adapter.
pub fn apply_to_hydro_profile(
    cfg: &mut crate::config::Config,
    prediction: &TransformPrediction,
    manual_gain: f32,
    apply: bool,
) -> f32 {
    let effective = if apply {
        prediction.force_intensity * manual_gain
    } else {
        1.0
    };
    if apply {
        cfg.physics.force_cap *= effective;
        cfg.physics.field_wake_scale *= effective;
        cfg.physics.field_wake_max *= effective;
        cfg.physics.splat_force_scale *= effective;
        cfg.physics.splat_force_max *= effective;
        cfg.physics.goal_force_scale *= effective;
        cfg.physics.goal_force_max *= effective;
    }
    effective
}

/// Original Algo_WIP process intensities (σ/θ/β/repulsion).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgoProcess {
    /// The "jiggle" — noise/force intensity. Below 0.04 the model freezes
    /// (Buridan's Ass); above 0.20 it garbles (Fason singularity).
    pub sigma: f32,
    /// Drift correction / mean reversion.
    pub theta: f32,
    /// Inverse temperature. Scales with size only — type does not affect it.
    pub beta: f32,
    pub loop_repulsion: f32,
    /// √(params / 3B).
    pub scale: f32,
    /// scale × type_mult — the actual force intensity vs golden 3B standard.
    pub intensity: f32,
    pub type_mult: f32,
}

impl AlgoProcess {
    /// True when every process param sits inside the traversable stability zone.
    #[allow(dead_code)] // public API for callers checking a config before a run
    pub fn is_stable(&self) -> bool {
        in_bounds(self.sigma, SIGMA_BOUNDS)
            && in_bounds(self.theta, THETA_BOUNDS)
            && in_bounds(self.beta, BETA_BOUNDS)
    }
}

fn in_bounds(value: f32, bounds: (f32, f32)) -> bool {
    value >= bounds.0 && value <= bounds.1
}

fn clamp(value: f32, bounds: (f32, f32)) -> f32 {
    value.clamp(bounds.0, bounds.1)
}

/// Round to `places` decimals, matching Python's `round()` in the source script
/// closely enough that the two outputs compare equal when printed.
fn round_to(value: f32, places: i32) -> f32 {
    let factor = 10f32.powi(places);
    (value * factor).round() / factor
}

/// Algo_WIPjuly core: σ/θ/repulsion take the type multiplier, β takes scale only,
/// everything clamps to the traversable stability zone.
pub fn algo_process(params_b: f32, model_type: ModelType) -> AlgoProcess {
    let params_b = params_b.max(0.0);
    let scale = (params_b / GOLDEN_PARAMS).sqrt();
    let type_mult = model_type.multiplier();
    let intensity = scale * type_mult;

    AlgoProcess {
        sigma: round_to(clamp(0.15 * intensity, SIGMA_BOUNDS), 3),
        theta: round_to(clamp(2.0 * intensity, THETA_BOUNDS), 2),
        beta: round_to(clamp(100.0 * scale, BETA_BOUNDS), 1),
        loop_repulsion: round_to(clamp(2.0 * intensity, REPULSION_BOUNDS), 2),
        scale: round_to(scale, 3),
        intensity: round_to(intensity, 3),
        type_mult,
    }
}

/// Golden mid-zone swarm knobs ≈ 3B *standard*. Force intensity multiplies these.
struct KnobRef {
    reference: f32,
    floor: f32,
    ceiling: f32,
}

/// (reference @ 3B standard, soft floor, hard ceiling) per knob.
/// Ceilings come from learning-lane B on ~27B instruct — do not exceed on bigger
/// models. Floors keep tiny models out of the dead zone.
const FORCE_CAP: KnobRef = KnobRef {
    reference: 3.0,
    floor: 1.2,
    ceiling: 3.5,
};
const SPLAT_FORCE_MAX: KnobRef = KnobRef {
    reference: 28.0,
    floor: 10.0,
    ceiling: 28.0,
};
const FIELD_WAKE_MAX: KnobRef = KnobRef {
    reference: 25.0,
    floor: 8.0,
    ceiling: 25.0,
};
const GOAL_FORCE_MAX: KnobRef = KnobRef {
    reference: 40.0,
    floor: 15.0,
    ceiling: 40.0,
};
const GOAL_FORCE_SCALE: KnobRef = KnobRef {
    reference: 0.12,
    floor: 0.05,
    ceiling: 0.15,
};
const FIELD_WAKE_SCALE: KnobRef = KnobRef {
    reference: 0.18,
    floor: 0.08,
    ceiling: 0.22,
};

fn scaled(knob: &KnobRef, intensity: f32) -> f32 {
    round_to(
        (knob.reference * intensity).clamp(knob.floor, knob.ceiling),
        3,
    )
}

/// The swarm `config.toml` knobs the √-law actually moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwarmKnobs {
    pub force_cap: f32,
    pub splat_force_max: f32,
    pub field_wake_max: f32,
    pub goal_force_max: f32,
    pub goal_force_scale: f32,
    pub field_wake_scale: f32,
    pub force_ramp_tokens: usize,
    pub force_ramp_start: f32,
    pub temperature: f32,
}

/// Map Algo force intensity onto swarm config fields, anchored on 3B standard.
///
/// Splat *geometry* (`splat_sigma`, `min_splat_dist`) is deliberately absent:
/// scar width follows the field/hidden dim, not the force √-law. See the source
/// script's comment at `:140`.
pub fn swarm_knobs(params_b: f32, model_type: ModelType) -> SwarmKnobs {
    let process = algo_process(params_b, model_type);
    let intensity = process.intensity;

    // Ramp: smaller → longer/gentler start (respect J-space prefill).
    let (force_ramp_tokens, force_ramp_start) = if params_b <= 2.0 {
        (18, 0.10)
    } else if params_b <= 5.0 {
        (15, 0.15)
    } else if params_b <= 12.0 {
        (12, 0.18)
    } else {
        (12, 0.20)
    };

    SwarmKnobs {
        force_cap: scaled(&FORCE_CAP, intensity),
        splat_force_max: scaled(&SPLAT_FORCE_MAX, intensity),
        field_wake_max: scaled(&FIELD_WAKE_MAX, intensity),
        goal_force_max: scaled(&GOAL_FORCE_MAX, intensity),
        goal_force_scale: scaled(&GOAL_FORCE_SCALE, intensity),
        field_wake_scale: scaled(&FIELD_WAKE_SCALE, intensity),
        force_ramp_tokens,
        force_ramp_start,
        temperature: if params_b <= 3.0 { 0.85 } else { 0.80 },
    }
}

/// Where a live knob sits relative to what the √-law predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    /// Well under prediction — little or no steering reaches the model.
    Cold,
    In,
    /// Well over prediction — the gibberish path on smaller models.
    Hot,
}

impl Zone {
    pub fn as_str(self) -> &'static str {
        match self {
            Zone::Cold => "COLD",
            Zone::In => "IN",
            Zone::Hot => "HOT",
        }
    }
}

/// Classify a live knob against its predicted value. Tolerance is deliberately
/// wide (0.6×–1.6×): the √-law gives a starting point, not a target.
pub fn zone_ratio(live: f32, predicted: f32) -> Zone {
    if predicted <= 1e-6 {
        return Zone::In;
    }
    let ratio = live / predicted;
    if ratio < 0.6 {
        Zone::Cold
    } else if ratio > 1.6 {
        Zone::Hot
    } else {
        Zone::In
    }
}

/// Classify the process params themselves against the traversable stability zone.
pub fn zone_algo(process: &AlgoProcess) -> Zone {
    if process.sigma < SIGMA_BOUNDS.0 || process.theta < THETA_BOUNDS.0 {
        Zone::Cold
    } else if process.sigma > SIGMA_BOUNDS.1 || process.theta > THETA_BOUNDS.1 {
        Zone::Hot
    } else {
        Zone::In
    }
}

/// Guess model size and type from a weights path, so the HUD works without flags.
///
/// Size comes from the first `<number>b` token (`gemma-3-4b-it-Q6_K.gguf` → 4.0);
/// type from the usual filename markers. Returns `None` for size when nothing
/// plausible is found — better a missing panel than a wrong prediction.
pub fn infer_from_path(path: &str) -> (Option<f32>, ModelType) {
    let lower = path.to_lowercase();

    let model_type = if lower.contains("coder") || lower.contains("-code") {
        ModelType::Coding
    } else if lower.contains("think") || lower.contains("reason") {
        ModelType::Thinking
    } else if lower.contains("instruct") || lower.contains("-it") || lower.contains("_it") {
        ModelType::Instruct
    } else if lower.contains("chat") {
        ModelType::Chat
    } else {
        ModelType::Standard
    };

    (parse_params_b(&lower), model_type)
}

/// First `<number>b` token in a lowercased path, e.g. "4b", "27b", "1.5b".
///
/// Requiring the trailing standalone `b` is what keeps quant tags and
/// architecture digits out: `q6_k` and `gemma3` never match, `e4b` does.
fn parse_params_b(lower: &str) -> Option<f32> {
    let bytes = lower.as_bytes();
    let mut index = 0usize;

    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        let number = &lower[start..index];
        // Must be immediately followed by a standalone 'b'.
        let next_is_b = bytes.get(index) == Some(&b'b');
        let b_ends_token = bytes
            .get(index + 1)
            .is_none_or(|c| !c.is_ascii_alphanumeric());
        if next_is_b && b_ends_token {
            if let Ok(value) = number.trim_end_matches('.').parse::<f32>() {
                if (0.1..=2000.0).contains(&value) {
                    return Some(value);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published table in Algo_WIPjuly.md — this is the contract.
    #[test]
    fn reproduces_published_algo_table() {
        let cases = [
            // (params, type, sigma, theta, beta)
            (1.0, ModelType::Standard, 0.087, 1.15, 57.7),
            (3.0, ModelType::Standard, 0.150, 2.00, 100.0),
            (3.0, ModelType::Instruct, 0.135, 1.80, 100.0),
            (4.0, ModelType::Thinking, 0.069, 0.92, 115.5),
            (7.0, ModelType::Coding, 0.062, 0.82, 150.0),
            (70.0, ModelType::Standard, 0.200, 3.00, 150.0),
        ];
        for (params, model_type, sigma, theta, beta) in cases {
            let got = algo_process(params, model_type);
            assert!(
                (got.sigma - sigma).abs() < 1e-3,
                "{params}B {}: sigma {} != {sigma}",
                model_type.as_str(),
                got.sigma
            );
            assert!(
                (got.theta - theta).abs() < 1e-2,
                "{params}B theta {}",
                got.theta
            );
            assert!(
                (got.beta - beta).abs() < 1e-1,
                "{params}B beta {}",
                got.beta
            );
            assert!(got.is_stable(), "{params}B should be in the stability zone");
        }
    }

    /// Qwen35 27B instruct saturates every clamp (Algo_WIPjuly.md:468).
    #[test]
    fn large_instruct_saturates_the_caps() {
        let got = algo_process(27.0, ModelType::Instruct);
        assert!((got.scale - 3.0).abs() < 1e-3);
        assert!((got.sigma - 0.20).abs() < 1e-6);
        assert!((got.theta - 3.00).abs() < 1e-6);
        assert!((got.beta - 150.0).abs() < 1e-6);
        assert!((got.loop_repulsion - 3.00).abs() < 1e-6);
    }

    #[test]
    fn golden_3b_standard_has_unit_intensity() {
        let got = algo_process(3.0, ModelType::Standard);
        assert!((got.scale - 1.0).abs() < 1e-6);
        assert!((got.intensity - 1.0).abs() < 1e-6);

        // At intensity 1.0 the swarm knobs equal the 3B reference exactly.
        let knobs = swarm_knobs(3.0, ModelType::Standard);
        assert!((knobs.force_cap - 3.0).abs() < 1e-3);
        assert!((knobs.goal_force_scale - 0.12).abs() < 1e-3);
        assert!((knobs.field_wake_scale - 0.18).abs() < 1e-3);
    }

    #[test]
    fn small_models_get_floored_not_zeroed() {
        // 1B thinking: intensity 0.231 — every knob would underflow without floors.
        let knobs = swarm_knobs(1.0, ModelType::Thinking);
        assert!((knobs.force_cap - 1.2).abs() < 1e-3, "floored to FLOOR");
        assert_eq!(knobs.force_ramp_tokens, 18);
        assert!((knobs.force_ramp_start - 0.10).abs() < 1e-6);
    }

    /// Parity with `scripts/scale_physics_for_model.py`. Values captured from
    /// `--algo-only` and `--toml` runs; if this drifts, the HUD's predicted
    /// column and the config generator disagree and one of them is lying.
    #[test]
    fn matches_the_python_generator() {
        // (params, type, sigma, theta, beta, force_cap, goal_force_scale, ramp_tokens, ramp_start)
        let cases = [
            (
                1.0,
                ModelType::Standard,
                0.087,
                1.15,
                57.7,
                1.731,
                0.069,
                18,
                0.10,
            ),
            (
                3.0,
                ModelType::Standard,
                0.150,
                2.00,
                100.0,
                3.0,
                0.12,
                15,
                0.15,
            ),
            (
                4.0,
                ModelType::Instruct,
                0.156,
                2.08,
                115.5,
                3.117,
                0.125,
                15,
                0.15,
            ),
            (
                4.0,
                ModelType::Thinking,
                0.069,
                0.92,
                115.5,
                1.386,
                0.055,
                15,
                0.15,
            ),
            (
                7.0,
                ModelType::Coding,
                0.062,
                0.82,
                150.0,
                1.236,
                0.05,
                12,
                0.18,
            ),
            (
                27.0,
                ModelType::Instruct,
                0.200,
                3.00,
                150.0,
                3.5,
                0.15,
                12,
                0.20,
            ),
            (
                70.0,
                ModelType::Standard,
                0.200,
                3.00,
                150.0,
                3.5,
                0.15,
                12,
                0.20,
            ),
        ];
        for (params, ty, sigma, theta, beta, cap, goal, ramp_tokens, ramp_start) in cases {
            let p = algo_process(params, ty);
            let k = swarm_knobs(params, ty);
            let label = format!("{params}B {}", ty.as_str());
            assert!((p.sigma - sigma).abs() < 1e-3, "{label} sigma {}", p.sigma);
            assert!((p.theta - theta).abs() < 1e-2, "{label} theta {}", p.theta);
            assert!((p.beta - beta).abs() < 1e-1, "{label} beta {}", p.beta);
            assert!(
                (k.force_cap - cap).abs() < 1e-3,
                "{label} cap {}",
                k.force_cap
            );
            assert!(
                (k.goal_force_scale - goal).abs() < 1e-3,
                "{label} goal {}",
                k.goal_force_scale
            );
            assert_eq!(k.force_ramp_tokens, ramp_tokens, "{label} ramp tokens");
            assert!(
                (k.force_ramp_start - ramp_start).abs() < 1e-6,
                "{label} ramp start"
            );
        }
    }

    #[test]
    fn zone_ratio_brackets_the_prediction() {
        assert_eq!(zone_ratio(3.0, 3.0), Zone::In);
        assert_eq!(zone_ratio(1.0, 3.0), Zone::Cold);
        assert_eq!(zone_ratio(20.0, 3.0), Zone::Hot);
        // Just inside the 0.6×–1.6× band. The band edges themselves are float-
        // fuzzy and not worth pinning — this is a coarse advisory, not a gate.
        assert_eq!(zone_ratio(1.9, 3.0), Zone::In);
        assert_eq!(zone_ratio(4.7, 3.0), Zone::In);
        assert_eq!(zone_ratio(1.7, 3.0), Zone::Cold);
        assert_eq!(zone_ratio(4.9, 3.0), Zone::Hot);
        // A zero prediction cannot be judged.
        assert_eq!(zone_ratio(5.0, 0.0), Zone::In);
    }

    #[test]
    fn zone_algo_flags_the_clamped_extremes() {
        assert_eq!(zone_algo(&algo_process(3.0, ModelType::Standard)), Zone::In);
        // Clamps keep even 70B inside the zone — that is the point of the law.
        assert_eq!(
            zone_algo(&algo_process(70.0, ModelType::Standard)),
            Zone::In
        );
    }

    #[test]
    fn twelve_b_cross_check_keeps_the_three_transforms_distinct() {
        let legacy = transform_prediction(12.0, ModelType::Instruct, SizeRule::Legacy, 0.7);
        assert!((legacy.size_scale - 2.0).abs() < 1e-6);
        assert!((legacy.force_intensity - 1.8).abs() < 1e-6);
        assert!((legacy.sigma - 0.20).abs() < 1e-6);
        assert!((legacy.theta - 3.0).abs() < 1e-6);
        assert!((legacy.beta - 150.0).abs() < 1e-6);

        let july = transform_prediction(12.0, ModelType::Instruct, SizeRule::EightBSqrt, 0.7);
        assert!((july.size_scale - (1.5f32).sqrt()).abs() < 1e-6);
        assert!((july.archetype_multiplier - 1.0).abs() < 1e-6);
        assert!(july.temperature_coupled);
        assert!((july.predicted_temperature - 0.571_547_6).abs() < 1e-5);

        let current = transform_prediction(12.0, ModelType::Instruct, SizeRule::Piecewise, 0.7);
        let expected_size = 1.0 + 0.35 * (1.5f32).ln();
        assert!((current.size_scale - expected_size).abs() < 1e-6);
        assert!((current.force_intensity - expected_size * 0.9).abs() < 1e-6);
        assert!(!current.temperature_coupled);
        assert!((current.predicted_temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn hydro_adapter_moves_only_the_residual_force_family() {
        let mut cfg = crate::config::Config::default();
        cfg.physics.force_cap = 1.0;
        cfg.physics.field_wake_scale = 0.02;
        cfg.physics.splat_force_scale = 0.03;
        cfg.physics.goal_force_scale = 0.008;
        cfg.physics.field_wake_max = 2.0;
        cfg.physics.splat_force_max = 4.0;
        cfg.physics.goal_force_max = 2.0;
        cfg.physics.force_ramp_tokens = 48;
        cfg.physics.force_ramp_start = 0.03;
        cfg.generation.temperature = 0.7;
        cfg.logit_physics.field_alpha = 0.02;
        cfg.logit_physics.splat_scale = 0.004;
        cfg.logit_physics.governor_brake = 1.5;
        cfg.logit_physics.governor_viscosity_gain = 3.0;
        let base = SeatProfile::from_config(&cfg);
        let prediction = transform_prediction(12.0, ModelType::Instruct, SizeRule::Piecewise, 0.7);
        let gain = apply_to_hydro_profile(&mut cfg, &prediction, 1.0, true);
        let final_profile = SeatProfile::from_config(&cfg);

        assert!((gain - prediction.force_intensity).abs() < 1e-6);
        assert!((final_profile.residual_cap - base.residual_cap * gain).abs() < 1e-6);
        assert!((final_profile.residual_field - base.residual_field * gain).abs() < 1e-6);
        assert!((final_profile.residual_splat - base.residual_splat * gain).abs() < 1e-6);
        assert!((final_profile.residual_goal - base.residual_goal * gain).abs() < 1e-6);
        assert_eq!(final_profile.force_ramp_tokens, base.force_ramp_tokens);
        assert_eq!(final_profile.force_ramp_start, base.force_ramp_start);
        assert_eq!(final_profile.temperature, base.temperature);
        assert_eq!(final_profile.logit_field_alpha, base.logit_field_alpha);
        assert_eq!(final_profile.logit_splat_scale, base.logit_splat_scale);
        assert_eq!(final_profile.governor_brake, base.governor_brake);
        assert_eq!(
            final_profile.governor_viscosity_gain,
            base.governor_viscosity_gain
        );
    }

    #[test]
    fn infers_size_and_type_from_weights_path() {
        let cases = [
            (
                "/models/gemma-3-4b-it-Q6_K.gguf",
                Some(4.0),
                ModelType::Instruct,
            ),
            (
                "/models/Llama-3.2-3B-Instruct.gguf",
                Some(3.0),
                ModelType::Instruct,
            ),
            (
                "/models/Qwen3.6-27B-Uncensored-Q6_K.gguf",
                Some(27.0),
                ModelType::Standard,
            ),
            ("/models/DASD-4B-Think.gguf", Some(4.0), ModelType::Thinking),
            (
                "/models/deepseek-coder-7b.gguf",
                Some(7.0),
                ModelType::Coding,
            ),
            ("/models/Qwen2.5-1.5B-chat.gguf", Some(1.5), ModelType::Chat),
        ];
        for (path, params, model_type) in cases {
            let (got_params, got_type) = infer_from_path(path);
            assert_eq!(got_params, params, "params from {path}");
            assert_eq!(got_type, model_type, "type from {path}");
        }
    }

    #[test]
    fn refuses_to_guess_when_the_path_has_no_size() {
        // Quant tags and arch digits must not be read as a parameter count.
        assert_eq!(parse_params_b("/models/mystery-model-q6_k.gguf"), None);
        assert_eq!(parse_params_b("/models/gemma3n-e4b.gguf"), Some(4.0));
        assert_eq!(parse_params_b("/models/tokenizer.json"), None);
    }
}
