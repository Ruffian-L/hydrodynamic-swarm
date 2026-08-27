//! Live scalar HUD — a sticky footer under the streaming text.
//!
//! Shows what the √-law scaling algo actually moves (live knobs vs. what
//! `algo_scale` predicts for this model, with a stability-zone verdict) and a
//! dense residual/hidden-state block, repainted as tokens arrive.
//!
//! Deliberately *not* an alternate-screen TUI: text still streams to stdout the
//! way it always did, so piping, `tee`, and the live stream file are unaffected,
//! and Ctrl-C still works (no raw mode). `src/tui.rs` remains the slider editor.
//!
//! The footer is pinned with a DECSTBM scroll region: rows above it scroll
//! normally, the last few rows are reserved and only this module writes there.
//! That is what makes it survive the generation loop's own diagnostics and the
//! endocrine worker thread, which both `println!` straight to stdout with no
//! idea the HUD exists.
//!
//! The HUD auto-disables when stdout is not a terminal, so redirected runs and
//! ablation scripts produce byte-identical output to before.

use crate::algo_scale::{
    self, AlgoProcess, ModelType, SeatProfile, SizeRule, SwarmKnobs, TransformPrediction,
};
use candle_core::Tensor;
use crossterm::cursor::{MoveTo, RestorePosition, SavePosition};
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};
use std::io::{self, IsTerminal, Stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};

/// Set while a live HUD owns the screen.
///
/// Global because the per-token chatter it silences is printed from places that
/// have no HUD handle and never will: `NiodooEngine` deep in the physics, and
/// the endocrine enzyme worker on its own thread. Those numbers are all on the
/// footer anyway — and they still reach `logs/live.txt` either way.
static QUIET: AtomicBool = AtomicBool::new(false);

/// True when per-token diagnostics should stay off the screen.
pub fn quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Silence per-token diagnostics. The full-screen REPL holds this for its whole
/// session — a stray `println!` there lands on top of the layout.
pub fn set_quiet(value: bool) {
    QUIET.store(value, Ordering::Relaxed);
}

/// Operator diagnostic. Never the mouth. When the HUD is up it already
/// shows this; when it is not, send stderr so chat stays tags/monitor/memory.
macro_rules! hud_quiet_println {
    ($($arg:tt)*) => {
        if !$crate::hud::quiet() {
            eprintln!($($arg)*);
        }
    };
}
pub(crate) use hud_quiet_println;

/// Number of body lines between the two rules. Keep in sync with `render_block`.
const BODY_LINES: usize = 9;
/// Body + the two horizontal rules.
const BLOCK_LINES: usize = BODY_LINES + 2;

const MIN_WIDTH: u16 = 60;
const DEFAULT_WIDTH: u16 = 100;

/// Model identity plus its predicted knobs — the static half of the ALGO panel.
#[derive(Debug, Clone)]
pub struct AlgoView {
    pub params_b: f32,
    pub model_type: ModelType,
    pub rule: SizeRule,
    pub process: AlgoProcess,
    pub predicted: SwarmKnobs,
}

impl AlgoView {
    pub fn new(params_b: f32, model_type: ModelType) -> Self {
        Self {
            params_b,
            model_type,
            rule: SizeRule::Legacy,
            process: algo_scale::algo_process(params_b, model_type),
            predicted: algo_scale::swarm_knobs(params_b, model_type),
        }
    }

    pub fn from_applied(
        params_b: f32,
        model_type: ModelType,
        prediction: &TransformPrediction,
        applied: &SeatProfile,
    ) -> Self {
        Self {
            params_b,
            model_type,
            rule: prediction.transform_id,
            process: AlgoProcess {
                sigma: prediction.sigma,
                theta: prediction.theta,
                beta: prediction.beta,
                loop_repulsion: prediction.loop_repulsion,
                scale: prediction.size_scale,
                intensity: prediction.force_intensity,
                type_mult: prediction.archetype_multiplier,
            },
            predicted: SwarmKnobs {
                force_cap: applied.residual_cap,
                splat_force_max: applied.residual_splat_max,
                field_wake_max: applied.residual_field_max,
                goal_force_max: applied.residual_goal_max,
                goal_force_scale: applied.residual_goal,
                field_wake_scale: applied.residual_field,
                force_ramp_tokens: applied.force_ramp_tokens,
                force_ramp_start: applied.force_ramp_start,
                temperature: applied.temperature,
            },
        }
    }
}

/// One token's worth of scalars. `Option` fields are rendered `—` when the
/// calling path does not compute them (the chat loop has no quality scoring).
#[derive(Debug, Clone, Default)]
pub struct HudFrame {
    pub step: usize,
    pub max_tokens: usize,

    // Live knobs the algo moves.
    pub force_cap: f32,
    pub goal_force_scale: f32,
    pub temperature: f32,
    pub force_ramp_start: f32,
    pub force_ramp_tokens: usize,
    pub field_grad_blend: f32,

    // Residual / hidden state.
    pub baseline_norm: f32,
    pub steered_norm: f32,
    pub pullback: f32,
    pub delta_h_norm: f32,
    pub clip_frac: f32,
    pub ramp: f32,
    pub eureka_boost: f32,
    pub cos_drift: Option<f32>,

    // Forces.
    pub grad_mag: f32,
    pub splat_mag: f32,
    pub goal_mag: f32,
    pub ocean_mag: f32,
    pub memory_ranked: bool,
    /// Per-force ceilings, so a railed force can be marked. A force sitting on
    /// its ceiling every step is a constant shove, not steering.
    pub field_wake_max: f32,
    pub splat_force_max: f32,
    pub goal_force_max: f32,

    // Logit surface / hook.
    pub logit_delta: Option<f32>,
    pub logit_velocity: f32,
    pub logit_viscosity: f32,
    pub hook_delta_mean: Option<f32>,
    pub hook_applications: Option<usize>,

    // Sampling quality.
    pub p_chosen: Option<f32>,
    pub entropy: Option<f32>,
    pub margin: Option<f32>,
    pub scars: usize,
}

pub struct Hud {
    enabled: bool,
    every: usize,
    algo: Option<AlgoView>,
    /// Set once the footer is on screen.
    painted: bool,
    /// Terminal size the current scroll region was cut for. A resize invalidates
    /// it and the region is re-cut.
    region: Option<(u16, u16)>,
    prev_hidden: Option<Tensor>,
    last: Option<HudFrame>,
    stdout: Stdout,
}

impl Hud {
    /// `enabled` is the operator's intent; a non-TTY stdout overrides it to off.
    pub fn new(enabled: bool, every: usize, algo: Option<AlgoView>) -> Self {
        let stdout = io::stdout();
        let enabled = enabled && stdout.is_terminal();
        QUIET.store(enabled, Ordering::Relaxed);
        Self {
            enabled,
            every: every.max(1),
            algo,
            painted: false,
            region: None,
            prev_hidden: None,
            last: None,
            stdout,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Arm the HUD for a generation run. Chat calls this once per turn, since
    /// `finish` un-quiets so the between-turn output prints in full.
    pub fn begin(&mut self) {
        QUIET.store(self.enabled, Ordering::Relaxed);
        self.prev_hidden = None;
    }

    /// Print generated text. The scroll region keeps it clear of the footer, so
    /// this is a plain write — no cursor gymnastics per token.
    pub fn stream(&mut self, text: &str) -> io::Result<()> {
        print!("{text}");
        io::stdout().flush()
    }

    /// Refresh the footer with a new frame. Honours `--hud-every`.
    pub fn update(&mut self, frame: HudFrame) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let due = frame.step % self.every == 0;
        self.last = Some(frame);
        if due {
            self.paint()?;
        }
        Ok(())
    }

    /// Cosine between this hidden state and the previous step's.
    ///
    /// Only computed while the HUD is on — it costs one extra GPU sync per
    /// token, which force-off ablation runs should not have to pay.
    pub fn cos_drift(&mut self, hidden: &Tensor) -> Option<f32> {
        if !self.enabled {
            return None;
        }
        let current = hidden.flatten_all().ok()?;
        let cos = match self.prev_hidden.as_ref() {
            Some(prev) if prev.dims() == current.dims() => cosine(prev, &current),
            _ => None,
        };
        self.prev_hidden = Some(current);
        cos
    }

    /// Release the reserved rows and leave the last frame in the scrollback.
    pub fn finish(&mut self) -> io::Result<()> {
        // Generation is over: the post-run phases should print in full again.
        QUIET.store(false, Ordering::Relaxed);
        if !self.enabled || !self.painted {
            return Ok(());
        }
        let lines = self.block();
        let (_, rows) = terminal_size();
        let first = rows.saturating_sub(lines.len() as u16);
        // Drop back to a full-height scroll region first: DECSTBM homes the
        // cursor, so positioning before it would clear the whole screen — the
        // generated text included — instead of just the footer.
        self.release_region()?;
        execute!(
            self.stdout,
            MoveTo(0, first),
            Clear(ClearType::FromCursorDown)
        )?;
        // Rewrite the block as ordinary output so it scrolls with everything else.
        for line in &lines {
            println!("{line}");
        }
        self.painted = false;
        self.stdout.flush()
    }

    /// Reserve the bottom rows by shrinking the scroll region to everything
    /// above them (DECSTBM). Normal output — ours, the generation loop's
    /// diagnostics, the endocrine thread's — can then only scroll within the
    /// top region and can never overwrite the footer.
    fn reserve_region(&mut self, block_lines: u16) -> io::Result<()> {
        let (cols, rows) = terminal_size();
        // Need room for the footer plus a usable text area.
        if rows <= block_lines + 2 {
            self.enabled = false;
            return Ok(());
        }
        if self.region == Some((cols, rows)) {
            return Ok(());
        }
        // Scroll existing output up so nothing is buried under the new footer.
        queue!(self.stdout, Print("\n".repeat(block_lines as usize)))?;
        self.stdout.flush()?;
        let bottom = rows - block_lines;
        // `block_lines` newlines only land the cursor on `bottom - 1` once the
        // screen is already full (steady-state scrolling). Early in a session
        // — or on a short terminal — the screen has not filled yet, so the
        // cursor is still wherever the last real line left it. Read that back
        // instead of assuming a fixed row: assuming wrong put the cursor back
        // on the line still streaming and the footer started overwriting the
        // prompt and first word of the reply from column 0.
        let after_pad_row = crossterm::cursor::position()
            .map(|(_, row)| row)
            .unwrap_or(bottom.saturating_sub(1));
        // DECSTBM rows are 1-based and inclusive; it also homes the cursor, so
        // park it explicitly afterwards — on the real last-used row, capped to
        // the region just carved out.
        write!(self.stdout, "\x1b[1;{bottom}r")?;
        let target_row = after_pad_row.min(bottom.saturating_sub(1));
        execute!(self.stdout, MoveTo(0, target_row))?;
        self.region = Some((cols, rows));
        Ok(())
    }

    fn release_region(&mut self) -> io::Result<()> {
        if self.region.take().is_some() {
            write!(self.stdout, "\x1b[r")?;
            self.stdout.flush()?;
        }
        Ok(())
    }

    fn paint(&mut self) -> io::Result<()> {
        let lines = self.block();
        if lines.is_empty() {
            return Ok(());
        }
        self.reserve_region(lines.len() as u16)?;
        if !self.enabled {
            return Ok(());
        }
        let (_, rows) = terminal_size();
        let first = rows - lines.len() as u16;
        // Absolute addressing into the reserved rows. Safe to save/restore the
        // cursor here because nothing in this block scrolls the region.
        queue!(self.stdout, SavePosition)?;
        for (index, line) in lines.iter().enumerate() {
            queue!(
                self.stdout,
                MoveTo(0, first + index as u16),
                Clear(ClearType::CurrentLine),
                SetForegroundColor(Color::DarkGrey),
                Print(line),
                ResetColor
            )?;
        }
        queue!(self.stdout, RestorePosition)?;
        self.painted = true;
        self.stdout.flush()
    }

    fn block(&self) -> Vec<String> {
        match self.last.as_ref() {
            Some(frame) => render_block(frame, self.algo.as_ref(), terminal_width()),
            None => Vec::new(),
        }
    }
}

/// A shrunken scroll region outlives the process if we do not put it back, and
/// the user is left with a terminal that will not scroll to the bottom.
impl Drop for Hud {
    fn drop(&mut self) {
        let _ = self.release_region();
        QUIET.store(false, Ordering::Relaxed);
    }
}

fn terminal_size() -> (u16, u16) {
    terminal::size().unwrap_or((DEFAULT_WIDTH, 40))
}

fn terminal_width() -> u16 {
    terminal_size().0
}

/// Cosine between the previous step's hidden state and this one, for callers
/// that keep their own `prev` (the REPL's turn loop). `None` on the first step
/// or a shape change — never a fabricated 1.0.
pub fn cosine_drift(prev: Option<&Tensor>, current: &Tensor) -> Option<f32> {
    let current = current.flatten_all().ok()?;
    let prev = prev?;
    if prev.dims() != current.dims() {
        return None;
    }
    cosine(prev, &current)
}

fn cosine(a: &Tensor, b: &Tensor) -> Option<f32> {
    let dot = (a * b).ok()?.sum_all().ok()?.to_scalar::<f32>().ok()?;
    let norm_a = a
        .sqr()
        .ok()?
        .sum_all()
        .ok()?
        .to_scalar::<f32>()
        .ok()?
        .sqrt();
    let norm_b = b
        .sqr()
        .ok()?
        .sum_all()
        .ok()?
        .to_scalar::<f32>()
        .ok()?
        .sqrt();
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return None;
    }
    Some(dot / (norm_a * norm_b))
}

/// Render the footer. Pure and terminal-free so it can be unit tested.
pub fn render_block(frame: &HudFrame, algo: Option<&AlgoView>, width: u16) -> Vec<String> {
    let width = width.max(MIN_WIDTH) as usize;
    let rule = "─".repeat(width.min(120));
    let mut lines = Vec::with_capacity(BLOCK_LINES);
    lines.push(rule.clone());

    match algo {
        Some(view) => {
            lines.push(format!(
                " ALGO  {:.0}B {} · {} scale={:.3} · mult {:.2} · int {:.3}",
                view.params_b,
                view.model_type.as_str(),
                view.rule.as_str(),
                view.process.scale,
                view.process.type_mult,
                view.process.intensity
            ));
            lines.push(format!(
                "       {:<12}{:>8}{:>8}  {:<5}   {:<10}{:>8}{:>8}  {}",
                "σ→cap",
                fmt(frame.force_cap, 2),
                fmt(view.predicted.force_cap, 2),
                algo_scale::zone_ratio(frame.force_cap, view.predicted.force_cap).as_str(),
                "θ→goal",
                fmt(frame.goal_force_scale, 3),
                fmt(view.predicted.goal_force_scale, 3),
                algo_scale::zone_ratio(frame.goal_force_scale, view.predicted.goal_force_scale)
                    .as_str(),
            ));
            lines.push(format!(
                "       {:<12}{:>8}{:>8}  {:<5}   {:<10}{:>8}{:>8}",
                "β→temp",
                fmt(frame.temperature, 2),
                fmt(view.predicted.temperature, 2),
                algo_scale::zone_ratio(frame.temperature, view.predicted.temperature).as_str(),
                "ramp",
                format!("{:.2}/{}", frame.force_ramp_start, frame.force_ramp_tokens),
                format!(
                    "{:.2}/{}",
                    view.predicted.force_ramp_start, view.predicted.force_ramp_tokens
                ),
            ));
        }
        None => {
            lines.push(format!(
                " ALGO  no model size given — pass --model-params <B> --model-type <t>"
            ));
            lines.push(format!(
                "       {:<12}{:>8}   {:<10}{:>8}   {:<8}{:>8}",
                "σ→cap",
                fmt(frame.force_cap, 2),
                "θ→goal",
                fmt(frame.goal_force_scale, 3),
                "β→temp",
                fmt(frame.temperature, 2),
            ));
            lines.push(format!(
                "       {:<12}{:>8}   {:<10}{:>8}",
                "ramp",
                format!("{:.2}/{}", frame.force_ramp_start, frame.force_ramp_tokens),
                "blend",
                fmt(frame.field_grad_blend, 3),
            ));
        }
    }

    lines.push(format!(
        " RESID ‖h‖ {:.1} → {:.1}   pull ×{:.4}   ‖δh‖ {:.2}   clip {:.1}%",
        frame.baseline_norm,
        frame.steered_norm,
        frame.pullback,
        frame.delta_h_norm,
        frame.clip_frac * 100.0
    ));
    lines.push(format!(
        "       cos(h_t,h_t-1) {}   ramp ×{:.2}   eureka ×{:.2}   blend {}",
        opt(frame.cos_drift, 4),
        frame.ramp,
        frame.eureka_boost,
        fmt(frame.field_grad_blend, 3)
    ));
    lines.push(format!(
        " FORCE F_g {:.2}{}  F_s {:.2}{}{}  F_a {:.2}{}  F_o {:.2}",
        frame.grad_mag,
        railed(frame.grad_mag, frame.field_wake_max),
        frame.splat_mag,
        if frame.memory_ranked { "ᵣ" } else { "" },
        railed(frame.splat_mag, frame.splat_force_max),
        frame.goal_mag,
        railed(frame.goal_mag, frame.goal_force_max),
        frame.ocean_mag
    ));
    lines.push(format!(
        " LOGIT δ {}  vel {:.2}  visc {:.2}      HOOK Δ̄ {}  n {}",
        opt(frame.logit_delta, 1),
        frame.logit_velocity,
        frame.logit_viscosity,
        opt(frame.hook_delta_mean, 5),
        frame
            .hook_applications
            .map_or_else(|| "—".to_string(), |n| n.to_string())
    ));
    lines.push(format!(
        " QUAL  p {}  H {}  margin {}      scars {}",
        opt(frame.p_chosen, 2),
        opt(frame.entropy, 2),
        opt(frame.margin, 2),
        frame.scars
    ));
    lines.push(format!(
        " step {}/{} · --no-hud to disable · /tui for sliders",
        frame.step, frame.max_tokens
    ));
    lines.push(rule);

    debug_assert_eq!(
        lines.len(),
        BLOCK_LINES,
        "block height must match BLOCK_LINES"
    );
    // A line wider than the terminal wraps onto a second row, which would throw
    // off the fixed-height cursor math in `paint`. Clip instead.
    lines.iter().map(|line| clip(line, width)).collect()
}

/// Truncate to `width` characters (not bytes — the footer is full of ‖, δ, √).
fn clip(line: &str, width: usize) -> String {
    if line.chars().count() <= width {
        return line.to_string();
    }
    line.chars().take(width).collect()
}

/// Marks a force pinned to its own ceiling.
fn railed(magnitude: f32, ceiling: f32) -> &'static str {
    if ceiling > 0.0 && magnitude >= ceiling - 1e-3 {
        "⊤"
    } else {
        ""
    }
}

fn fmt(value: f32, places: usize) -> String {
    format!("{value:.places$}")
}

fn opt(value: Option<f32>, places: usize) -> String {
    value.map_or_else(|| "—".to_string(), |v| format!("{v:.places$}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> HudFrame {
        HudFrame {
            step: 214,
            max_tokens: 500,
            force_cap: 5.0,
            goal_force_scale: 0.15,
            temperature: 0.9,
            force_ramp_start: 0.20,
            force_ramp_tokens: 12,
            field_grad_blend: 0.15,
            baseline_norm: 452.1,
            steered_norm: 455.9,
            pullback: 0.9917,
            delta_h_norm: 3.81,
            clip_frac: 0.0,
            ramp: 1.0,
            eureka_boost: 1.0,
            cos_drift: Some(0.9971),
            grad_mag: 2.41,
            splat_mag: 0.88,
            goal_mag: 1.02,
            ocean_mag: 0.0,
            memory_ranked: true,
            field_wake_max: 40.0,
            splat_force_max: 60.0,
            goal_force_max: 60.0,
            logit_delta: Some(71.3),
            logit_velocity: 0.62,
            logit_viscosity: 0.41,
            hook_delta_mean: Some(0.0004),
            hook_applications: Some(18),
            p_chosen: Some(0.41),
            entropy: Some(2.81),
            margin: Some(0.18),
            scars: 137,
        }
    }

    #[test]
    fn block_height_is_fixed_across_widths_and_algo_presence() {
        let view = AlgoView::new(4.0, ModelType::Instruct);
        for width in [60u16, 80, 100, 200] {
            assert_eq!(
                render_block(&frame(), Some(&view), width).len(),
                BLOCK_LINES
            );
            assert_eq!(render_block(&frame(), None, width).len(), BLOCK_LINES);
        }
    }

    /// The erase/paint cycle moves a fixed number of lines. A line wider than
    /// the terminal wraps onto a second row, so the cursor would climb one row
    /// per token and smear the footer up through the text.
    #[test]
    fn no_line_ever_exceeds_the_terminal_width() {
        let view = AlgoView::new(27.0, ModelType::Instruct);
        for width in [MIN_WIDTH, 60, 72, 80, 100, 200] {
            let limit = width.max(MIN_WIDTH) as usize;
            for line in render_block(&frame(), Some(&view), width) {
                assert!(
                    line.chars().count() <= limit,
                    "width {width}: line is {} chars and would wrap:\n{line}",
                    line.chars().count()
                );
            }
            for line in render_block(&frame(), None, width) {
                assert!(
                    line.chars().count() <= limit,
                    "width {width}, no algo:\n{line}"
                );
            }
        }
    }

    /// Clipping must not split a multi-byte char — the footer is full of them.
    #[test]
    fn clipping_is_char_safe() {
        assert_eq!(clip("‖h‖ δ√σ", 3), "‖h‖");
        assert_eq!(clip("short", 99), "short");
    }

    #[test]
    fn live_over_prediction_reads_hot() {
        // 4B instruct predicts force_cap 3.12; the frame runs 5.0.
        let view = AlgoView::new(4.0, ModelType::Instruct);
        let rendered = render_block(&frame(), Some(&view), 100).join("\n");
        assert!(rendered.contains("σ→cap"));
        assert!(
            rendered.contains("HOT"),
            "expected HOT verdict in:\n{rendered}"
        );
    }

    #[test]
    fn missing_scalars_render_as_a_dash_not_zero() {
        let mut f = frame();
        f.cos_drift = None;
        f.p_chosen = None;
        f.hook_delta_mean = None;
        f.hook_applications = None;
        let rendered = render_block(&f, None, 100).join("\n");
        assert!(rendered.contains("cos(h_t,h_t-1) —"));
        assert!(rendered.contains("p —"));
        assert!(rendered.contains("Δ̄ —  n —"));
        assert!(
            rendered.contains("--model-params"),
            "should prompt for size"
        );
    }

    /// A force pinned to its ceiling is a constant shove, not steering — the
    /// readout has to make that visible or you tune the wrong knob for an hour.
    #[test]
    fn a_force_sitting_on_its_ceiling_is_marked() {
        let mut f = frame();
        f.grad_mag = 40.0; // exactly field_wake_max
        let rendered = render_block(&f, None, 100).join("\n");
        assert!(rendered.contains("F_g 40.00⊤"), "got:\n{rendered}");

        f.grad_mag = 12.0; // well under
        let rendered = render_block(&f, None, 100).join("\n");
        assert!(rendered.contains("F_g 12.00 "), "got:\n{rendered}");

        // A disabled ceiling (0) must never read as saturated.
        assert_eq!(railed(0.0, 0.0), "");
        assert_eq!(railed(99.0, 0.0), "");
    }

    #[test]
    fn clip_fraction_renders_as_a_percentage() {
        let mut f = frame();
        f.clip_frac = 0.427;
        let rendered = render_block(&f, None, 100).join("\n");
        assert!(rendered.contains("clip 42.7%"), "got:\n{rendered}");
    }

    #[test]
    fn hud_is_inert_when_disabled() {
        let mut hud = Hud::new(false, 1, None);
        assert!(!hud.is_enabled());
        assert!(hud.update(frame()).is_ok());
        assert!(hud.finish().is_ok());
    }
}
