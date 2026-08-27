//! Full-screen REPL: chat, live scalars, and live sliders in one view.
//!
//! `--tui` opens this instead of the plain streaming chat. Everything is visible
//! at once — output on top, the √-law scalars and the physics sliders side by
//! side below, prompt at the bottom:
//!
//! ```text
//! ┌─ output ──────────────────────────────────────────────┐
//! │ gemma3> …the field pulls it toward▌                   │
//! ├─ scalars ─────────────────────┬─ sliders ─────────────┤
//! │ σ→cap   5.00 / 3.12   HOT     │ ▶ cap  [██████····]   │
//! ├───────────────────────────────┴───────────────────────┤
//! │ you> ▌                                                │
//! └───────────────────────────────────────────────────────┘
//! ```
//!
//! Sliders stay live *during* generation: nudging `residual.cap` between tokens
//! shows up in `clip` and `cos` on the very next one. That is the whole point —
//! the scaling algo is a thing you feel, not a thing you restart for.
//!
//! This module owns no physics. It renders what it is handed and reports the
//! control edits the operator made, exactly like `tui::run_slider_tui`.

use crate::algo_scale::{self, Zone};
use crate::hud::{AlgoView, HudFrame};
use crate::tui::Slider;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, queue};
use std::io::{self, Stdout, Write};
use std::time::Duration;

/// Content rows in the scalars/sliders band.
const PANEL_ROWS: usize = 6;
/// Borders + panel + prompt: everything that is not the output pane.
const CHROME_ROWS: usize = PANEL_ROWS + 5;
const MIN_ROWS: u16 = (CHROME_ROWS + 3) as u16;
const MIN_COLS: u16 = 64;

/// Control edits made while tokens were flowing. The caller applies them —
/// this module never touches the engine.
#[derive(Debug, Default)]
pub struct Edits {
    pub step_abort: bool,
    pub sets: Vec<(String, f32)>,
}

/// Which pane the keyboard is driving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Input,
    Sliders,
}

/// Restores the terminal even if the caller unwinds.
struct ScreenGuard;

impl ScreenGuard {
    fn enter(stdout: &mut Stdout) -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error);
        }
        Ok(Self)
    }
}

impl Drop for ScreenGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(stdout, Show, LeaveAlternateScreen, ResetColor);
        let _ = terminal::disable_raw_mode();
    }
}

pub struct App {
    algo: Option<AlgoView>,
    /// Logical output lines; the last one is still being appended to.
    lines: Vec<String>,
    input: String,
    sliders: Vec<Slider>,
    selected: usize,
    slider_scroll: usize,
    frame: Option<HudFrame>,
    status: String,
    focus: Focus,
    stdout: Stdout,
    _guard: ScreenGuard,
}

impl App {
    pub fn new(algo: Option<AlgoView>, sliders: Vec<Slider>) -> io::Result<Self> {
        let mut stdout = io::stdout();
        let guard = ScreenGuard::enter(&mut stdout)?;
        Ok(Self {
            algo,
            lines: vec![String::new()],
            input: String::new(),
            sliders,
            selected: 0,
            slider_scroll: 0,
            frame: None,
            status: "Tab switches panes · ↑↓ pick slider · ←→ adjust".to_string(),
            focus: Focus::Input,
            stdout,
            _guard: guard,
        })
    }

    /// Terminal too small to lay out: the caller should fall back to plain chat.
    pub fn fits() -> bool {
        terminal::size().is_ok_and(|(cols, rows)| cols >= MIN_COLS && rows >= MIN_ROWS)
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn set_frame(&mut self, frame: HudFrame) {
        self.frame = Some(frame);
    }

    /// Append output, honouring embedded newlines.
    pub fn push(&mut self, text: &str) {
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.lines.push(String::new());
            }
            if let Some(last) = self.lines.last_mut() {
                last.push_str(part);
            }
        }
        // Keep the buffer bounded; the transcript file is the real record.
        const MAX_LINES: usize = 2000;
        if self.lines.len() > MAX_LINES {
            self.lines.drain(..self.lines.len() - MAX_LINES);
        }
    }

    pub fn push_line(&mut self, text: &str) {
        self.push(text);
        self.push("\n");
    }

    /// Refresh the slider values from the engine after the caller applied edits.
    pub fn sync_sliders(&mut self, sliders: Vec<Slider>) {
        self.selected = self.selected.min(sliders.len().saturating_sub(1));
        self.sliders = sliders;
    }

    /// Block until the operator submits a line. `None` means quit.
    pub fn read_prompt(&mut self) -> io::Result<Option<String>> {
        loop {
            self.draw()?;
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                continue;
            }
            if is_quit(key) {
                return Ok(None);
            }
            if key.code == KeyCode::Tab {
                self.toggle_focus();
                continue;
            }
            match self.focus {
                Focus::Sliders => {
                    // Edits are collected and returned to the caller to apply.
                    if let Some(edit) = self.slider_key(key) {
                        return Ok(Some(format!("/set {} {}", edit.0, edit.1)));
                    }
                }
                Focus::Input => match key.code {
                    // A real terminal sends \r for Enter, but piped input sends
                    // \n and crossterm passes it through as a plain char.
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        let line = self.input.trim().to_string();
                        self.input.clear();
                        if line.is_empty() {
                            return Ok(None);
                        }
                        return Ok(Some(line));
                    }
                    KeyCode::Backspace => {
                        self.input.pop();
                    }
                    KeyCode::Char(c) => self.input.push(c),
                    _ => {}
                },
            }
        }
    }

    /// Non-blocking key drain for use between tokens. Returns the control edits
    /// the operator made so the caller can apply them to the live engine.
    pub fn poll_edits(&mut self) -> io::Result<Edits> {
        let mut edits = Edits::default();
        while event::poll(Duration::from_millis(0))? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                continue;
            }
            if key.code == KeyCode::Esc || is_quit(key) {
                edits.step_abort = true;
                self.set_status("turn aborted");
                continue;
            }
            if key.code == KeyCode::Tab {
                self.toggle_focus();
                continue;
            }
            if let Some(edit) = self.slider_key(key) {
                edits.sets.push(edit);
            }
        }
        Ok(edits)
    }

    fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Input => Focus::Sliders,
            Focus::Sliders => Focus::Input,
        };
        self.status = match self.focus {
            Focus::Input => "typing · Tab for sliders".to_string(),
            Focus::Sliders => "sliders · ←→ adjust (Shift ×10) · Tab back".to_string(),
        };
    }

    /// Interpret a key as a slider move. Returns the edit to apply, if any.
    fn slider_key(&mut self, key: KeyEvent) -> Option<(String, f32)> {
        if self.sliders.is_empty() {
            return None;
        }
        let fast = key
            .modifiers
            .intersects(KeyModifiers::SHIFT | KeyModifiers::CONTROL);
        let ticks = if fast { 10.0 } else { 1.0 };
        let changed = match key.code {
            KeyCode::Up => {
                self.selected = (self.selected + self.sliders.len() - 1) % self.sliders.len();
                false
            }
            KeyCode::Down => {
                self.selected = (self.selected + 1) % self.sliders.len();
                false
            }
            KeyCode::Left => self.sliders[self.selected].nudge(-ticks),
            KeyCode::Right => self.sliders[self.selected].nudge(ticks),
            _ => return None,
        };
        if !changed {
            return None;
        }
        let slider = &self.sliders[self.selected];
        self.status = format!("{} = {}", slider.name, slider.value_label());
        Some((slider.name.clone(), slider.value))
    }

    pub fn draw(&mut self) -> io::Result<()> {
        let (cols, rows) = terminal::size().unwrap_or((100, 40));
        if cols < MIN_COLS || rows < MIN_ROWS {
            return self.draw_too_small(cols, rows);
        }
        let width = cols as usize;
        let output_rows = rows as usize - CHROME_ROWS;
        let inner = width - 2;
        // Scalars get the wider half; sliders need less to stay readable.
        let left = inner * 3 / 5;
        let right = inner - left - 1;

        let body = wrap_tail(&self.lines, inner, output_rows);
        let scalars = scalar_rows(self.frame.as_ref(), self.algo.as_ref(), left);
        self.clamp_slider_scroll();
        let sliders = slider_rows(
            &self.sliders,
            self.selected,
            self.slider_scroll,
            self.algo.as_ref(),
            right,
            self.focus == Focus::Sliders,
        );

        queue!(self.stdout, MoveTo(0, 0), Clear(ClearType::All))?;
        self.line(0, &rule_top(width, "output"))?;
        for (index, text) in body.iter().enumerate() {
            self.line(1 + index as u16, &framed(text, inner))?;
        }
        let mid = 1 + output_rows as u16;
        self.line(mid, &rule_split(width, left, "scalars", "sliders"))?;
        for row in 0..PANEL_ROWS {
            let l = scalars.get(row).map(String::as_str).unwrap_or("");
            let r = sliders.get(row).map(String::as_str).unwrap_or("");
            let text = format!("│{}│{}│", pad(l, left), pad(r, right));
            self.line(mid + 1 + row as u16, &text)?;
        }
        let foot = mid + 1 + PANEL_ROWS as u16;
        self.line(foot, &rule_join(width, left))?;
        let caret = if self.focus == Focus::Input {
            "▌"
        } else {
            " "
        };
        let prompt = format!("you> {}{caret}", self.input);
        self.line(foot + 1, &framed(&prompt, inner))?;
        self.line(foot + 2, &rule_bottom(width, &self.status))?;
        self.stdout.flush()
    }

    fn clamp_slider_scroll(&mut self) {
        if self.selected < self.slider_scroll {
            self.slider_scroll = self.selected;
        } else if self.selected >= self.slider_scroll + PANEL_ROWS {
            self.slider_scroll = self.selected + 1 - PANEL_ROWS;
        }
    }

    fn draw_too_small(&mut self, cols: u16, rows: u16) -> io::Result<()> {
        queue!(
            self.stdout,
            MoveTo(0, 0),
            Clear(ClearType::All),
            Print(format!(
                "terminal {cols}x{rows} is too small — need at least {MIN_COLS}x{MIN_ROWS}"
            ))
        )?;
        self.stdout.flush()
    }

    fn line(&mut self, row: u16, text: &str) -> io::Result<()> {
        queue!(
            self.stdout,
            MoveTo(0, row),
            SetForegroundColor(Color::DarkGrey),
            Print(text),
            ResetColor,
            SetAttribute(Attribute::Reset)
        )
    }
}

fn is_quit(key: KeyEvent) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
}

/// Wrap logical lines to `width` and keep the last `rows` display lines.
fn wrap_tail(lines: &[String], width: usize, rows: usize) -> Vec<String> {
    let mut display: Vec<String> = Vec::new();
    for line in lines {
        if line.is_empty() {
            display.push(String::new());
            continue;
        }
        let mut current = String::new();
        let mut count = 0usize;
        for ch in line.chars() {
            current.push(ch);
            count += 1;
            if count == width {
                display.push(std::mem::take(&mut current));
                count = 0;
            }
        }
        if !current.is_empty() {
            display.push(current);
        }
    }
    if display.len() > rows {
        display.drain(..display.len() - rows);
    }
    while display.len() < rows {
        display.push(String::new());
    }
    display
}

fn scalar_rows(frame: Option<&HudFrame>, algo: Option<&AlgoView>, width: usize) -> Vec<String> {
    let Some(frame) = frame else {
        return vec![" (no tokens yet — send a prompt)".to_string()];
    };
    let mut rows = Vec::with_capacity(PANEL_ROWS);
    match algo {
        Some(view) => {
            rows.push(format!(
                " σ→cap  {:>7.2} /{:>7.2}  {}",
                frame.force_cap,
                view.predicted.force_cap,
                algo_scale::zone_ratio(frame.force_cap, view.predicted.force_cap).as_str()
            ));
            rows.push(format!(
                " θ→goal {:>7.3} /{:>7.3}  {}",
                frame.goal_force_scale,
                view.predicted.goal_force_scale,
                algo_scale::zone_ratio(frame.goal_force_scale, view.predicted.goal_force_scale)
                    .as_str()
            ));
        }
        None => {
            rows.push(format!(
                " σ→cap  {:>7.2}   (no model size)",
                frame.force_cap
            ));
            rows.push(format!(
                " θ→goal {:>7.3}   --model-params <B>",
                frame.goal_force_scale
            ));
        }
    }
    rows.push(format!(
        " ‖h‖ {:.1}→{:.1}  pull ×{:.4}  clip {:.1}%",
        frame.baseline_norm,
        frame.steered_norm,
        frame.pullback,
        frame.clip_frac * 100.0
    ));
    rows.push(format!(
        " cos {}  ‖δh‖ {:.2}  ramp ×{:.2}",
        opt(frame.cos_drift, 4),
        frame.delta_h_norm,
        frame.ramp
    ));
    rows.push(format!(
        " F_g {:.1}{}  F_s {:.2}{}{}  F_a {:.1}{}  F_o {:.1}",
        frame.grad_mag,
        railed(frame.grad_mag, frame.field_wake_max),
        frame.splat_mag,
        if frame.memory_ranked { "ᵣ" } else { "" },
        railed(frame.splat_mag, frame.splat_force_max),
        frame.goal_mag,
        railed(frame.goal_mag, frame.goal_force_max),
        frame.ocean_mag
    ));
    rows.push(format!(
        " step {}/{}  scars {}  H {}",
        frame.step,
        frame.max_tokens,
        frame.scars,
        opt(frame.entropy, 2)
    ));
    rows.into_iter().map(|row| clip(&row, width)).collect()
}

/// Knobs the √-law predicts a value for, so a slider can show its own verdict.
fn predicted_for(name: &str, algo: &AlgoView) -> Option<f32> {
    match name {
        "residual.cap" => Some(algo.predicted.force_cap),
        "residual.goal" => Some(algo.predicted.goal_force_scale),
        "residual.field" => Some(algo.predicted.field_wake_scale),
        "residual.field_max" => Some(algo.predicted.field_wake_max),
        "residual.splat_max" => Some(algo.predicted.splat_force_max),
        "residual.goal_max" => Some(algo.predicted.goal_force_max),
        "sample.temp" => Some(algo.predicted.temperature),
        _ => None,
    }
}

fn slider_rows(
    sliders: &[Slider],
    selected: usize,
    scroll: usize,
    algo: Option<&AlgoView>,
    width: usize,
    focused: bool,
) -> Vec<String> {
    if sliders.is_empty() {
        return vec![" (no live controls)".to_string()];
    }
    // name + value + verdict take a fixed share; the bar gets what is left.
    let bar_width = width.saturating_sub(30).clamp(6, 18);
    sliders
        .iter()
        .enumerate()
        .skip(scroll)
        .take(PANEL_ROWS)
        .map(|(index, slider)| {
            let marker = if index == selected {
                if focused {
                    "▶"
                } else {
                    "·"
                }
            } else {
                " "
            };
            let filled = (slider.fraction() * bar_width as f32).round() as usize;
            let bar: String = (0..bar_width)
                .map(|i| if i < filled { '█' } else { '·' })
                .collect();
            let verdict = algo
                .and_then(|view| predicted_for(&slider.name, view))
                .map(
                    |predicted| match algo_scale::zone_ratio(slider.value, predicted) {
                        Zone::In => " IN".to_string(),
                        other => format!(" {}", other.as_str()),
                    },
                )
                .unwrap_or_default();
            clip(
                &format!(
                    "{marker}{:<14}[{bar}]{:>8}{verdict}",
                    short_name(&slider.name),
                    slider.value_label()
                ),
                width,
            )
        })
        .collect()
}

/// `residual.cap` → `cap`: the group is already the column header.
fn short_name(name: &str) -> &str {
    name.split_once('.').map_or(
        name,
        |(group, rest)| {
            if group == "residual" {
                rest
            } else {
                name
            }
        },
    )
}

/// Marks a force pinned to its own ceiling — `⊤` means saturated.
fn railed(magnitude: f32, ceiling: f32) -> &'static str {
    if ceiling > 0.0 && magnitude >= ceiling - 1e-3 {
        "⊤"
    } else {
        ""
    }
}

fn opt(value: Option<f32>, places: usize) -> String {
    value.map_or_else(|| "—".to_string(), |v| format!("{v:.places$}"))
}

fn clip(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        text.to_string()
    } else {
        text.chars().take(width).collect()
    }
}

fn pad(text: &str, width: usize) -> String {
    let text = clip(text, width);
    let short = width - text.chars().count();
    format!("{text}{}", " ".repeat(short))
}

fn framed(text: &str, inner: usize) -> String {
    format!("│{}│", pad(text, inner))
}

fn rule_top(width: usize, label: &str) -> String {
    let head = format!("┌─ {label} ");
    let fill = width.saturating_sub(head.chars().count() + 1);
    format!("{head}{}┐", "─".repeat(fill))
}

fn rule_bottom(width: usize, status: &str) -> String {
    let head = format!("└─ {status} ");
    let fill = width.saturating_sub(head.chars().count() + 1);
    format!("{head}{}┘", "─".repeat(fill))
}

/// `├` + `left` cells + `┬` + `right` cells + `┤` — must total exactly `width`,
/// matching the content rows, or the box tears.
fn rule_split(width: usize, left: usize, left_label: &str, right_label: &str) -> String {
    let right = width.saturating_sub(left + 3);
    let head = clip(&format!("─ {left_label} "), left);
    let tail = clip(&format!("─ {right_label} "), right);
    let left_fill = left - head.chars().count();
    let right_fill = right - tail.chars().count();
    format!(
        "├{head}{}┬{tail}{}┤",
        "─".repeat(left_fill),
        "─".repeat(right_fill)
    )
}

fn rule_join(width: usize, left: usize) -> String {
    let right = width.saturating_sub(left + 3);
    format!("├{}┴{}┤", "─".repeat(left), "─".repeat(right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo_scale::ModelType;

    fn frame() -> HudFrame {
        HudFrame {
            step: 12,
            max_tokens: 200,
            force_cap: 5.0,
            goal_force_scale: 0.15,
            baseline_norm: 131.1,
            steered_norm: 129.9,
            pullback: 1.0092,
            delta_h_norm: 1.92,
            clip_frac: 0.004,
            ramp: 1.0,
            cos_drift: Some(0.8879),
            grad_mag: 40.0,
            splat_mag: 0.01,
            goal_mag: 43.17,
            ocean_mag: 8.1,
            memory_ranked: true,
            entropy: Some(2.04),
            scars: 18,
            ..HudFrame::default()
        }
    }

    fn sliders() -> Vec<Slider> {
        vec![
            Slider::live("residual.cap", 5.0, 0.0, 20.0),
            Slider::live("residual.goal", 0.15, 0.0, 2.0),
            Slider::live("sample.temp", 0.9, 0.0, 2.0),
        ]
    }

    /// Every drawn row must be exactly the terminal width, or the box tears.
    #[test]
    fn every_border_row_is_exactly_the_terminal_width() {
        for width in [64usize, 80, 100, 173] {
            let inner = width - 2;
            let left = inner * 3 / 5;
            let right = inner - left - 1;
            let rows = [
                rule_top(width, "output"),
                rule_split(width, left, "scalars", "sliders"),
                rule_join(width, left),
                rule_bottom(width, "typing · Tab for sliders"),
                framed("you> hello", inner),
                format!("│{}│{}│", pad("scalar", left), pad("slider", right)),
            ];
            for row in rows {
                assert_eq!(
                    row.chars().count(),
                    width,
                    "width {width}: row is {} chars:\n{row}",
                    row.chars().count()
                );
            }
        }
    }

    #[test]
    fn panel_content_never_overflows_its_column() {
        let view = AlgoView::new(4.0, ModelType::Instruct);
        for width in [20usize, 30, 40, 60] {
            for row in scalar_rows(Some(&frame()), Some(&view), width) {
                assert!(row.chars().count() <= width, "scalars @{width}: {row}");
            }
            for row in slider_rows(&sliders(), 0, 0, Some(&view), width, true) {
                assert!(row.chars().count() <= width, "sliders @{width}: {row}");
            }
        }
    }

    #[test]
    fn output_pane_always_fills_exactly_the_rows_it_is_given() {
        let lines = vec!["short".to_string(), "x".repeat(250), String::new()];
        for rows in [1usize, 5, 12] {
            let body = wrap_tail(&lines, 40, rows);
            assert_eq!(body.len(), rows);
            for line in body {
                assert!(line.chars().count() <= 40);
            }
        }
    }

    /// A long line must show its tail — that is where the newest tokens are.
    #[test]
    fn wrapping_keeps_the_newest_output() {
        let lines = vec![format!("{}TAIL", "a".repeat(80))];
        let body = wrap_tail(&lines, 40, 2);
        assert!(body.last().unwrap().ends_with("TAIL"), "got {body:?}");
    }

    #[test]
    fn sliders_show_the_algo_verdict_for_knobs_the_law_predicts() {
        let view = AlgoView::new(4.0, ModelType::Instruct);
        let rows = slider_rows(&sliders(), 0, 0, Some(&view), 44, true);
        // cap 5.00 against a predicted 3.12 is hot.
        assert!(rows[0].contains("HOT"), "{rows:?}");
        // A knob with no prediction gets no verdict rather than a made-up one.
        let plain = slider_rows(
            &[Slider::live("gov.brake", 3.0, 0.0, 10.0)],
            0,
            0,
            Some(&view),
            44,
            true,
        );
        assert!(
            !plain[0].contains("IN") && !plain[0].contains("HOT"),
            "{plain:?}"
        );
    }

    #[test]
    fn short_name_strips_only_the_residual_group() {
        assert_eq!(short_name("residual.cap"), "cap");
        assert_eq!(short_name("sample.temp"), "sample.temp");
        assert_eq!(short_name("gov.brake"), "gov.brake");
    }
}
